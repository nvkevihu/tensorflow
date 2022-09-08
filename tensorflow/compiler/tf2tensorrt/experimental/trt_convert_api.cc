/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/tf2tensorrt/experimental/trt_convert_api.h"

#include <iostream>
#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/cc/tools/freeze_saved_model.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_lru_cache.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/clusters/utils.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/grappler_item_builder.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

namespace {

// Creates and provisions a new cluster. The caller must call Shutdown before
// the cluster is destroyed.
Status NewCluster(grappler::Cluster** cluster) {
  int num_cpu_cores = grappler::GetNumAvailableLogicalCPUCores();
  int num_gpus = grappler::GetNumAvailableGPUs();
  int timeout_s = 60 * 10;
  *cluster = new grappler::SingleMachine(timeout_s, num_cpu_cores, num_gpus);
  (*cluster)->DisableDetailedStats(true);
  (*cluster)->AllowSoftPlacement(true);
  (*cluster)->SetNumWarmupSteps(10);
  TF_RETURN_IF_ERROR((*cluster)->Provision());
  return Status::OK();
}

Status RunGrappler(const MetaGraphDef& meta_graph_def,
                   const std::vector<std::string>& input_names,
                   const std::vector<std::string>& output_names,
                   const ConfigProto& config_proto, grappler::Cluster* cluster,
                   GraphDef* out_graph_def) {
  grappler::ItemConfig item_config;

  for (const std::string& name : input_names) {
    item_config.feed_nodes.insert(name);
  }
  for (const std::string& name : output_names) {
    item_config.fetch_nodes.insert(name);
  }

  std::unique_ptr<grappler::GrapplerItem> item =
      grappler::GrapplerItemFromMetaGraphDef("tf_graph", meta_graph_def,
                                             item_config);
  if (!item) {
    return tensorflow::errors::Internal(
        "Failed to create grappler item from MetaGraphDef.");
  }

  tensorflow::DeviceBase* cpu_device = nullptr;
  TF_RETURN_IF_ERROR(grappler::RunMetaOptimizer(
      std::move(*item), config_proto, cpu_device, cluster, out_graph_def));
  VLOG(2) << "Grappler finished\n";
  return Status::OK();
}

Status GetTrtRewriterConfig(const TrtConversionParams& params,
                            RewriterConfig* opt_config) {
  opt_config->set_meta_optimizer_iterations(tensorflow::RewriterConfig::ONE);
  opt_config->set_min_graph_nodes(-1);  // do not skip small graphs

  // Turn off remapping.
  opt_config->set_remapping(RewriterConfig_Toggle::RewriterConfig_Toggle_OFF);

  // If the graph has QDQ nodes, then we need to disable folding of the
  // QDQ with constants. Otherwise, the conversion will not work corectly.
  // Ideally, we do this after segmentation and outlining of TRT regions to
  // functions, but we currently lack that capability. Disabling QDQ-const
  // folding doesn't matter if you don't have QDQ nodes, so we always enable
  // this.
  opt_config->set_experimental_disable_folding_quantization_emulation(
      IS_TRT_VERSION_GE(8, 0, 0, 0));

  // Initial transformations before TensorRTOptimizer is called
  opt_config->add_optimizers("pruning");
  opt_config->add_optimizers("debug_stripper");
  opt_config->add_optimizers("layout");
  opt_config->add_optimizers("dependency");
  opt_config->add_optimizers("constfold");
  opt_config->add_optimizers("common_subgraph_elimination");

  // Parameters for TensorRTOptimizer
  auto trt_optimizer = opt_config->add_custom_optimizers();
  trt_optimizer->set_name("TensorRTOptimizer");

  auto trt_parameter_map = trt_optimizer->mutable_parameter_map();
  (*trt_parameter_map)["dla_core"].set_i(-1);
  (*trt_parameter_map)["dla_fallback_layers"].set_i(-1);
  (*trt_parameter_map)["enable_sparse_compute"].set_b(true);
  (*trt_parameter_map)["is_dynamic_op"].set_b(true);
  (*trt_parameter_map)["minimum_segment_size"].set_i(
      params.minimum_segment_size);
  std::string prec_string;
  TF_RETURN_IF_ERROR(
      TrtPrecisionModeToName(params.precision_mode, &prec_string));
  (*trt_parameter_map)["precision_mode"].set_s(prec_string);
  (*trt_parameter_map)["max_workspace_size_bytes"].set_i(
      params.max_workspace_size_bytes);
  (*trt_parameter_map)["max_cached_engines"].set_i(params.maximum_cached_engines);
  (*trt_parameter_map)["use_calibration"].set_b(params.use_calibration);
  (*trt_parameter_map)["profile_strategy"].set_s(
      ProfileStrategyToName(params.dynamic_shape_profile_strategy));
  (*trt_parameter_map)["use_implicit_batch"].set_b(!params.use_dynamic_shape);
  (*trt_parameter_map)["allow_build_at_runtime"].set_b(
      params.allow_build_at_runtime);

  opt_config->add_custom_optimizers()->set_name("constfold");
  return Status::OK();
}

// Runs TRTOptimizer grappler pass.
Status RunTfTrt(const MetaGraphDef& meta_graph_def,
                const std::vector<std::string>& input_names,
                const std::vector<std::string>& output_names,
                const RewriterConfig& rewriter_config,
                GraphDef* segmented_graph_def) {
  ConfigProto config_proto;
  config_proto.mutable_graph_options()->mutable_rewrite_options()->CopyFrom(
      rewriter_config);

  VLOG(4) << "Setting up Grappler parameters\n" << config_proto.DebugString();
  std::unique_ptr<grappler::Cluster> cluster;
  grappler::Cluster* p_cluster;
  mutex mu_cluster;  // There can be only one provisioned cluster per process.
  mutex_lock lock(mu_cluster);
  TF_RETURN_IF_ERROR(NewCluster(&p_cluster));
  cluster.reset(p_cluster);
  TF_RETURN_IF_ERROR(RunGrappler(meta_graph_def, input_names, output_names,
                                 config_proto, cluster.get(),
                                 segmented_graph_def));
  TF_RETURN_IF_ERROR(cluster->Shutdown());
  return Status::OK();
}

// Sets the _profile_generation mode attribute of all TRTEngineOp nodes in the
// graph to mode.
Status SetProfileGenerationMode(GraphDef* graph_def, bool mode) {
  VLOG(3) << "Setting _profile_generation_mode=" << mode;
  std::string op{"TRTEngineOp"};
  for (auto& node : *(graph_def->mutable_node())) {
    if (!op.compare(node.op())) {
      auto* attr = node.mutable_attr();
      AttrValue profile_generation_mode;
      profile_generation_mode.set_b(mode);
      (*attr)["_profile_generation_mode"] = profile_generation_mode;
    }
  }
  return Status::OK();
}

Status ImportGraphDefToSession(Session* session, const GraphDef& graph_def,
                               const string& prefix) {
  ImportGraphDefOptions opts;
  opts.prefix = prefix;
  Graph graph(OpRegistry::Global());
  TF_RETURN_IF_ERROR(ImportGraphDef(opts, graph_def, &graph, nullptr));
  GraphDef new_graph_def;
  graph.ToGraphDef(&new_graph_def);
  TF_RETURN_IF_ERROR(session->Extend(new_graph_def));
  return Status::OK();
}

// Returns configuration used during the build step session run.
tensorflow::SessionOptions GetSessionConfg() {
  // We also need to disable constant folding because we already ran constant
  // folding and may have prevented quantization operation folding on purpose.
  tensorflow::SessionOptions opts;
  auto* rewriter_opts =
      opts.config.mutable_graph_options()->mutable_rewrite_options();
  rewriter_opts->set_experimental_disable_folding_quantization_emulation(true);

  // It seems  that we need to disable the optimizer entirely to prevent the
  // folding.
  rewriter_opts->set_disable_meta_optimizer(true);
  return opts;
}

Status RunSession(Session* session, const std::vector<std::string>& input_names,
                  const std::vector<std::string>& output_names,
                  const std::vector<Tensor>& input_tensors,
                  std::string prefix = "") {
  TRT_ENSURE(!input_names.empty());
  TRT_ENSURE(!output_names.empty());
  TRT_ENSURE(!input_tensors.empty());

  std::vector<std::pair<std::string, tensorflow::Tensor>> input_pairs;
  std::vector<std::string> prefixed_output_names;
  auto prefixed_name = [](std::string prefix, std::string name) {
    return prefix.size() > 0 ? absl::StrJoin({prefix, name}, "/") : name;
  };
  for (int i = 0; i < input_names.size(); i++) {
    input_pairs.push_back(
        {prefixed_name(prefix, input_names.at(i)), input_tensors.at(i)});
  }
  for (int i = 0; i < output_names.size(); i++) {
    prefixed_output_names.push_back(prefixed_name(prefix, output_names.at(i)));
  }
  std::vector<tensorflow::Tensor> output_tensors;
  for (int i = 0; i < output_names.size(); i++) {
    output_tensors.push_back({});
  }
  VLOG(3) << "TF-TRT Build mode: running inference\n";
  TF_RETURN_IF_ERROR(
      session->Run(input_pairs, prefixed_output_names, {}, &output_tensors));
  return Status::OK();
}

// TODO
Status IsFrozenGraph(const GraphDef& graph) {
  return Status::OK();
}

// TODO
Status CheckConversionParams(const TrtConversionParams& conversion_params) {
  return Status::OK();
}

}  // namespace

// static
StatusOr<std::unique_ptr<TrtGraphConverter>> TrtGraphConverter::Create(
    const GraphDef& frozen_graph_def,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const TrtConversionParams& conversion_params) {
  std::unique_ptr<TrtGraphConverter> converter = absl::WrapUnique(new TrtGraphConverter(
      frozen_graph_def, input_names, output_names, conversion_params));
  TF_RETURN_IF_ERROR(converter->Validate());
  return converter;
}

TrtGraphConverter::TrtGraphConverter(
    const GraphDef& frozen_graph_def,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const TrtConversionParams& conversion_params)
    : frozen_graph_def_(frozen_graph_def),
      input_names_(input_names),
      output_names_(output_names),
      conversion_params_(conversion_params) {}

Status TrtGraphConverter::Validate() {
  TF_RETURN_IF_ERROR(IsFrozenGraph(frozen_graph_def_));
  TF_RETURN_IF_ERROR(CheckConversionParams(conversion_params_));
  return Status::OK();
}

StatusOr<GraphDef> TrtGraphConverter::Convert(const std::vector<std::vector<tensorflow::Tensor>>& inputs) {
  MetaGraphDef meta_graph;
  meta_graph.mutable_graph_def()->CopyFrom(frozen_graph_def_);

  RewriterConfig rewriter_config;
  TF_RETURN_IF_ERROR(
      GetTrtRewriterConfig(conversion_params_, &rewriter_config));

  TF_RETURN_IF_ERROR(RunTfTrt(meta_graph, input_names_, output_names_,
                              rewriter_config, &segmented_graph_def_));

  VLOG(1) << "TF-TRT conversion finished";
  // TODO: Calibration

  return segmented_graph_def_;
}

Status TrtGraphConverter::Build(const std::vector<std::vector<tensorflow::Tensor>>& inputs) {
  // The TRTOptimization pass has inserted placeholder TRTEngineOps. Here we
  // trigger conversion by inferring the graph.
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(GetSessionConfg()));
  if (!session.get()) {
    return errors::Internal("Failed to create build session");
  }

  VLOG(2) << "Building the model";
  bool need_collect_profiles = conversion_params_.use_dynamic_shape && inputs.size() > 1;
  if (need_collect_profiles) {
    TF_RETURN_IF_ERROR(SetProfileGenerationMode(&segmented_graph_def_, true));
  }
  TF_RETURN_IF_ERROR(session->Create(segmented_graph_def_));
  std::string prefix = "";
  if (need_collect_profiles) {
    for (auto const& input : inputs) {
      TF_RETURN_IF_ERROR(RunSession(session.get(), input_names_, output_names_, input));
    }
    prefix = "TrtBuildStep";
    TF_RETURN_IF_ERROR(SetProfileGenerationMode(&segmented_graph_def_, false));
    VLOG(3) << "Importing graph with _profile_generation_mode disabled";
    TF_RETURN_IF_ERROR(
        ImportGraphDefToSession(session.get(), segmented_graph_def_, prefix));
  }
  TF_RETURN_IF_ERROR(
      RunSession(session.get(), input_names_, output_names_, *inputs.begin(), prefix));
  // TODO: Calibration

  return Status::OK();
}

// TODO
void TrtGraphConverter::Summary(uint line_length, bool detailed, std::ostream& ostream) {
}


}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
