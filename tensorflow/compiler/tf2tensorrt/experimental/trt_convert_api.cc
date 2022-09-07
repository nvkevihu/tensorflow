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

// TODO
Status IsFrozenGraph(const GraphDef& graph) {
  return Status::OK();
}

// TODO
Status CheckConversionParams(const TrtConversionParams& conversion_params) {
  return Status::OK();
}

// static
StatusOr<std::unique_ptr<TrtGraphConverter>> TrtGraphConverter::Create(
    const GraphDef& frozen_graph_def,
    const TrtConversionParams& conversion_params) {
  std::unique_ptr<TrtGraphConverter> converter = absl::WrapUnique(new TrtGraphConverter(
      frozen_graph_def, conversion_params));
  TF_RETURN_IF_ERROR(converter->Validate());
  return converter;
}

TrtGraphConverter::TrtGraphConverter(
    const GraphDef& frozen_graph_def,
    const TrtConversionParams& conversion_params)
    : frozen_graph_def_(frozen_graph_def),
      conversion_params_(conversion_params) {}

Status TrtGraphConverter::Validate() {
  TF_RETURN_IF_ERROR(IsFrozenGraph(frozen_graph_def_));
  TF_RETURN_IF_ERROR(CheckConversionParams(conversion_params_));
  return Status::OK();
}

// TODO
Status TrtGraphConverter::Convert(const std::vector<std::vector<tensorflow::Tensor>>& inputs) {
  return Status::OK();
}

// TODO
StatusOr<GraphDef> TrtGraphConverter::Build(const std::vector<std::vector<tensorflow::Tensor>>& inputs) {
  return Status::OK();
}

// TODO
void TrtGraphConverter::Summary(uint line_length, bool detailed, std::ostream& ostream) {
}


}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
