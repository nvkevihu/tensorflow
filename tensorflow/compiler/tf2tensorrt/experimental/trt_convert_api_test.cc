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

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/compiler/tf2tensorrt/experimental/trt_convert_api.h"

#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace tensorrt {

// TODO
class TrtGraphConverterTest
    : public ::testing::TestWithParam<TrtConversionParams> {
 protected:
  TrtGraphConverterTest()
      : params_(GetParam()) {
    Reset();
  }

  void Reset() {
    GraphDef graph;
    converter_ = std::move(TrtGraphConverter::Create(
        graph, params_).ValueOrDie());
  }

  TrtConversionParams params_;
  std::unique_ptr<TrtGraphConverter> converter_;
};

INSTANTIATE_TEST_CASE_P(
    TrtGraphConverterTestInstantiation, TrtGraphConverterTest,
    ::testing::Values(TrtConversionParams()));

TEST_P(TrtGraphConverterTest, Basic) {
  TF_ASSERT_OK(converter_->Convert({}));
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
