/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cstdio>
#include <memory>
#include <string>

#include <ngraph/ngraph.hpp>

#include "layers.hpp"
#include "mnist.hpp"

using namespace ngraph;

int main(int argc, const char* argv[])
{
    size_t batch_size = 128;
    MNistDataLoader test_loader{batch_size, MNistImageLoader::TEST, MNistLabelLoader::TEST};
    size_t input_size = test_loader.get_columns() * test_loader.get_rows();
    test_loader.open();

    auto x_input = std::make_shared<InputLayer>("X", element::f32, Shape{batch_size, input_size});
    auto mlp_0 = std::make_shared<MLPLayer>("MLP_0", x_input, 10);

    return 0;
}
