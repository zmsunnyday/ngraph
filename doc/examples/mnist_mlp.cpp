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
#include <iostream>
#include <memory>
#include <string>

#include <ngraph/ngraph.hpp>

#include "mnist.hpp"

using namespace ngraph;

std::ostream& operator<<(std::ostream& s, const Shape& shape)
{
    s << "Shape{";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        s << shape.at(i);
        if (i + 1 < shape.size())
        {
            s << ", ";
        }
    }
    s << "}";
    return s;
}

int main(int argc, const char* argv[])
{
    size_t batch_size = 128;
    size_t hidden_size = 500;
    size_t output_size = 10;
    float log_min = -50.0f;
    MNistDataLoader test_loader{batch_size, MNistImageLoader::TEST, MNistLabelLoader::TEST};
    test_loader.open();
    size_t input_size = test_loader.get_columns() * test_loader.get_rows();

    // The data input
    auto X = std::make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    auto Y = std::make_shared<op::Parameter>(element::f32, Shape{batch_size});
    auto learning_rate = std::make_shared<op::Parameter>(element::f32, Shape{});

    // Layer 0
    auto W0 = std::make_shared<op::Parameter>(element::f32, Shape{input_size, hidden_size});
    auto b0 = std::make_shared<op::Parameter>(element::f32, Shape{hidden_size});
    auto l0_dot = std::make_shared<op::Dot>(X, W0, 1);
    auto b0_broadcast =
        std::make_shared<op::Broadcast>(b0, Shape{batch_size, hidden_size}, AxisSet{0});
    auto l0_sum = std::make_shared<op::Add>(l0_dot, b0_broadcast);
    auto l0 = std::make_shared<op::Tanh>(l0_sum);

    // Layer 1
    auto W1 = std::make_shared<op::Parameter>(element::f32, Shape{hidden_size, output_size});
    auto b1 = std::make_shared<op::Parameter>(element::f32, Shape{output_size});
    auto l1_dot = std::make_shared<op::Dot>(l0, W1, 1);
    auto b1_broadcast =
        std::make_shared<op::Broadcast>(b1, Shape{batch_size, output_size}, AxisSet{0});
    auto l1_sum = std::make_shared<op::Add>(l1_dot, b1_broadcast);
    auto l1 = std::make_shared<op::Tanh>(l1_sum);

    // Softmax
    auto sm = std::make_shared<op::Softmax>(l1, AxisSet{1});

    // Cost computation
    auto labels = std::make_shared<op::OneHot>(Y, Shape{batch_size, output_size}, 1);
    auto sm_clip_value =
        std::make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{log_min});
    auto sm_clip_broadcast = std::make_shared<op::Broadcast>(
        sm_clip_value, Shape{batch_size, output_size}, AxisSet{0, 1});
    auto sm_clip = std::make_shared<op::Maximum>(sm, sm_clip_broadcast);
    auto sm_log = std::make_shared<op::Log>(sm_clip);
    std::cout << labels->get_output_shape(0) << std::endl;
    auto prod = std::make_shared<op::Multiply>(sm_clip, labels);
    auto loss = std::make_shared<op::Sum>(prod, AxisSet{0, 1});

    // Backprop
    // Each of W0, b0, W1, and b1
    auto delta =
        std::make_shared<op::Multiply>(std::make_shared<op::Negative>(learning_rate), loss);

    auto W0_delta = loss->backprop_node(W0, delta);
    auto b0_delta = loss->backprop_node(b0, delta);
    auto W1_delta = loss->backprop_node(W1, delta);
    auto b1_delta = loss->backprop_node(b1, delta);

    // Updates
    auto W0_next = std::make_shared<op::Add>(W0, W0_delta);
    auto b0_next = std::make_shared<op::Add>(b0, b0_delta);
    auto W1_next = std::make_shared<op::Add>(W1, W1_delta);
    auto b1_next = std::make_shared<op::Add>(b1, b1_delta);

    // Plain inference
    // X, W0, b0, W1, b1 -> sm

    // Inference test function
    // X, Y, W0, b0, W1, b1 -> sm, loss

    // Train
    // X, Y, W0, b0, W1, b1 -> loss, W0_next, b0_next, W1_next, b1_next

    return 0;
}
