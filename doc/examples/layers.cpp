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

#include <ngraph/ngraph.hpp>

#include "layers.hpp"

using namespace ngraph;

Layer::Layer(const std::string& name, const std::set<std::shared_ptr<Layer>>& input_layers)
    : m_name(name)
    , m_input_layers(input_layers)
{
}

InputLayer::InputLayer(const std::string& name,
                       const ngraph::element::Type& element_type,
                       const ngraph::Shape& shape)
    : Layer(name, {})
{
    std::shared_ptr<op::Parameter> parameter = std::make_shared<op::Parameter>(element_type, shape);
    parameter->set_name(name);
    m_input_parameters.push_back(parameter);
    m_output_ops.push_back(parameter);
}

MLPLayer::MLPLayer(const std::string& name, const std::shared_ptr<Layer>& input, size_t output_size)
    : Layer(name, {input})
{
    // Get the weights and bias
    std::shared_ptr<Node> input_node = input->get_output_ops().at(0);
    const Shape& input_shape = input_node->get_output_shape(0);
    const element::Type& element_type = input_node->get_output_element_type(0);
    size_t input_size = input_shape.at(1);
    std::shared_ptr<op::Parameter> weights =
        std::make_shared<op::Parameter>(element_type, Shape{input_size, output_size});
    std::shared_ptr<op::Parameter> bias =
        std::make_shared<op::Parameter>(element_type, Shape{output_size});
    m_variable_parameters.push_back(weights);
    m_variable_parameters.push_back(bias);

    // Compute the output
    std::shared_ptr<Node> dot = std::make_shared<op::Dot>(input_node, weights, 1);
    std::shared_ptr<Node> add = std::make_shared<op::Add>(dot, bias);
    std::shared_ptr<Node> tanh = std::make_shared<op::Tanh>(add);
    m_output_ops.push_back(tanh);
}
