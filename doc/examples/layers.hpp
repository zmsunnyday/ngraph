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

#include <set>

class Layer
{
protected:
    Layer(const std::string& name, const std::set<std::shared_ptr<Layer>>& input_layers);

public:
    virtual ~Layer() {}
    const ngraph::NodeVector& get_output_ops() const { return m_output_ops; }
    // Parameters for inputs passed to the function
    const ngraph::op::ParameterVector& get_input_parameters() const { return m_input_parameters; }
    // Parameters for trainable variables passed to the function
    const ngraph::op::ParameterVector& get_variable_parameters() const
    {
        return m_variable_parameters;
    }

    // Parameters for other variables, such as momentums
    const ngraph::op::ParameterVector& get_other_parameters() const { return m_other_parameters; }
    const std::string& get_name() const { return m_name; }
protected:
    std::string m_name;
    std::set<std::shared_ptr<Layer>> m_input_layers;
    ngraph::NodeVector m_output_ops;
    ngraph::op::ParameterVector m_input_parameters;
    ngraph::op::ParameterVector m_variable_parameters;
    ngraph::op::ParameterVector m_other_parameters;
};

// Make a layer for a function input
class InputLayer : public Layer
{
public:
    InputLayer(const std::string& name,
               const ngraph::element::Type& element_type,
               const ngraph::Shape& shape);
};

class MLPLayer : public Layer
{
public:
    MLPLayer(const std::string& name, const std::shared_ptr<Layer>& input, size_t output_size);
};
