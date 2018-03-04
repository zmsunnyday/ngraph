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

class Layer
{
protected:
    Layer(const std::string& name);

public:
    virtual ~Layer(){};

    virtual const ngraph::op::NodeVector& get_output_ops() = 0;

    // Parameters for inputs passed to the function
    virtual const ngraph::op::ParameterVector& get_input_parameters() = 0;

    // Parameters for trainable variables passed to the function
    virtual const ngraph::op::ParameterVector& get_variable_parameters() = 0;

    // Parameters for other variables, such as momentums
    virtual const ngraph::op::ParameterVector& get_other_parameters() = 0;

    const std::string& get_name() const { return m_name; }
protected:
    std::string m_name;
};

class InputLayer : public Layer
{
public:
    InputLayer(const ngraph::element::Type& element_type, const ngraph::Shape& shape);
};

class MLPLayer : public Layer
{
public:
    MLPLayer(std::shared_ptr<Layer>& input, size_t output_size);
};
