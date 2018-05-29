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

#include <fstream>
#include <string>

#include "ngraph/except.hpp"
#include "onnx_util.hpp"
#include "model.hpp"
#include "graph.hpp"

using namespace std;
using namespace ngraph;

onnx::ModelProto onnx_util::load_onnx_file(const string& filepath)
{
    onnx::ModelProto model_proto;

    {
        fstream input(filepath, ios::in | ios::binary);
        if (!input)
        {
            throw ngraph_error("File not found: " + filepath);
        }
        else if (!model_proto.ParseFromIstream(&input))
        {
            throw ngraph_error("Failed to parse ONNX file: " + filepath);
        }
    }

    return model_proto;
}

vector<shared_ptr<Function>> onnx_util::import_onnx_file(const string& filepath)
{
    onnx::ModelProto model_proto = ngraph::onnx_util::load_onnx_file(filepath);
    return onnx_util::import_onnx_model(model_proto);
}

vector<shared_ptr<Function>> onnx_util::import_onnx_model(const onnx::ModelProto& onnx_model)
{
    vector<shared_ptr<Function>> output_functions;
    onnx_import::Model model_wrapper(onnx_model);
    onnx_import::Graph graph_wrapper(onnx_model.graph());

    for (const auto& output : graph_wrapper.get_outputs())
    {
        auto output_name = output.get_name();
        auto model = graph_wrapper.get_ng_node_from_cache(output_name);
        auto parameters = graph_wrapper.get_ng_parameters();
        auto function = std::make_shared<Function>(model, parameters);
        output_functions.emplace_back(function);
    }
    return output_functions;
}

shared_ptr<Function> onnx_util::import_onnx_function(const onnx::ModelProto& onnx_model)
{
    return onnx_util::import_onnx_model(onnx_model)[0];
}
