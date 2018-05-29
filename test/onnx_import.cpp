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
#include <sstream>

#include "gtest/gtest.h"
#include "ngraph/frontend/onnx_import/graph.hpp"
#include "ngraph/frontend/onnx_import/model.hpp"
#include "ngraph/frontend/onnx_import/onnx_util.hpp"
#include "ngraph/ngraph.hpp"
#include "onnx.pb.h"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(onnx, model_add_abc)
{
    // Load ONNX protobuf from file
    const string filepath = file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx");
    onnx::ModelProto model_proto = ngraph::onnx_util::load_onnx_file(filepath);
    ASSERT_EQ("ngraph ONNXImporter", model_proto.producer_name());

    // Wrap ONNX Model protobuf
    onnx_import::Model model_wrapper(model_proto);
    std::stringstream model_stream;
    model_stream << model_wrapper;
    ASSERT_EQ("<Model: ngraph ONNXImporter>", model_stream.str());

    // Wrap ONNX Graph protobuf
    onnx_import::Graph graph_wrapper(model_proto.graph());
    std::stringstream graph_stream;
    graph_stream << graph_wrapper;
    ASSERT_EQ("<Graph: test_graph>", graph_stream.str());

    // Test parsing Graph inputs (ValueInfo)
    auto value_wrappers = graph_wrapper.get_inputs();
    ASSERT_EQ(value_wrappers.size(), 3);
    auto value = value_wrappers[0];
    std::stringstream value_stream;
    value_stream << value;
    ASSERT_EQ("<ValueInfo: A>", value_stream.str());
    ASSERT_EQ(element::f32, value.get_element_type());

    // Test parsing Graph nodes
    auto node_wrappers = graph_wrapper.get_nodes();
    ASSERT_EQ(node_wrappers.size(), 2);
    ASSERT_EQ(graph_wrapper.get_inputs().size(), 3);
    ASSERT_EQ(graph_wrapper.get_outputs().size(), 1);

    auto node_wrapper = node_wrappers[0];
    std::stringstream node_stream;
    node_stream << node_wrapper;
    ASSERT_EQ("<Node(Add): add_node1>", node_stream.str());

    auto ng_inputs = node_wrapper.get_ng_inputs();
    ASSERT_EQ(ng_inputs.size(), 2);

    // Test converting ONNX node to nGraph node
    auto ng_nodes = node_wrapper.get_ng_nodes();
    ASSERT_EQ(ng_nodes.size(), 1);

    auto ng_node = ng_nodes[0];
    ASSERT_FALSE(ng_node->is_parameter());
    ASSERT_EQ(ng_node->get_arguments().size(), 2);

    // Perform nGraph calculation on ONNX model
    auto model = graph_wrapper.get_ng_node_from_cache("Y");
    auto parameters = graph_wrapper.get_ng_parameters();

    auto function = std::make_shared<Function>(model, parameters);
    auto backend = runtime::Backend::create("CPU");

    Shape shape{1};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{3});

    auto result = backend->create_tensor(element::f32, shape);

    backend->call(function, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{6}), read_vector<float>(result));
}

TEST(onnx, public_api)
{
    auto backend = runtime::Backend::create("CPU");
    Shape shape{1};
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{3});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{4});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{5});
    auto result = backend->create_tensor(element::f32, shape);

    // Test load_onnx_file
    const string filepath = file_util::path_join(SERIALIZED_ZOO, "onnx/add_abc.onnx");
    onnx::ModelProto model_proto = ngraph::onnx_util::load_onnx_file(filepath);
    vector<shared_ptr<Function>> model_functions = onnx_util::import_onnx_model(model_proto);
    ASSERT_EQ(model_functions.size(), 1);
    shared_ptr<Function> model_function = model_functions[0];
    backend->call(model_function, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{12}), read_vector<float>(result));

    // Test import_onnx_file
    model_functions = onnx_util::import_onnx_file(filepath);
    ASSERT_EQ(model_functions.size(), 1);
    shared_ptr<Function> model_function2 = model_functions[0];
    backend->call(model_function2, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{12}), read_vector<float>(result));

    // Test import_onnx_function
    shared_ptr<Function> model_function3 = onnx_util::import_onnx_function(model_proto);
    backend->call(model_function3, {result}, {a, b, c});
    EXPECT_EQ((vector<float>{12}), read_vector<float>(result));
}
