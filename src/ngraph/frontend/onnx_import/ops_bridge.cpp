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

#include <algorithm>
#include <functional>

#include "ngraph/frontend/onnx_import/op/add.hpp"
#include "ops_bridge.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            namespace error
            {
                struct unknown_operation : ngraph_error
                {
                    explicit unknown_operation(const std::string& op_type)
                        : ngraph_error{"unknown operation: " + op_type}
                    {
                    }
                };

            } // namespace error

            NodeVector add(const Node& node) { return op::add(node); }
            class ops_bridge
            {
            public:
                ops_bridge(const ops_bridge&) = delete;
                ops_bridge& operator=(const ops_bridge&) = delete;
                ops_bridge(ops_bridge&&) = delete;
                ops_bridge& operator=(ops_bridge&&) = delete;

                static NodeVector make_ng_nodes(const Node& node)
                {
                    return ops_bridge::get()(node);
                }

            private:
                std::map<std::string, std::function<NodeVector(const Node&)>> m_map;

                static const ops_bridge& get()
                {
                    static ops_bridge instance;
                    return instance;
                }

                ops_bridge() { m_map.emplace("Add", std::bind(add, std::placeholders::_1)); }
                NodeVector operator()(const Node& node) const
                {
                    try
                    {
                        return m_map.at(node.op_type())(node);
                    }
                    catch (const std::exception&)
                    {
                        throw detail::error::unknown_operation{node.op_type()};
                    }
                }
            };

        } // namespace detail

        namespace ops_bridge
        {
            NodeVector make_ng_nodes(const Node& node)
            {
                return detail::ops_bridge::make_ng_nodes(node);
            }

        } // namespace ops_bridge

    } // namespace onnx_import

} // namespace ngraph
