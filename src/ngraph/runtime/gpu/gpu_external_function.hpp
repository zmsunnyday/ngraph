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

#pragma once

#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/codegen/compiler.hpp"
#include "ngraph/codegen/execution_engine.hpp"
#include "ngraph/function.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/dump_sorted.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/memory_layout.hpp"
#include "ngraph/runtime/gpu/gpu_call_frame.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_view_wrapper.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPU_Emitter;
            class GPU_CallFrame;
            struct GPURuntimeContext;

            using OpFunction =
                std::function<void(GPU_ExternalFunction* external_function,
                                   codegen::CodeWriter&,
                                   const ngraph::Node*,
                                   const std::vector<GPU_TensorViewWrapper>& inputs,
                                   const std::vector<GPU_TensorViewWrapper>& outputs)>;

            using OpMap = std::unordered_map<std::type_index, OpFunction>;

            class GPU_ExternalFunction : public std::enable_shared_from_this<GPU_ExternalFunction>
            {
                friend class GPU_CallFrame;
                friend class GPU_Backend;

            public:
                GPU_ExternalFunction(const std::shared_ptr<ngraph::Function>& function,
                                     bool release_function = true);
                ~GPU_ExternalFunction();
                std::shared_ptr<ngraph::runtime::gpu::GPU_CallFrame> make_call_frame();
                std::unique_ptr<runtime::gpu::GPURuntimeContext>& ctx();
                const std::unique_ptr<GPUPrimitiveEmitter>& get_primitive_emitter() const
                {
                    return m_primitive_emitter;
                }

            protected:
                void compile();

                EntryPoint m_compiled_function;

            private:
                void emit_header();
                void emit_timer_functions();
                void emit_declare_constants();
                void emit_declare_functions();
                void collect_unique_functions();
                void emit_functions();
                void store_emitted_functions(const std::string& code);
                void emit_debug_function_entry(Node* node);
                void emit_debug_function_exit(Node* node);
                void handle_output_alias(
                    const Node&,
                    const std::unordered_map<descriptor::TensorView*, std::vector<size_t>>&);
                void release_function() { m_function = nullptr; }
                std::string emit_op_as_function(const Node& node, const std::string& function_name);
                std::string strip_comments(const std::string& s) const;
                std::unique_ptr<codegen::Compiler> m_compiler;
                std::unique_ptr<codegen::ExecutionEngine> m_execution_engine;
                bool m_emit_timing;
                std::unordered_map<std::string, std::string> m_variable_name_map;
                std::unordered_map<const Node*, std::string> m_node_function_map;
                std::map<std::string, size_t> m_name_index_map;
                std::shared_ptr<ngraph::Function> m_function;
                bool m_release_function;
                bool m_is_compiled;
                std::string m_function_name;

                codegen::CodeWriter m_writer;
                pass::Manager m_pass_manager;

                std::string m_pch_header_source;
                bool m_temporaries_used = false;
                cublasHandle_t m_cublas_handle;
                cudnnHandle_t m_cudnn_handle;
                std::unique_ptr<GPUPrimitiveEmitter> m_primitive_emitter;
                std::unique_ptr<GPURuntimeContext> m_ctx;
            };
        }
    }
}
