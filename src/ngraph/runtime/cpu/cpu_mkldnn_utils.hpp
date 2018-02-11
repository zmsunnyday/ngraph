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

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/common.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace mkldnn
            {
                struct CatchException
                {
                    static void block_begin(codegen::CodeWriter& writer)
                    {
                        writer << "try {\n";
                        writer.indent++;
                    }

                    static void block_end(codegen::CodeWriter& writer)
                    {
                        writer.indent--;
                        writer << "} catch (const mkldnn::error& e) {\n";
                        writer.indent++;
                        writer
                            << "std::cerr << \"MKLDNN ERROR (\" << e.status << \"): \" << e.message << std::endl; \n"
                                "throw; \n";
                        writer.indent--;
                        writer << "}\n";
                    }
                };

                template<typename ExceptionPolicy = CatchException>
                class ScopedEmitterUtil
                {
                public:
                    // disable copy
                    ScopedEmitterUtil(ScopedEmitterUtil const&) = delete;
                    ScopedEmitterUtil& operator=(ScopedEmitterUtil const&) = delete;
                    // avoid rvalue (temporary) parameter
                    ScopedEmitterUtil(const codegen::CodeWriter&&) = delete;

                    ScopedEmitterUtil(codegen::CodeWriter& writer)
                        : m_writer(writer)
                    {
                        m_writer << "{\n";
                        m_writer.indent++;
                        ExceptionPolicy::block_begin(m_writer);
                        m_writer << "engine cpu_engine = engine(engine::cpu, 0);\n";
                    }
                    ~ScopedEmitterUtil()
                    {
                        ExceptionPolicy::block_end(m_writer);
                        m_writer.indent--;
                        m_writer << "}\n";
                    }

                    void emit_memory_desc(const std::string& var,
                                          const std::string& shape,
                                          const std::string& type,
                                          const std::string& layout)
                    {

                        m_writer
                            << "memory::desc " + var + " = memory::desc({" + shape + "}, " + type
                                + ", "
                                    "memory::format::" + layout + ");\n";
                    };

                    void emit_memory(const std::string& var,
                                     const std::string& desc,
                                     const std::string& data)
                    {

                        m_writer
                            << "memory " + var + " = memory({" + desc + ", cpu_engine}, " + data
                                + ");\n";
                    };

                    void emit_memory_dims(const std::string& var,
                                          const std::string& dims)
                    {

                        m_writer << "memory::dims " << var << "{" << dims << "};\n";
                    };

                private:
                    codegen::CodeWriter& m_writer;
                };
            }
        }
    }
}
