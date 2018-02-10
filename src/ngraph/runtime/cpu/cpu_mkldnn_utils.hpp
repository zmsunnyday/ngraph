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
inline void emit_memory_desc(codegen::CodeWriter& writer,
                             const std::string& var,
                             const std::string& shape,
                             const std::string& type,
                             const std::string& layout)
{

    writer << "memory::desc " + var + " = memory::desc({" + shape + "}, " + type + ", "
        "memory::format::" + layout + ");\n";
};

inline void emit_memory(codegen::CodeWriter& writer,
                        const std::string& var,
                        const std::string& desc,
                        const std::string& data)
{

    writer << "memory " + var + " = memory({" + desc + ", cpu_engine}, " + data + ");\n";
};

inline void emit_memory_dims(codegen::CodeWriter& writer,
                             const std::string& var,
                             const std::string& dims)
{

    writer << "memory::dims " << var << "{" << dims << "};\n";
};

inline void emit_exception_block_begin(codegen::CodeWriter& writer)
{
    writer << "try {\n";
    writer.indent++;
}

inline void emit_exception_block_end(codegen::CodeWriter& writer)
{
    writer.indent--;
    writer << "} catch (const mkldnn::error& e) {\n";
    writer.indent++;
    writer << "std::cerr << \"MKLDNN ERROR (\" << e.status << \"): \" << e.message << std::endl; \n"
        "throw; \n";
    writer.indent--;
    writer << "}\n";
}
}
}
}
}
