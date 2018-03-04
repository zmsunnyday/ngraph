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

#include "code_writer.hpp"

using namespace std;
using namespace ngraph;

codegen::CodeWriter::CodeWriter()
    : indent(0)
    , m_pending_indent(true)
    , m_temporary_name_count(0)
{
}

string codegen::CodeWriter::get_code() const
{
    stringstream ss;

    for (const string& include : m_includes)
    {
        ss << include << "\n";
    }
    ss << m_ss.str();

    return ss.str();
}

void codegen::CodeWriter::operator+=(const string& s)
{
    *this << s;
}

string codegen::CodeWriter::generate_temporary_name(string prefix)
{
    stringstream ss;

    ss << prefix << m_temporary_name_count;
    m_temporary_name_count++;

    return ss.str();
}

void codegen::CodeWriter::block_begin()
{
    *this << "{\n";
    indent++;
}

void codegen::CodeWriter::block_end()
{
    indent--;
    *this << "}\n";
}

void codegen::CodeWriter::add_include(const string& s)
{
    m_includes.push_back(s);
}
