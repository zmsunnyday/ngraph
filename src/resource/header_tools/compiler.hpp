// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <clang/Frontend/CompilerInstance.h>
#include <memory>
#include <string>
#include <vector>

#include "collect_headers.hpp"

class Compiler
{
public:
    HeaderInfo collect_headers(const std::string& source);
    void configure_search_path();

private:
    void add_header_search_path(const std::string& path);
    bool is_version_number(const std::string& path);

    std::unique_ptr<clang::CompilerInstance> m_compiler;
    std::vector<std::string> m_search_path_list;
};
