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

#include <string>
#include <vector>

class ResourceInfo
{
public:
    ResourceInfo(const std::string& source,
                 const std::vector<std::string>& _subdirs,
                 bool recursive = false)
        : search_path(source)
        , subdirs(_subdirs)
        , is_recursive(recursive)

    {
    }

    const std::string search_path;
    const std::vector<std::string> subdirs;
    const bool is_recursive;

    std::vector<std::string> files;
};

std::vector<ResourceInfo> collect_headers();
