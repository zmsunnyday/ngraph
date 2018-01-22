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

#include "collect_headers.hpp"
#include "file_util.hpp"
#include "util.hpp"

using namespace std;

static string find_path(const string& path)
{
    string rc;
    iterate_files(path,
                  [&](const string& file, bool is_dir) {
                      if (is_dir)
                      {
                          string dir_name = file_util::get_file_name(file);
                          if (is_version_number(dir_name))
                          {
                              rc = file;
                          }
                      }
                  },
                  true);
    return rc;
}

vector<ResourceInfo> collect_headers()
{
    string cpp0 = find_path("/usr/include/x86_64-linux-gnu/c++/");
    string cpp1 = find_path("/usr/include/c++/");

    vector<ResourceInfo> include_paths;
    include_paths.push_back({CLANG_BUILTIN_HEADERS_PATH, {}, true});
    include_paths.push_back({"/usr/include/x86_64-linux-gnu", {"asm", "sys", "bits", "gnu"}});
    include_paths.push_back({"/usr/include", {"linux", "asm-generic"}});
    include_paths.push_back({cpp0, {"bits"}});
    include_paths.push_back({cpp1, {"bits", "ext", "debug", "backward"}});
    include_paths.push_back({EIGEN_HEADERS_PATH, {}, true});
    include_paths.push_back({NGRAPH_HEADERS_PATH, {}, true});
    include_paths.push_back({TBB_HEADERS_PATH, {}, true});

    return include_paths;
}
