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

#include <sstream>
#include <vector>

#include "find_headers.hpp"
#include "header_info.hpp"
#include "util.hpp"

using namespace std;

class ResourceInfo
{
public:
    ResourceInfo(const string& source, const vector<string>& _subdirs, bool recursive = false)
        : search_path(source)
        , subdirs(_subdirs)
        , is_recursive(recursive)

    {
    }

    const string search_path;
    const vector<string> subdirs;
    const bool is_recursive;

    vector<string> files;
};

string find_path(const string& path)
{
    string rc;
    iterate_files(path,
                  [&](const string& file, bool is_dir) {
                      if (is_dir)
                      {
                          string dir_name = get_file_name(file);
                          if (is_version_number(dir_name))
                          {
                              rc = file;
                          }
                      }
                  },
                  true);
    return rc;
}

vector<HeaderInfo> FindHeaders::collect_headers()
{
    cout << "collect_headers\n";
    vector<ResourceInfo> include_paths;
    static vector<string> valid_ext = {".h", ".hpp", ".tcc", ""};

#ifdef __APPLE__
    include_paths.push_back({EIGEN_HEADERS_PATH, {}, true});
    include_paths.push_back({MKLDNN_HEADERS_PATH, {}, true});
#ifdef NGRAPH_TBB_ENABLE
    include_paths.push_back({TBB_HEADERS_PATH, {}, true});
#endif
    include_paths.push_back({NGRAPH_HEADERS_PATH, {}, true});
    include_paths.push_back({CLANG_BUILTIN_HEADERS_PATH, {}, true});
    include_paths.push_back({"/Library/Developer/CommandLineTools/usr/include/c++/v1", {}});
#else // __APPLE__
    string cpp0 = find_path("/usr/include/x86_64-linux-gnu/c++/");
    string cpp1 = find_path("/usr/include/c++/");

    include_paths.push_back({CLANG_BUILTIN_HEADERS_PATH, {}, true});
    include_paths.push_back({"/usr/include/x86_64-linux-gnu", {"asm", "sys", "bits", "gnu"}});
    include_paths.push_back(
        {"/usr/include", {"asm", "sys", "bits", "gnu", "linux", "asm-generic"}});
    include_paths.push_back({cpp0, {"bits"}});
    include_paths.push_back({"/usr/include/c++/4.8.2/x86_64-redhat-linux", {"bits"}});
    include_paths.push_back({cpp1, {"bits", "ext", "debug", "backward"}});
    include_paths.push_back({EIGEN_HEADERS_PATH, {}, true});
    include_paths.push_back({MKLDNN_HEADERS_PATH, {}, true});
    include_paths.push_back({NGRAPH_HEADERS_PATH, {}, true});
#ifdef NGRAPH_TBB_ENABLE
    include_paths.push_back({TBB_HEADERS_PATH, {}, true});
#endif
#endif

    vector<HeaderInfo> rc;
    for (ResourceInfo& path : include_paths)
    {
        vector<string> path_list;
        path_list.push_back(path.search_path);
        for (const string& p : path.subdirs)
        {
            path_list.push_back(path_join(path.search_path, p));
        }
        for (const string& p : path_list)
        {
            iterate_files(p,
                          [&](const string& file, bool is_dir) {
                              if (!is_dir)
                              {
                                  string ext = get_file_ext(file);
                                  if (contains(valid_ext, ext))
                                  {
                                      string f = file.substr(path.search_path.size() + 1);
                                      rc.push_back({path.search_path, f});
                                      path.files.push_back(file);
                                  }
                              }
                          },
                          path.is_recursive);
        }
    }
    return rc;
}
