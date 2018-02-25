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
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "find_headers.hpp"
#include "find_headers_used.hpp"
#include "header_info.hpp"
#include "header_rewrite.hpp"
#include "uncomment.hpp"
#include "util.hpp"

using namespace std;

int main(int argc, char** argv)
{
    cout << "Hello world\n";
    time_t main_timestamp = get_timestamp(argv[0]);
    string output_path;
    string base_name;

    for (size_t i = 1; i < argc; i++)
    {
        if (string(argv[i]) == "--output")
        {
            output_path = argv[++i];
        }
        else if (string(argv[i]) == "--base_name")
        {
            base_name = argv[++i];
        }
    }

    if (output_path.empty())
    {
        cout << "must specify output path with --output option" << endl;
        return -1;
    }

    vector<HeaderInfo> header_info = FindHeaders::collect_headers();

    time_t output_timestamp = get_timestamp(output_path);

    // test for changes to any headers
    bool update_needed = main_timestamp > output_timestamp;
    if (!update_needed)
    {
        for (const HeaderInfo& info : header_info)
        {
            time_t file_timestamp = get_timestamp(info.absolute_path());
            if (file_timestamp > output_timestamp)
            {
                update_needed = true;
                break;
            }
        }
    }

    if (update_needed)
    {
        size_t total_size = 0;
        size_t total_count = 0;
        const string prefix = "pReFiX";
        ofstream out(output_path);
        out << "#pragma clang diagnostic ignored \"-Weverything\"\n";
        out << "#include <vector>\n";
        out << "namespace ngraph\n";
        out << "{\n";
        out << "    const std::vector<std::string> builtin_search_paths =\n";
        out << "    {\n";
        unordered_set<string> search_paths;
        for (const HeaderInfo& info : header_info)
        {
            if (!contains(search_paths, info.search_path()))
            {
                search_paths.insert(info.search_path());
                out << "        \"" << info.search_path() << "\",\n";
            }
        }
        out << "    };\n";

        out << "    const std::vector<std::pair<std::string, std::string>> builtin_headers =\n";
        out << "    {\n";
        for (const HeaderInfo& info : header_info)
        {
            string header_data = read_file_to_string(info.absolute_path());
            header_data = rewrite_header(header_data, info.relative_path());
            // header_data = uncomment(header_data);
            total_size += header_data.size();
            total_count++;

            out << "        {";
            out << "\"" << info.absolute_path() << "\",\nR\"" << prefix << "(" << header_data << ")"
                << prefix << "\"},\n";
        }
        out << "    };\n";
        out << "}\n";
        cout.imbue(locale(""));
        cout << "Total size " << total_size << " in " << total_count << " files\n";
    }
}
