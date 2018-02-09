// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
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

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "collect_headers.hpp"
#include "header_rewrite.hpp"
#include "uncomment.hpp"
#include "util.hpp"

using namespace std;

int main(int argc, char** argv)
{
    time_t main_timestamp = get_timestamp(argv[0]);
    static vector<string> valid_ext = {".h", ".hpp", ".tcc", ""};
    string output_path;

    for (size_t i = 1; i < argc; i++)
    {
        if (string(argv[i]) == "--output")
        {
            output_path = argv[++i];
        }
    }

    if (output_path.empty())
    {
        cout << "must specify output path with --output option" << endl;
        return -1;
    }

    time_t output_timestamp = get_timestamp(output_path);

    HeaderInfo header_info = collect_headers();

    // test for changes to any headers
    bool update_needed = main_timestamp > output_timestamp;
    if (!update_needed)
    {
        for (const string& header_file : header_info.headers)
        {
            time_t file_timestamp = get_timestamp(header_file);
            if (file_timestamp > output_timestamp)
            {
                update_needed = true;
                break;
            }
        }
    }

    if (update_needed)
    {
        ofstream out(output_path);
        out << "#pragma clang diagnostic ignored \"-Weverything\"\n";
        out << "#include <vector>\n";
        out << "namespace ngraph\n";
        out << "{\n";
        out << "    static const uint8_t header_resources[] =\n";
        out << "    {\n";
        vector<pair<size_t, size_t>> offset_size_list;
        size_t offset = 0;
        size_t total_size = 0;
        size_t total_count = 0;
        stopwatch timer;
        timer.start();

        for (const string& header_file : header_info.headers)
        {
            string header_data = read_file_to_string(header_file);
            string base_path;
            for (const string& p : header_info.search_paths)
            {
                if (starts_with(header_file, p))
                {
                    base_path = p;
                    break;
                }
            }
            header_data = rewrite_header(header_data, base_path);
            // header_data = uncomment(header_data);
            total_size += header_data.size();
            total_count++;

            // data layout is triplet of strings containing:
            // 1) search path
            // 2) header path within search path
            // 3) header data
            // all strings are null terminated and the length includes the null
            // The + 1 below is to account for the null terminator
            dump(out, base_path.c_str(), base_path.size() + 1);
            offset_size_list.push_back({offset, base_path.size() + 1});
            offset += base_path.size() + 1;

            dump(out, header_file.c_str(), header_file.size() + 1);
            offset_size_list.push_back({offset, header_file.size() + 1});
            offset += header_file.size() + 1;

            dump(out, header_data.c_str(), header_data.size() + 1);
            offset_size_list.push_back({offset, header_data.size() + 1});
            offset += header_data.size() + 1;
        }
        timer.stop();
        cout << "collection time " << timer.get_milliseconds() << "ms\n";
        out << "    };\n";
        out << "    struct HeaderInfo\n";
        out << "    {\n";
        out << "        const char* search_path;\n";
        out << "        const char* header_path;\n";
        out << "        const char* header_data;\n";
        out << "    };\n";
        out << "    std::vector<HeaderInfo> header_info\n";
        out << "    {\n";
        for (size_t i = 0; i < offset_size_list.size();)
        {
            out << "        {(char*)(&header_resources[" << offset_size_list[i++].first;
            out << "]), (char*)(&header_resources[" << offset_size_list[i++].first;
            out << "]), (char*)(&header_resources[" << offset_size_list[i++].first << "])},\n";
        }
        out << "    };\n";
        out << "}\n";
        cout.imbue(locale(""));
        cout << "Total size " << total_size << " in " << total_count << " files\n";
    }
}
