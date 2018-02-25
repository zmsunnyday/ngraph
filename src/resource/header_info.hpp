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

#include <string>

class HeaderInfo
{
public:
    HeaderInfo(const std::string& spath, const std::string& rpath)
        : m_relative_path(rpath)
        , m_search_path(spath)
        , m_absolute_path(spath + "/" + rpath)
    {
    }

    std::string relative_path() const { return m_relative_path; }
    std::string search_path() const { return m_search_path; }
    std::string absolute_path() const { return m_absolute_path; }
private:
    std::string m_relative_path;
    std::string m_search_path;
    std::string m_absolute_path;
};
