# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

function(NGRAPH_GET_HEADER_SEARCH_PATHS)
# for gcc "echo | gcc -c -xc++ -Wp,-v -"
    message(STATUS "*********** CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER}")
    exec_program(${CMAKE_CXX_COMPILER} ARGS echo | clang++-3.9 -c -xc++ -Wp,-v -
        OUTPUT_VARIABLE TMP_OUTPUT)
    string(REPLACE "\n" ";" TMP_LIST ${TMP_OUTPUT})
    set(SNARF FALSE)
    set(NGRAPH_SEARCH_PATHS)
    foreach(LINE ${TMP_LIST})
        string(STRIP ${LINE} LINE)
        if (${LINE} MATCHES "^#include")
            set(SNARF TRUE)
        elseif (${LINE} MATCHES "^End of search list")
            set(SNARF FALSE)
        elseif (SNARF)
            set(TMP_PATH ${TMP_PATH} ${LINE})
        endif()
    endforeach()
    set(NGRAPH_SEARCH_PATHS ${TMP_PATH} PARENT_SCOPE)
endfunction()
