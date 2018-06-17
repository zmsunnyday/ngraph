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

function(NGRAPH_FIND_HEADER_SEARCH_PATHS)
    set(options)
    set(oneValueArgs OUTPUT_VAR)
    set(multiValueArgs)
    cmake_parse_arguments(NGRAPH_FIND_HEADER_SEARCH_PATHS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    get_target_property(EIGEN_INCLUDE_DIR libeigen INTERFACE_INCLUDE_DIRECTORIES)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER}
        -H
        -I ${PROJECT_SOURCE_DIR}/src
        -I ${EIGEN_INCLUDE_DIR}
        ${CMAKE_MODULE_PATH}header_search.cpp
        OUTPUT_VARIABLE RESULTS
        ERROR_VARIABLE RESULTS)
    # message(STATUS "************************** ${CMAKE_MODULE_PATH}/header_search.cpp")
    # message(STATUS "************************** ${RESULTS}")
    # message(STATUS "************************** EIGEN_INCLUDE_DIR ${EIGEN_INCLUDE_DIR}")
    string(REPLACE "\n" ";" RESULTS ${RESULTS})
    # message(STATUS "************************** ${RESULTS}")
    foreach(LINE ${RESULTS})
        if(${LINE} MATCHES "^[.]")
            string(REGEX REPLACE "[.]+ " "" LINE ${LINE})
            get_filename_component(LINE ${LINE} ABSOLUTE)
            # message(STATUS "full '${LINE}'")
            get_filename_component(LINE ${LINE} DIRECTORY)
            if(${LINE} MATCHES "/bits$" OR
               ${LINE} MATCHES "ngraph" OR
               ${LINE} MATCHES "/sys$" OR
               ${LINE} MATCHES "/ext$")
                continue()
            endif()
            # message(STATUS "dir  '${LINE}'")
            list(FIND PATH_LIST ${LINE} INDEX)
            if (${INDEX} EQUAL -1)
                message(STATUS "ADD ${LINE}")
                list(APPEND PATH_LIST ${LINE})
            endif()
        endif()
    endforeach()
    set(NGRAPH_FIND_HEADER_SEARCH_PATHS_OUTPUT_VAR "blah")
endfunction()