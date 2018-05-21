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

include(get_header_search_paths)

ngraph_get_header_search_paths()
message(STATUS "*********** Search Paths ${NGRAPH_SEARCH_PATHS}")

set(SRC
    code_writer.cpp
    compiler.cpp
    execution_engine.cpp
)

# LLVM binary builds are typically built without RTTI
# The built-in headers are in a version-specific directory
# This must be kept in sync with the LLVM + Clang version in use
set_source_files_properties(compiler.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti")

get_target_property(MKLDNN_INCLUDE_DIR libmkldnn INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(EIGEN_INCLUDE_DIR libeigen INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(LLVM_LIB_DIR libllvm INTERFACE_INCLUDE_DIRECTORIES)

set(HEADER_SEARCH_PATH_LIST
    ${NGRAPH_SEARCH_PATHS}
    ${EIGEN_INCLUDE_DIR}
    ${MKLDNN_INCLUDE_DIR}
    ${LLVM_LIB_DIR}/../lib/clang/5.0.1/include
    ${NGRAPH_INCLUDE_PATH}
    ${CMAKE_INSTALL_PREFIX}/include
)

if(NGRAPH_DISTRIBUTED_ENABLE)
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
    link_directories(${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
    set(HEADER_SEARCH_PATH_LIST ${HEADER_SEARCH_PATH_LIST} ${MPI_C_INCLUDE_PATH})
    add_definitions(-DNGRAPH_DISTRIBUTED)
endif()

if(NGRAPH_GPU_ENABLE)
    set(HEADER_SEARCH_PATH_LIST ${HEADER_SEARCH_PATH_LIST} ${CUDA_INCLUDE_DIRS})
    set(HEADER_SEARCH_PATH_LIST ${HEADER_SEARCH_PATH_LIST} ${CUDNN_INCLUDE_DIRS})
endif()

string(REPLACE ";" "\\;" HEADER_SEARCH_PATH_LIST "${HEADER_SEARCH_PATH_LIST}")
message(STATUS "&&&&&&&&&&& ${HEADER_SEARCH_PATH_LIST}")
set_source_files_properties(compiler.cpp PROPERTIES COMPILE_DEFINITIONS
    "HEADER_SEARCH_PATH_LIST=\"${HEADER_SEARCH_PATH_LIST}\"")

# Generate the resource file containing all headers used by the codegen compiler
add_custom_target(header_resource
    resource_generator --output ${CMAKE_BINARY_DIR}/header_resource.hpp --base codegen
    DEPENDS resource_generator ext_eigen ext_llvm ext_mkldnn
    BYPRODUCTS
)

add_library(codegen SHARED EXCLUDE_FROM_ALL ${SRC})
add_dependencies(codegen header_resource)
target_include_directories(codegen PUBLIC ${CMAKE_BINARY_DIR})
target_link_libraries(codegen PUBLIC libllvm libmkldnn libeigen pthread dl)
