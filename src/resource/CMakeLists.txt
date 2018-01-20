# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set (SRC
    main.cpp
)

add_subdirectory(header_tools)
include_directories(header_tools)

add_executable(resource_generator ${SRC} header_tools)
add_dependencies(resource_generator header_tools)
target_link_libraries(resource_generator header_tools)

# include_directories("${LLVM_INCLUDE_DIR}")
# message("LLVM AT ${LLVM_INCLUDE_DIR}")
set_source_files_properties(main.cpp PROPERTIES COMPILE_DEFINITIONS
    "EIGEN_HEADERS_PATH=\"${EIGEN_INCLUDE_DIR}\";CLANG_BUILTIN_HEADERS_PATH=\"${LLVM_LIB_DIR}/clang/5.0.0/include\";TBB_HEADERS_PATH=\"${TBB_ROOT}/include\";NGRAPH_HEADERS_PATH=\"${NGRAPH_INCLUDE_PATH}\";")
