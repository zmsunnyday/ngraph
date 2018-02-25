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

if (NGRAPH_CPU_ENABLE)
    set (SRC
        main.cpp
        util.cpp
        uncomment.cpp
        header_rewrite.cpp
        find_headers.cpp
        find_headers_used.cpp
        file_util.cpp
    )

    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)

    link_directories(${LLVM_LIB_DIR})
    include_directories(SYSTEM ${LLVM_INCLUDE_DIR})

    add_executable(resource_generator ${SRC})

    add_dependencies(resource_generator ext_llvm ext_eigen ext_mkldnn)
    target_link_libraries(resource_generator ${LLVM_LINK_LIBS} Threads::Threads dl)
    set_source_files_properties(find_headers_used.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti")

    set(HEADER_SEARCH_DEFINES
        "EIGEN_HEADERS_PATH=\"${EIGEN_INCLUDE_DIR}\""
        "MKLDNN_HEADERS_PATH=\"${MKLDNN_INCLUDE_DIR}\""
        "CLANG_BUILTIN_HEADERS_PATH=\"${LLVM_LIB_DIR}/clang/5.0.1/include\""
        "NGRAPH_HEADERS_PATH=\"${NGRAPH_INCLUDE_PATH}\""
    )

    if(NGRAPH_TBB_ENABLE)
        list(APPEND HEADER_SEARCH_DEFINES "TBB_HEADERS_PATH=\"${TBB_ROOT}/include\"")
        set(HEADER_SEARCH_DEFINES ${HEADER_SEARCH_DEFINES} "NGRAPH_TBB_ENABLE")
    endif()

    set_source_files_properties(find_headers.cpp PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES}")
    set_source_files_properties(find_headers_used.cpp PROPERTIES COMPILE_DEFINITIONS "${HEADER_SEARCH_DEFINES}")
endif()
