set(SRC
    collect_headers.cpp
    compiler.cpp
    file_util.cpp
    header_rewrite.cpp
    uncomment.cpp
    util.cpp
)

include_directories(SYSTEM ${LLVM_INCLUDE_DIR})

add_library(header_tools STATIC ${SRC})

target_include_directories(header_tools PUBLIC ".")
add_dependencies(header_tools ext_llvm)
target_link_libraries(header_tools ${LLVM_LINK_LIBS} dl)

set_source_files_properties(collect_headers.cpp PROPERTIES COMPILE_DEFINITIONS
    "EIGEN_HEADERS_PATH=\"${EIGEN_INCLUDE_DIR}\";CLANG_BUILTIN_HEADERS_PATH=\"${LLVM_LIB_DIR}/clang/5.0.0/include\";TBB_HEADERS_PATH=\"${TBB_ROOT}/include\";NGRAPH_HEADERS_PATH=\"${NGRAPH_INCLUDE_PATH}\";")
set_source_files_properties(compiler.cpp PROPERTIES COMPILE_DEFINITIONS
    "EIGEN_HEADERS_PATH=\"${EIGEN_INCLUDE_DIR}\";CLANG_BUILTIN_HEADERS_PATH=\"${LLVM_LIB_DIR}/clang/5.0.0/include\";TBB_HEADERS_PATH=\"${TBB_ROOT}/include\";NGRAPH_HEADERS_PATH=\"${NGRAPH_INCLUDE_PATH}\";")
