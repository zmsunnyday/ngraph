set(SRC
    code_writer.cpp
    compiler.cpp
    execution_engine.cpp
)

message(STATUS "codegen SRC ${SRC}")

# Generate the resource file containing all headers used by the codegen compiler
add_custom_target(header_resource
    resource_generator --output ${CMAKE_BINARY_DIR}/header_resource.hpp --base codegen
    DEPENDS resource_generator eigen ext_llvm ext_mkldnn
    BYPRODUCTS
)

add_library(codegen STATIC ${SRC})
add_dependencies(codegen header_resource ext_llvm ext_mkldnn eigen)
target_link_libraries(codegen PUBLIC ${LLVM_LINK_LIBS})

# this will pick up the generated header_resource.hpp
include_directories(${CMAKE_BINARY_DIR})
include_directories(SYSTEM ${LLVM_INCLUDE_DIR})

# LLVM binary builds are typically built without RTTI
# The built-in headers are in a version-specific directory
# This must be kept in sync with the LLVM + Clang version in use
set_source_files_properties(compiler.cpp PROPERTIES COMPILE_FLAGS "-fno-rtti")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set_source_files_properties(codegen/compiler.cpp PROPERTIES COMPILE_DEFINITIONS
    "EIGEN_HEADERS_PATH=\"${EIGEN_INCLUDE_DIR}\";MKLDNN_HEADERS_PATH=\"${MKLDNN_INCLUDE_DIR}\";CLANG_BUILTIN_HEADERS_PATH=\"${LLVM_LIB_DIR}/clang/5.0.0/include\";TBB_HEADERS_PATH=\"${TBB_ROOT}/include\";NGRAPH_HEADERS_PATH=\"${NGRAPH_INCLUDE_PATH}\";INSTALLED_HEADERS_PATH=\"${CMAKE_INSTALL_PREFIX}/include\";")
set(NGRAPH_CPU_DEBUGINFO_ENABLE 0 CACHE STRING "Enable debuginfo in the CPU backend")
set_property(SOURCE codegen/compiler.cpp APPEND_STRING PROPERTY COMPILE_DEFINITIONS
    "NGCPU_DEBUGINFO=${NGRAPH_CPU_DEBUGINFO_ENABLE};")
