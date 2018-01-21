set(SRC
    header_rewrite.cpp
    uncomment.cpp
    util.cpp
)

include_directories(SYSTEM ${LLVM_INCLUDE_DIR})

add_library(header_tools STATIC ${SRC})

target_include_directories(header_tools PUBLIC ".")
add_dependencies(header_tools ext_llvm)
target_link_libraries(header_tools ${LLVM_LINK_LIBS})
