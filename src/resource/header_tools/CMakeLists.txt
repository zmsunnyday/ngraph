set(SRC
    header_rewrite.cpp
    uncomment.cpp
    util.cpp
)

add_library(header_tools STATIC ${SRC})
target_include_directories(header_tools PUBLIC ".")
