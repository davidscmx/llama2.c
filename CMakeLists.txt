cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
project(llama2c_library VERSION 1.0.0 LANGUAGES C)
add_library(llama2c_library SHARED run.c run.def)
add_executable(llama2c_exe run.c)

# Link the math library to the llama2c_test target
target_link_libraries(llama2c_exe m)

set_target_properties(llama2c_library PROPERTIES
    PUBLIC_HEADER run.h
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "llama2c"
    XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Hex_Identity_ID_Goes_Here"
)
