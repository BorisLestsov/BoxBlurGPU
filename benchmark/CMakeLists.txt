cmake_minimum_required(VERSION 3.2.2)
project(benchmark)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -DDEBUG")
else()
    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -Werror")
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")


set(SOURCE_DIR source)
set(INCLUDE_DIR
    include
    ${CMAKE_SOURCE_DIR}/common/include
    ${CMAKE_SOURCE_DIR}/boxblur_cpu/include
    ${CMAKE_SOURCE_DIR}/boxblur_cuda/include
)

file(GLOB SOURCES
    "${SOURCE_DIR}/*.cpp"
    "${SOURCE_DIR}/*.c"
)

file(GLOB INCLUDE
    "${INCLUDE_DIR}/*.h"
    "${INCLUDE_DIR}/*.hpp"
)

include_directories(${INCLUDE_DIR})

add_executable(benchmark ${SOURCES})
target_link_libraries(benchmark boxblur_cpu_lib boxblur_cuda_lib)

set_target_properties(benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
)

