cmake_minimum_required(VERSION 3.2.2)
project(boxblur_cpu)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -DDEBUG")
else()
    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -Werror")
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")


set(SOURCE_DIR source)
set(INCLUDE_DIR include ${CMAKE_SOURCE_DIR}/common/include)

file(GLOB SOURCES_LIB
    "${SOURCE_DIR}/*.c"
    "${SOURCE_DIR}/*.cpp")
list(REMOVE_ITEM SOURCES_LIB
    ${SOURCE_DIR}/main.cpp)

file(GLOB INCLUDE
        "${INCLUDE_DIR}/*.h"
        "${INCLUDE_DIR}/*.hpp")

include_directories(${INCLUDE_DIR})

add_library(boxblur_cpu_lib ${SOURCES_LIB})
add_executable(boxblur_cpu ${SOURCE_DIR}/main.cpp)
target_link_libraries(boxblur_cpu boxblur_cpu_lib)

set_target_properties(boxblur_cpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}
)
