cmake_minimum_required(VERSION 3.2.2)
project(boxblur_cpu)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -DDEBUG")
else()
    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall -Werror")
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")


set(SOURCE_DIR source)
set(INCLUDE_DIR include ${CMAKE_SOURCE_DIR}/common/include)

file(GLOB SOURCES
        "${SOURCE_DIR}/*.cpp"
        "${SOURCE_DIR}/*.c")

file(GLOB INCLUDE
        "${CMAKE_SOURCE_DIR}/common/include/*.h"
        "${CMAKE_SOURCE_DIR}/common/include/*.hpp"
        "${INCLUDE_DIR}/*.h"
        "${INCLUDE_DIR}/*.hpp")

include_directories(${INCLUDE_DIR})

add_executable(boxblur_cpu ${SOURCES})

set_target_properties(boxblur_cpu PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}
        )
