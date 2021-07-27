cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

cmake_policy(SET CMP0104 OLD)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(unique_index_2d)

file(GLOB cpu_source_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
file(GLOB gpu_source_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

enable_language("CUDA")
add_executable(unique_index_2d ${cpu_source_files} ${gpu_source_files})
