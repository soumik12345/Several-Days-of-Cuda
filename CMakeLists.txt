cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(several_days_of_cuda)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

include_directories(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/lib)

enable_language("CUDA")
add_executable(several_days_of_cuda ${cpu_source_files} ${gpu_source_files})
