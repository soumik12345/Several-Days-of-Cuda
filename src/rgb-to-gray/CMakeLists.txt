cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(rgb_to_gray)
set(EXECUTABLE_NAME rgb_to_gray)

file(GLOB cpu_source_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
file(GLOB gpu_source_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

enable_language("CUDA")
add_executable(${EXECUTABLE_NAME} ${cpu_source_files} ${gpu_source_files})
target_link_libraries(${EXECUTABLE_NAME} ${OpenCV_LIBS})
