cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

cmake_policy(SET CMP0104 OLD)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(several_days_of_cuda)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/include)

enable_language("CUDA")
add_executable(several_days_of_cuda ${cpu_source_files} ${gpu_source_files})
target_link_libraries(several_days_of_cuda ${OpenCV_LIBS})

# Examples
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/hello-world)
add_custom_target(hello-world)
add_dependencies(hello-world hello_world)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/thread-id-demo)
add_custom_target(thread-id-demo)
add_dependencies(thread-id-demo thread_id_demo)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/block-dim-demo)
add_custom_target(block-dim-demo)
add_dependencies(block-dim-demo block_dim_demo)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/unique-index)
add_custom_target(unique-index)
add_dependencies(unique-index unique_index)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/unique-index-2d)
add_custom_target(unique-index-2d)
add_dependencies(unique-index-2d unique_index_2d)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/array-manipulation)
add_custom_target(array-manipulation)
add_dependencies(array-manipulation array_manipulation)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/array-manipulation-manual-memory)
add_custom_target(array-manipulation-manual-memory)
add_dependencies(array-manipulation-manual-memory array_manipulation_manual_memory)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/parallel-loop)
add_custom_target(parallel-loop)
add_dependencies(parallel-loop parallel_loop)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/parallel-loop-multi-blocks)
add_custom_target(parallel-loop-multi-blocks)
add_dependencies(parallel-loop-multi-blocks parallel_loop_multi_blocks)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/vector-addition)
add_custom_target(vector-addition)
add_dependencies(vector-addition vector_addition)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/vector-addition-unified-memory)
add_custom_target(vector-addition-unified-memory)
add_dependencies(vector-addition-unified-memory vector_addition_unified_memory)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/fibonacci)
add_custom_target(fibo)
add_dependencies(fibo fibonacci)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/fibonacci-dp)
add_custom_target(fibonacci-dp)
add_dependencies(fibonacci-dp fibonacci_dp)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/rgb-to-gray)
add_custom_target(rgb-to-gray)
add_dependencies(rgb-to-gray rgb_to_gray)
