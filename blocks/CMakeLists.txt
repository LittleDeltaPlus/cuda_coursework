cmake_minimum_required(VERSION 3.17)
project(blocks CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(blocks main.cu)

set_target_properties(
        blocks
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)