cmake_minimum_required(VERSION 3.17)
project(threads CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(threads main.cu)

set_target_properties(
        threads
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)