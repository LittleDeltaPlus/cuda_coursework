cmake_minimum_required(VERSION 3.17)
project(device_properties CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(device_properties main.cu)

set_target_properties(
        device_properties
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)