cmake_minimum_required(VERSION 3.10)
project(prefix-sum VERSION 1.0.0 LANGUAGES C)
set(CMAKE_C_STANDARD 99)

add_executable(prefix-sum main.c)
target_link_libraries(prefix-sum OpenCL)
