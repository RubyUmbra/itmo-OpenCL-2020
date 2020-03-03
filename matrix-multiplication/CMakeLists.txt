cmake_minimum_required(VERSION 3.10)
project(matrix-multiplication VERSION 1.0.0 LANGUAGES C)
set(CMAKE_C_STANDARD 99)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fopenmp")

add_executable(matrix-multiplication main.c)
target_link_libraries(matrix-multiplication OpenCL)
