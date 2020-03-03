cmake_minimum_required(VERSION 3.10)

project(itmo-OpenCL-2020
        VERSION 1.0.0
        DESCRIPTION "OpenCL course at ITMO University in 2020."
        HOMEPAGE_URL "https://github.com/RubyUmbra/itmo-OpenCL-2020"
        LANGUAGES C)

set(CMAKE_C_STANDARD 99)

add_subdirectory(matrix-multiplication)
add_subdirectory(prefix-sum)
