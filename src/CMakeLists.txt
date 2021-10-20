cmake_minimum_required(VERSION 3.17)
project(EpilepsyGAN C)

set(CMAKE_C_STANDARD 99)

add_executable(EpilepsyGAN src/main.c src/main.h fcn.c)