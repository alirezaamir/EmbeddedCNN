cmake_minimum_required(VERSION 3.17)
project(EpilepsyGAN C)

set(CMAKE_C_STANDARD 99)

add_executable(EpilepsyGAN main.c gan.h fcn.c)