cmake_minimum_required(VERSION 3.17)
project(EpilepsyGAN_HW C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(EpilepsyGAN_HW
        data.c
        gan.h
        main.c)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

