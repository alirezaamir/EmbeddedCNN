cmake_minimum_required(VERSION 3.17)
project(EpilepsyGAN C)

set(CMAKE_C_STANDARD 99)
set(GCC_COVERAGE_COMPILE_FLAGS "-Wall")
add_executable(EpilepsyGAN main.c gan.h ecg.c filter.c filter.h)
target_link_libraries(EpilepsyGAN m)