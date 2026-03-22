cmake_minimum_required(VERSION 3.14)

#
# This tells CMake that your project's name is homeexam-01, and that the programs
# here are written in C.
# 
project(Codec63
	VERSION 3.0
	DESCRIPTION "CMake configuration file for Codec63 in C"
        LANGUAGES C)

# Use the C standard from 2011. No effect on Windows (Visual Studio ignores
# the standards anyway).
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS ON) # definitely use GNU extensions (actually default)

# Overwrite default -O3 flag for Release mode
set(CMAKE_C_FLAGS_RELEASE "-O1 -fno-tree-vectorize -ffast-math")
# Make sure Debug mode flags are similar to Release mode flags
set(CMAKE_C_FLAGS_DEBUG "-g -O1 -fno-tree-vectorize -ffast-math")

# Debug compiles with -g or equivalent for debugging.
# Change to Release for good speed.
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

#
# If this is a Tegra or something similar, set our processor subrevision.
# On Mac, CMake adds arch flags by itself.
#
if(CMAKE_C_COMPILER_ID STREQUAL GNU)
  if(EXISTS "/etc/nv_tegra_release")
    add_compile_options(-march=armv8.2-a)#+fp16) uncomment for half-prec support
  endif()
else()
  message(FATAL "Not compiling with an gcc")
endif()

find_path(NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h
  PATHS /usr/local/cuda/include /usr/local/cuda-11/include /usr/local/cuda-10/include
  REQUIRED)


find_package(OpenMP REQUIRED)

add_library(c63
            STATIC dsp.c dsp.h
	           tables.c tables.h
	           io.c io.h
	           common.c common.h
	           me.c me.h)
target_include_directories(c63 PRIVATE ${NVTX_INCLUDE_DIR})

add_executable(c63enc
               c63enc.c c63.h
	       c63_write.c c63_write.h)
target_include_directories(c63enc PRIVATE ${NVTX_INCLUDE_DIR})
target_link_libraries(c63 OpenMP::OpenMP_C)
target_link_libraries(c63enc c63 OpenMP::OpenMP_C m dl)
