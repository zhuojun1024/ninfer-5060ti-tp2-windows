cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

# Single source for the release version: the tagged release vX.Y.Z carries exactly this version,
# and tools/win_port/package.ps1 refuses to package a tree whose tag does not match.
project(ninfer VERSION 0.1.2 LANGUAGES C CXX CUDA)

# WSL2's virtualized driver (591.x) segfaults inside the driver JIT compiler
# during CUDA initialization with the dynamically linked CUDA 13.1 runtime;
# the static runtime avoids that path. Off by default: native builds keep the
# dynamic runtime.
option(NINFER_STATIC_CUDART "Link the CUDA runtime statically" OFF)
if(NINFER_STATIC_CUDART)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
  set(NINFER_CUDART_TARGET CUDA::cudart_static)
else()
  set(NINFER_CUDART_TARGET CUDA::cudart)
endif()

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 13 ships CCCL, which rejects the traditional MSVC preprocessor; nvcc forwards host flags
# through -Xcompiler, so both paths need the conforming preprocessor.
if(MSVC)
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>")
  # Sources are UTF-8, and the bundled fmt in spdlog refuses to build without its Unicode support.
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
  # windows.h otherwise defines min/max as macros, which breaks every std::min/std::max and
  # std::numeric_limits<T>::max() in the codebase.
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
endif()

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
