# Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

include_guard()

include(CheckCXXCompilerFlag)
include(FindPackageHandleStandardArgs)

set(DPCPP_USER_FLAGS "" CACHE STRING "Additional user-specified compiler flags for DPC++")

# DPCLANG is an open source DPC++ compiler shipped with Linux distros
function(parse_dpclang_version major minor patch)
  # Execute the SYCL compiler with the --version flag to match the version string.
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE SYCL_VERSION_STRING)
  string(REGEX REPLACE "DPC\\+\\+ compiler ([0-9]+\\.[0-9]+\\.[0-9]+) (.*)" "\\1"
               SYCL_VERSION_STRING_MATCH ${SYCL_VERSION_STRING})
  string(REPLACE "." ";" SYCL_VERSION_LIST ${SYCL_VERSION_STRING_MATCH})
  # Split the version number list into major, minor, and patch components.
  list(GET SYCL_VERSION_LIST 0 VERSION_MAJOR)
  list(GET SYCL_VERSION_LIST 1 VERSION_MINOR)
  list(GET SYCL_VERSION_LIST 2 VERSION_PATCH)
  set(${major} "${VERSION_MAJOR}" PARENT_SCOPE)
  set(${minor} "${VERSION_MINOR}" PARENT_SCOPE)
  set(${patch} "${VERSION_PATCH}" PARENT_SCOPE)
endfunction()

cmake_path(GET CMAKE_CXX_COMPILER FILENAME COMPILER_NAME)
if (COMPILER_NAME MATCHES "dpclang")
  find_package(PkgConfig REQUIRED)
  parse_dpclang_version(
    SYCL_COMPILER_VERSION_MAJOR
    SYCL_COMPILER_VERSION_MINOR
    SYCL_COMPILER_VERSION_PATCH)
 pkg_check_modules(LIBSYCL REQUIRED IMPORTED_TARGET sycl-dpcpp-${SYCL_COMPILER_VERSION_MAJOR})
else()
  # LIBSYCL_INCLUDE_DIRS and LIBSYCL_LINK_LIBRARIES variable names are algined with
  # the variables set by pkg_check_modules in the "dpclang" branch above.
  get_filename_component(DPCPP_BIN_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
  if (UNIX)
    set(LIBSYCL_INCLUDE_DIRS "${DPCPP_BIN_DIR}/../include/sycl;${DPCPP_BIN_DIR}/../include")
  else()
    set(LIBSYCL_INCLUDE_DIRS "${DPCPP_BIN_DIR}/../include/sycl")
  endif()
  find_library(LIBSYCL_LINK_LIBRARIES NAMES sycl sycl6 PATHS "${DPCPP_BIN_DIR}/../lib")
endif()

add_library(DPCPP::DPCPP INTERFACE IMPORTED)

set(DPCPP_FLAGS "-fsycl;")
if(DPCPP_HOST_COMPILER)
  list(APPEND DPCPP_FLAGS "-fsycl-host-compiler=${DPCPP_HOST_COMPILER}")
  set(_host_opts "-Wno-changes-meaning $<$<BOOL:$<TARGET_PROPERTY:POSITION_INDEPENDENT_CODE>>:-fPIC> -D$<JOIN:$<TARGET_PROPERTY:COMPILE_DEFINITIONS>, -D> -I$<JOIN:$<TARGET_PROPERTY:INCLUDE_DIRECTORIES>, -I>")
  if(DEFINED DPCPP_HOST_COMPILER_OPTIONS AND NOT "${DPCPP_HOST_COMPILER_OPTIONS}" STREQUAL "")
    set(_host_opts "${DPCPP_HOST_COMPILER_OPTIONS} ${_host_opts}")
    string(STRIP "${_host_opts}" _host_opts)
  endif()
  list(APPEND DPCPP_FLAGS "-fsycl-host-compiler-options=${_host_opts}")
endif()
set(DPCPP_COMPILE_ONLY_FLAGS "")
set(DPCPP_LINK_ONLY_FLAGS "")

option(DPCPP_DISABLE_ITT_FOR_CUTLASS "Disables linking of the Instrumentation and Tracing Technology (ITT) device libraries for VTune" ON)

if(NOT "${DPCPP_USER_FLAGS}" STREQUAL "")
  list(APPEND DPCPP_FLAGS "${DPCPP_USER_FLAGS};")
endif()

string(REPLACE "," ";" DPCPP_SYCL_TARGET_LIST "${DPCPP_SYCL_TARGET}")

if(NOT "${DPCPP_SYCL_ARCH}" STREQUAL "")
  if(SYCL_NVIDIA_TARGET)
    list(APPEND DPCPP_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda;")
    list(APPEND DPCPP_FLAGS "-Xsycl-target-backend")
    list(APPEND DPCPP_FLAGS "--cuda-gpu-arch=${DPCPP_SYCL_ARCH}")
    list(APPEND DPCPP_COMPILE_ONLY_FLAGS; "-mllvm;-enable-global-offset=false;")
  endif()
endif()

if (SYCL_INTEL_TARGET)
  if(DPCPP_DISABLE_ITT_FOR_CUTLASS)
    list(APPEND DPCPP_FLAGS "-fno-sycl-instrument-device-code")
  endif()

  set(SYCL_DEVICES)

  # For multitarget build, set target as spir64_gen and if user gave spir64, then overwrite it.
  set(SYCL_TARGET "spir64_gen")

  list(LENGTH DPCPP_SYCL_TARGET_LIST SYCL_TARGET_COUNT)
  if(SYCL_TARGET_COUNT GREATER 1)
    list(FIND DPCPP_SYCL_TARGET_LIST "spir64" _spir64_index)
    if(_spir64_index GREATER -1)
      message(FATAL_ERROR "MultiTarget Build is not supported if one of target is spir64.")
    endif()
  endif()

  foreach(TGT IN LISTS DPCPP_SYCL_TARGET_LIST)
    if(TGT STREQUAL "bmg")
      list(APPEND SYCL_DEVICES "bmg-g21")
      list(APPEND SYCL_DEVICES "bmg-g31")
    elseif(TGT STREQUAL "intel_gpu_bmg_g21")
      list(APPEND SYCL_DEVICES "bmg-g21")
    elseif(TGT STREQUAL "intel_gpu_bmg_g31")
      list(APPEND SYCL_DEVICES "bmg-g31")
    elseif(TGT STREQUAL "intel_gpu_pvc" OR TGT STREQUAL "pvc")
      list(APPEND SYCL_DEVICES "pvc")
    elseif(TGT STREQUAL "intel_gpu_cri" OR TGT STREQUAL "cri")
      list(APPEND SYCL_DEVICES "cri")
      list(APPEND DPCPP_FLAGS "-D__SYCL_TARGET_INTEL_GPU_CRI__")
    elseif(TGT STREQUAL "spir64")
      set(SYCL_TARGET "spir64")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES SYCL_DEVICES)

  string(JOIN "," SYCL_DEVICES_STR ${SYCL_DEVICES})

  list(APPEND DPCPP_FLAGS "-fsycl-targets=${SYCL_TARGET}")
  list(APPEND DPCPP_LINK_ONLY_FLAGS "-Xsycl-target-backend=${SYCL_TARGET};-device ${SYCL_DEVICES_STR}")

  list(APPEND DPCPP_LINK_ONLY_FLAGS "-Xspirv-translator")

  if((CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_LESS 2025.2) OR CUTLASS_SYCL_BUILTIN_ENABLE)
    set(SPIRV_EXT "+SPV_INTEL_split_barrier")
  else()
    set(SPIRV_EXT "+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate")
  endif()
  list(APPEND DPCPP_LINK_ONLY_FLAGS "-spirv-ext=${SPIRV_EXT}")

endif()

set_target_properties(DPCPP::DPCPP PROPERTIES
  INTERFACE_COMPILE_OPTIONS "${DPCPP_FLAGS};${DPCPP_COMPILE_ONLY_FLAGS}"
  INTERFACE_LINK_OPTIONS "${DPCPP_FLAGS};${DPCPP_LINK_ONLY_FLAGS}"
  INTERFACE_LINK_LIBRARIES ${LIBSYCL_LINK_LIBRARIES}
  INTERFACE_INCLUDE_DIRECTORIES "${LIBSYCL_INCLUDE_DIRS}")
message(STATUS "DPCPP INCLUDE DIR: ${LIBSYCL_INCLUDE_DIRS}")
message(STATUS "Using DPCPP compile flags: ${DPCPP_FLAGS};${DPCPP_COMPILE_ONLY_FLAGS}")
message(STATUS "Using DPCPP link flags: ${DPCPP_FLAGS};${DPCPP_LINK_ONLY_FLAGS}")

function(add_sycl_to_target)
  set(options)
  set(oneValueArgs TARGET)
  set(multiValueArgs SOURCES)
  cmake_parse_arguments(CUTLASS_ADD_SYCL
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )
  target_compile_options(
    ${CUTLASS_ADD_SYCL_TARGET}
    PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:${DPCPP_FLAGS}>
  )
  get_target_property(target_type ${CUTLASS_ADD_SYCL_TARGET} TYPE)
  if (NOT target_type STREQUAL "OBJECT_LIBRARY")
    target_link_options(${CUTLASS_ADD_SYCL_TARGET} PUBLIC ${DPCPP_FLAGS} ${DPCPP_LINK_ONLY_FLAGS})
  endif()
endfunction()

function(add_sycl_include_directories_to_target NAME)
  target_include_directories(${NAME} SYSTEM
    PUBLIC ${LIBSYCL_INCLUDE_DIRS}
  )
endfunction()
