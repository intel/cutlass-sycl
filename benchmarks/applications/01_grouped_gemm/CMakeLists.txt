# Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
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

# The `grouped_gemm` benchmark suite is served by the MoE grouped-GEMM
# infrastructure (all sources now live in this directory). It builds:
#   * cutlass_benchmarks_grouped_gemm_sycl  — the MERGED all-dtype binary
#     (bf16 + fp8-tensor + mxfp8-e4m3 + mxfp8-e5m2 + mxfp4, all tile shapes),
#     selecting dtype+tile at runtime from the .in line's first token. This is
#     the target higher-level tooling (gpuforge benchmark run grouped_gemm, the
#     parallel benchmarking harness) calls by name, out of the box.
#   * cutlass_benchmarks_moe_gemm_<dtype>_sycl — one binary per dtype, for
#     fine-grained builds (a single dtype's device image).
#
# All targets share main.cpp + moe_benchmark_runner.hpp (lean Google-Benchmark
# TU) and moe_kernel_launch.cpp (the only TU that instantiates the MoE::MoEGEMM
# device kernel). The dtype selection is a build define:
#   per-dtype: -DMOE_BENCH_CONFIG=<Config> -DMOE_DTYPE_<TAG>
#   merged:    -DMOE_BENCH_ALL (enables every MOE_DTYPE_<TAG>; moe_tile_list.hpp
#              concatenates all dtypes' tiles into the one name->config table).
# Historically the merged image tripped an IGC ICE (SIGSEGV in ocloc, -device
# cri) — the reason for the per-dtype split — but it compiles cleanly on the
# current toolchain.

# Config files for the CI.
if(CUTLASS_TEST_FOR_CRI AND SYCL_TARGET_INTEL_GPU_CRI)
  set(CONFIG_FILE_CRI_GROUPED --config_file=${CUTLASS_DIR}/benchmarks/config_files/01_grouped_gemm/cri/small.in)
  set(CONFIG_FILE_CRI_MOE --config_file=${CUTLASS_DIR}/benchmarks/config_files/01_grouped_gemm/cri/moe_small.in)
  # ByteDance real-scenario LLM shapes from xpu-perf/micro_perf workloads
  set(CONFIG_FILE_BYTEDANCE --config_file=${CUTLASS_DIR}/benchmarks/config_files/01_grouped_gemm/cri/test1/small.in)
else()
  set(CONFIG_FILE_BMG_GROUPED --config_file=${CUTLASS_DIR}/benchmarks/config_files/01_grouped_gemm/bmg/small.in)
endif()

cutlass_benchmark_add_suite(cutlass_benchmarks_grouped_gemm)

# oneMKL headers at directory scope: the shared benchmark TU transitively
# includes sycl_tensor_fill.h -> <oneapi/mkl/rng/device.hpp>; a SYSTEM oneMKL
# only propagates via MKL::MKL, which does not always reach these targets.
# MKL_INCLUDE is set only for a system oneMKL, so this is a no-op in-tree.
if(MKL_INCLUDE)
  include_directories(${MKL_INCLUDE})
endif()

if(CUTLASS_ENABLE_SYCL)

# The hand-written MoE grouped-GEMM kernel (and its low-precision block-scaled
# variants) currently target CRI / Xe3p only.
if(DEFINED SYCL_INTEL_TARGET AND SYCL_TARGET_INTEL_GPU_CRI)

# ---------------------------------------------------------------------------
# Per-dtype binaries (one dtype's device image each). VARIANT fields:
#   <target_suffix> <Config type> <registered benchmark name> <MOE_DTYPE_<TAG>>
# Field 4 selects the dtype's tile list in moe_tile_list.hpp.
set(MOE_BENCH_VARIANTS
    "bf16|Bf16Config|MoE_BF16|MOE_DTYPE_BF16"
    "mxfp8_e4m3|MxFp8E4m3Config|MoE_MXFP8_E4M3|MOE_DTYPE_MXFP8_E4M3"
    "mxfp8_e5m2|MxFp8E5m2Config|MoE_MXFP8_E5M2|MOE_DTYPE_MXFP8_E5M2"
    "mxfp4_e2m1|MxFp4E2m1Config|MoE_MXFP4_E2M1|MOE_DTYPE_MXFP4_E2M1"
    "fp8_tensor_e4m3|Fp8TensorE4m3Config|MoE_FP8_TENSOR_E4M3|MOE_DTYPE_FP8_TENSOR_E4M3")

foreach(VARIANT ${MOE_BENCH_VARIANTS})
  string(REPLACE "|" ";" VPARTS ${VARIANT})
  list(GET VPARTS 0 VSUFFIX)
  list(GET VPARTS 1 VCONFIG)
  list(GET VPARTS 2 VNAME)
  list(GET VPARTS 3 VDTYPE)

  cutlass_benchmark_add_executable(
      cutlass_benchmarks_moe_gemm_${VSUFFIX}_sycl
      main.cpp
      moe_kernel_launch.cpp
      SUITE cutlass_benchmarks_grouped_gemm
      INCLUDES
      ${CUTLASS_DIR}
      ${CUTLASS_DIR}/applications
      TEST_COMMAND_OPTIONS
      CONFIG_FILE_CRI_MOE
  )
  # CUTE_ENABLE_XE_BLOCK_2D_ASSERT: debug-only asserts for the MX-scale 2D-block
  # load (xe_mma_blockscaled_scale_traits.hpp / copy_traits_xe_2d.hpp) — catch
  # M=0 token groups and mis-aligned/oversized scale width/pitch/base-ptr on the
  # Xe lsc_load_block2d path. Opt-in and compiled out under NDEBUG, so no release
  # perf cost.
  target_compile_definitions(cutlass_benchmarks_moe_gemm_${VSUFFIX}_sycl
      PRIVATE MOE_BENCH_CONFIG=${VCONFIG} MOE_BENCH_NAME=${VNAME} ${VDTYPE}
              CUTE_ENABLE_XE_BLOCK_2D_ASSERT)
  if(NOT DPCPP_SYCL_TARGET STREQUAL "spir64")
    target_link_options(cutlass_benchmarks_moe_gemm_${VSUFFIX}_sycl PRIVATE
        -Xs "-options \"-igc_opts 'VectorAliasBBThreshold=10000'\"")
  endif()
endforeach()

# ---------------------------------------------------------------------------
# MERGED all-dtype binary, built under the grouped_gemm name so the tooling
# calls it out of the box. -DMOE_BENCH_ALL enables every MOE_DTYPE_<TAG>.
cutlass_benchmark_add_executable(
    cutlass_benchmarks_grouped_gemm_sycl
    main.cpp
    moe_kernel_launch.cpp
    SUITE cutlass_benchmarks_grouped_gemm
    INCLUDES
    ${CUTLASS_DIR}
    ${CUTLASS_DIR}/applications
    TEST_COMMAND_OPTIONS
    CONFIG_FILE_CRI_GROUPED
    CONFIG_FILE_CRI_MOE
    CONFIG_FILE_BYTEDANCE
)
target_compile_definitions(cutlass_benchmarks_grouped_gemm_sycl
    PRIVATE MOE_BENCH_ALL CUTE_ENABLE_XE_BLOCK_2D_ASSERT)
if(NOT DPCPP_SYCL_TARGET STREQUAL "spir64")
  target_link_options(cutlass_benchmarks_grouped_gemm_sycl PRIVATE
      -Xs "-options \"-igc_opts 'VectorAliasBBThreshold=10000'\"")
endif()

endif()

endif()
