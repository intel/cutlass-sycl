# Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
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

# Pass these configuration files for the CI
set(CONFIG_FILE_BMG_SGLANG --config_file=${CUTLASS_DIR}/benchmarks/device/bmg/input_files/legacy/input_sglang_gemm.in)
set(CONFIG_FILE_BMG_SGLANG_SPLITK --config_file=${CUTLASS_DIR}/benchmarks/device/bmg/input_files/legacy/input_sglang_gemm_splitk.in)
set(CONFIG_FILE_BMG_MIXED_DTYPE --config_file=${CUTLASS_DIR}/benchmarks/device/bmg/input_files/legacy/input_sglang_gemm_mixed_dtype.in)
set(CONFIG_FILE_CRI_SGLANG --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/legacy/input_sglang_gemm.in)
set(CONFIG_FILE_PVC_DEFAULT --config_file=${CUTLASS_DIR}/benchmarks/device/pvc/input_files/input_gemm.in)
set(CONFIG_FILE_PVC_LINEAR --config_file=${CUTLASS_DIR}/benchmarks/device/pvc/input_files/input_linear.in)
set(CONFIG_FILE_PVC_PYTORCH --config_file=${CUTLASS_DIR}/benchmarks/device/pvc/input_files/input_pytorch_2.in)

set(CONFIG_FILE_CUDA --config_file=${CUTLASS_DIR}/benchmarks/device/ampere/input_files/input_gemm.in)

cutlass_benchmark_add_suite(cutlass_benchmarks_gemm_legacy)

if(CUTLASS_ENABLE_SYCL)

if(SYCL_INTEL_TARGET)

if(CUTLASS_TEST_FOR_CRI AND SYCL_TARGET_INTEL_GPU_CRI)
  # CRI: lightweight sglang-only benchmark
  set(INTEL_LEGACY_CONFIG_ARGS CONFIG_FILE_CRI_SGLANG)
elseif(("intel_gpu_bmg_g21" IN_LIST DPCPP_SYCL_TARGET_LIST) OR ("intel_gpu_bmg_g31" IN_LIST DPCPP_SYCL_TARGET_LIST) OR ("bmg" IN_LIST DPCPP_SYCL_TARGET_LIST))
  # BMG: only run BMG-specific sglang benchmarks (PVC shapes OOM on ~12-14GB VRAM)
  set(INTEL_LEGACY_CONFIG_ARGS CONFIG_FILE_BMG_SGLANG CONFIG_FILE_BMG_SGLANG_SPLITK CONFIG_FILE_BMG_MIXED_DTYPE)
else()
  # PVC and others: full benchmark suite including large PVC shapes
  set(INTEL_LEGACY_CONFIG_ARGS CONFIG_FILE_PVC_DEFAULT CONFIG_FILE_PVC_PYTORCH CONFIG_FILE_BMG_SGLANG CONFIG_FILE_BMG_SGLANG_SPLITK CONFIG_FILE_BMG_MIXED_DTYPE)
endif()

cutlass_benchmark_add_executable(
    cutlass_benchmarks_gemm_sycl_legacy
    main.cpp
    SUITE cutlass_benchmarks_gemm_legacy
    TEST_COMMAND_OPTIONS
    ${INTEL_LEGACY_CONFIG_ARGS}
)

else()
cutlass_benchmark_add_executable(
    cutlass_benchmarks_gemm_sycl_legacy
    main.cpp
    SUITE cutlass_benchmarks_gemm_legacy
    TEST_COMMAND_OPTIONS
    CONFIG_FILE_CUDA
)
endif()

else()

cutlass_benchmark_add_executable(
    cutlass_benchmarks_gemm_cuda_legacy
    main.cu
    SUITE cutlass_benchmarks_gemm_legacy
    TEST_COMMAND_OPTIONS
    CONFIG_FILE_CUDA
)

endif()
