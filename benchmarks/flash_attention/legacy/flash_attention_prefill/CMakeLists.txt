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

set(CUTLASS_APPLICATIONS_DIR ${CUTLASS_DIR}/applications)

# Pass these configuration files for the CI
set(CONFIG_FILE_PVC --config_file=${CUTLASS_DIR}/benchmarks/device/pvc/input_files/input_flash_attention_prefill.in)
set(CONFIG_FILE_BMG --config_file=${CUTLASS_DIR}/benchmarks/device/bmg/input_files/legacy/input_sglang_flash_attention_prefill_extend_nokvcache.in)
set(CONFIG_FILE_CRI --config_file=${CUTLASS_DIR}/benchmarks/device/cri/input_files/legacy/input_sglang_flash_attention_prefill_extend_nokvcache.in)

cutlass_benchmark_add_suite(cutlass_benchmarks_flash_attention_prefill_legacy
                            SUPERSUITE cutlass_benchmarks_flash_attention_legacy)

if(CUTLASS_TEST_FOR_CRI AND SYCL_TARGET_INTEL_GPU_CRI)
  set(PREFILL_LEGACY_CONFIG_ARGS CONFIG_FILE_CRI)
elseif(("intel_gpu_bmg_g21" IN_LIST DPCPP_SYCL_TARGET_LIST) OR
       ("intel_gpu_bmg_g31" IN_LIST DPCPP_SYCL_TARGET_LIST) OR
       ("bmg" IN_LIST DPCPP_SYCL_TARGET_LIST))
  set(PREFILL_LEGACY_CONFIG_ARGS CONFIG_FILE_BMG)
else()
  set(PREFILL_LEGACY_CONFIG_ARGS CONFIG_FILE_PVC CONFIG_FILE_BMG)
endif()

cutlass_benchmark_add_executable(
    cutlass_benchmarks_flash_attention_prefill_legacy_xe
    main.cpp
    TEST_COMMAND_OPTIONS
    ${PREFILL_LEGACY_CONFIG_ARGS}
    SUITE cutlass_benchmarks_flash_attention_prefill_legacy
)
