# Copyright (C) 2026 Intel Corporation, All rights reserved.
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

# GDN chunkwise attention is gated at the parent examples/CMakeLists.txt to
# CRI (Xe-3.5) and BMG (Xe20); PVC is excluded. This file is entered on both
# CRI and BMG builds.

if (CUTLASS_TEST_FOR_CRI)
  # CRI simulator: tiny single-sub-chunk shape, no warmup. Verification is
  # forced on (--verify=1) for GDN regardless of the global CRI_DEFAULT_VERIFY,
  # because the GDN inverse/state path is correctness-sensitive and the shape is
  # kept small enough to verify within the simulator timeout budget.
  set(GDN_ARGS "--batch=1 --num_v_heads=2 --num_k_heads=1 --seq_len=64 --iterations=${CRI_DEFAULT_ITERATIONS} --warmup=0 --verify=1")
else()
  # BMG (and other real hardware): empty args -> the runner iterates its
  # built-in case table with verify=1, giving full functional coverage.
  set(GDN_ARGS "")
endif()

# The GDN kernels and launcher are header-only under
# applications/gdn_attention/ (the launcher is an inline template instantiated
# here at the call site), shared with the unit tests and benchmarks. This
# example owns only the test-harness driver.
set(CUTLASS_APPLICATIONS_DIR ${CUTLASS_DIR}/applications)

# ---------------------------------------------------------------------------
# GDN attention executables
#
# The SSM recurrent state (StateT) is fixed at float.
# The activation dtype T (Q/K/V/O, dt_bias) is bfloat16_t.
#
# HEAD_K_DIM and HEAD_V_DIM only set the DEFAULT head dims (the kernel takes
# them as runtime args); override per-run with --head_k_dim / --head_v_dim.
# ACT_TYPE is a true compile-time constant: it selects the kernel template
# instantiation, so changing the activation dtype requires a rebuild.
#
# Targets:
#   14_xe35_gdn_attention_bfloat16   bf16 activations, fp32 state  (production default)
# ---------------------------------------------------------------------------

if(NOT "${DPCPP_HOST_COMPILER}" MATCHES "g\\+\\+")
  cutlass_example_add_executable(
    14_xe35_gdn_attention_bfloat16
    xe35_gdn_attention.cpp
    TEST_COMMAND_OPTIONS GDN_ARGS
  )
  target_compile_definitions(14_xe35_gdn_attention_bfloat16 PRIVATE
    HEAD_K_DIM=128
    HEAD_V_DIM=128
    ACT_TYPE=cutlass::bfloat16_t
  )
endif()
