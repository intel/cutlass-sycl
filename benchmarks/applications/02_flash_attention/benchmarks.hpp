/***************************************************************************************************
 * Copyright (c) 2026 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <benchmarks_decode_bf16.cpp>
#include <benchmarks_decode_fp8.cpp>
#include <benchmarks_decode_fp8kv_fp16mma.cpp>
#include <benchmarks_prefill_bf16.cpp>
#include <benchmarks_prefill_fp8.cpp>
#include <benchmarks_prefill_fp8kv_fp16mma.cpp>
#if defined(SYCL_INTEL_TARGET) && SYCL_INTEL_TARGET == 35
#include <benchmarks_prefill_mxfp8.cpp>
#include <benchmarks_prefill_mxfp4.cpp>
#endif

static void register_flash_attention_decode_benchmarks() {
  register_flash_attention_decode_benchmarks_bf16();
  register_flash_attention_decode_benchmarks_fp8();
  register_flash_attention_decode_benchmarks_fp8kv_fp16mma();
}

static void register_flash_attention_prefill_benchmarks() {
  register_flash_attention_prefill_benchmarks_bf16();
  register_flash_attention_prefill_benchmarks_fp8();
  register_flash_attention_prefill_benchmarks_fp8kv_fp16mma();
#if defined(SYCL_INTEL_TARGET) && SYCL_INTEL_TARGET == 35
  register_flash_attention_prefill_benchmarks_mxfp8();
  register_flash_attention_prefill_benchmarks_mxfp4();
#endif
}