/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
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
/*! \file
    \brief Flash Attention V2 Prefill with LSE output & Sequence-Parallel combine
           for Intel BMG

    This example exercises the LSE output feature of the FMHA forward kernel
    (float *Lse in the kernel arguments, written by FMHAFwdEpilogue). It
    performs a sequence-parallel (SP) validation:

      * The full KV sequence is split into `--seq_chunk_len`-sized chunks.
      * For every chunk the kernel is launched with seq_len_kv = chunk length,
        producing a partial output O_c and a partial log-sum-exp LSE_c.
      * The partials are combined on the host using only the LSE values:
            M   = max_c  LSE_c
            w_c = exp(LSE_c - M)
            O   = sum_c (w_c * O_c) / sum_c w_c
      * The combined output is compared against a full-sequence reference, the
        kernel LSE is compared against a host log-sum-exp reference, and the
        performance of the single full launch vs the chunked SP launches is
        reported.

    Build & run (from your build dir):
      $ ninja 06_xe_fmha_fwd_lse_bfloat16_t_hdim128
      $ ./examples/sycl/06_bmg_flash_attention/06_xe_fmha_fwd_lse_bfloat16_t_hdim128 \
            --seq_len_qo=512 --seq_len_kv=512 --seq_chunk_len=128

    Call with `--help` for information about available options.
*/

#include "xe_fmha_fwd_runner.hpp"

int main(int argc, const char **argv) {
  //
  // Parse options
  //

  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

#ifdef IS_FLOAT_E5M2
#error "LSE / sequence-parallel example currently only supports bfloat16 inputs."
#elif defined(IS_FLOAT_E4M3)
#error "LSE / sequence-parallel example currently only supports bfloat16 inputs."
#elif defined(IS_FLOAT_E2M1)
#error "LSE / sequence-parallel example currently only supports bfloat16 inputs."
#else
  using ElementQ = bfloat16_t;
  using ElementK = bfloat16_t;
  using ElementV = bfloat16_t;
#endif

  // Sequence-parallel LSE example: PREFILL only, no causal / varlen / cached KV.
#ifndef PREFILL
#error "The LSE example must be compiled with PREFILL defined."
#endif

  if (options.is_causal) {
    std::cerr << "Error: Sequence-parallel LSE example does not support --is_causal yet." << std::endl;
    return -1;
  }
  if (options.varlen) {
    std::cerr << "Error: Sequence-parallel LSE example does not support --varlen yet." << std::endl;
    return -1;
  }
  if (options.seq_len_kv_cache > 0 || options.use_paged_kv) {
    std::cerr << "Error: Sequence-parallel LSE example does not support cached/paged KV." << std::endl;
    return -1;
  }

 // Define the work-group tile shape depending on the head-size of the second matmul
#if HEAD_DIM == 16
  /* Tiny config for testing */
  using ShapeQK = Shape<_16, _16, _32>;       // (q,k,d)
  using ShapePV = Shape<_16, _32, _16>;       // (q,v,k)
  using ShapeOut = Shape<_16, _16>;           // (q,v)
  using SubgroupLayoutQK = Layout<Shape<_1, _1, _1>>;

#elif HEAD_DIM == 64
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _64>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;

#elif HEAD_DIM == 96
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _96>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;

#elif HEAD_DIM == 128
#if !(defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
  using ShapeQK = Shape<_256, _32, _32>;
  using ShapePV = Shape<_256, _32, _32>;
  using ShapeOut = Shape<_256, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#else
  using ShapeQK = Shape<_512, _64, _64>;
  using ShapePV = Shape<_512, _64, _64>;
  using ShapeOut = Shape<_512, _128>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
#endif
#elif HEAD_DIM == 192
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _192>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;

#else
#error "Unsupported HEAD_DIM"
#endif

  constexpr int PipelineStages = 2;

  using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>;

  // Sequence-parallel LSE path: non-causal, non-varlen, no cache, no paged KV.
  using FMHANonCausal = FMHAConfig<false, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages, false, ElementQ, ElementK, ElementV>;

  return FMHANonCausal::template run_lse<false, false, false, Scheduler>(options);
}
