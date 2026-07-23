/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
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
    \brief Flash Attention V2 Prefill for Intel BMG

    This example constructs and executes a Flash Attention Prefill kernel on Intel BMG. The
    definition of the GEMM, options etc for this example are defined in the associated
    bmg_flash_attn_runner.hpp header file.

    See https://arxiv.org/pdf/2307.08691 for details of Flash Attention V2 algorithm

    To run one of the examples:
      $ ./examples/06_bmg_flash_attention/06_xe_fmha_fwd_prefill_fp8kvcachefp16mma_hdim128
        --seq_len_qo=1024 --seq_len_kv=1024 --num_heads_q=32 --num_heads_kv=8

    To build & run one example (from your build dir):

      $ ninja 06_xe_fmha_fwd_prefill_fp8kvcachefp16mma_hdim128
      $ ./examples/06_bmg_flash_attention/06_xe_fmha_fwd_prefill_fp8kvcachefp16mma_hdim128

    Call with `--help` for information about available options
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

  // Define the work-group tile shape depending on the head-size of the second matmul

#ifdef PREFILL
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
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;

#elif HEAD_DIM == 192
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _192>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;

#endif
#elif defined(DECODE)
#if HEAD_DIM == 16
  /* Tiny config for testing */
  using ShapeQK = Shape<_1, _16, _16>;       // (q,k,d)
  using ShapePV = Shape<_1, _16, _16>;       // (q,v,k)
  using ShapeOut = Shape<_1, _16>;           // (q,v)
  using SubgroupLayoutQK = Layout<Shape<_1, _2, _1>>;

#elif HEAD_DIM == 64
    using ShapeQK = Shape<_1, _512, _64>;
    using ShapePV = Shape<_1, _32, _512>;
    using ShapeOut = Shape<_1, _64>;
    using SubgroupLayoutQK = Layout<Shape<_1, _8, _1>>;

#elif HEAD_DIM == 96
    using ShapeQK = Shape<_1, _512, _32>;
    using ShapePV = Shape<_1, _32, _512>;
    using ShapeOut = Shape<_1, _96>;
    using SubgroupLayoutQK = Layout<Shape<_1, _8, _1>>;

#elif HEAD_DIM == 128
    using ShapeQK = Shape<_1, _512, _64>;
    using ShapePV = Shape<_1, _32, _512>;
    using ShapeOut = Shape<_1, _128>;
    using SubgroupLayoutQK = Layout<Shape<_1, _8, _1>>;

#elif HEAD_DIM == 192
    using ShapeQK = Shape<_1, _512, _64>;
    using ShapePV = Shape<_1, _32, _512>;
    using ShapeOut = Shape<_1, _192>;
    using SubgroupLayoutQK = Layout<Shape<_1, _8, _1>>;
#endif
#else
#error Either DECODE or PREFILL should be defined.
#endif

#ifdef DECODE
  constexpr int PipelineStages = 1;
#else
  constexpr int PipelineStages = 2;
#endif

  using ElementQ = cutlass::half_t;
  using ElementK = cutlass::float_e4m3_t;
  using ElementV = cutlass::float_e4m3_t;
  using ElementScale = float;

  using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>;
  using FMHACausal    = FMHAConfig<true, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages, false, ElementQ, ElementK, ElementV, ElementScale>;
  using FMHANonCausal = FMHAConfig<false, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages, false, ElementQ, ElementK, ElementV, ElementScale>;

  if (options.seq_len_kv_cache > 0 || options.use_paged_kv) {
    std::cerr << "Error: CachedKV/PagedKV requested. Use the cached_kv binary." << std::endl;
    return -1;
  }

  if (options.is_causal) {
    if (options.varlen) {
      return FMHACausal::template run<true, false, false, Scheduler>(options);
    } else {
      return FMHACausal::template run<false, false, false, Scheduler>(options);
    }
  } else {
    if (options.varlen) {
      return FMHANonCausal::template run<true, false, false, Scheduler>(options);
    } else {
      return FMHANonCausal::template run<false, false, false, Scheduler>(options);
    }
  }
}
