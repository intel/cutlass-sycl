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
    \brief Flash Attention V2 with Cached KV for Intel BMG

    This file instantiates only the CachedKV=true kernel variants,
    split out from the main 06_xe_fmha_fwd.cpp to reduce per-binary compile time.

    Instantiated kernels (8 total):
      - Causal × {true, false}
      - VarLen × {true, false}
      - CachedKV = true
      - PagedKV × {true, false}

    To build & run (from your build dir):
      $ ninja 06_xe_fmha_fwd_prefill_cached_kv_bfloat16_t_hdim128
      $ ./examples/sycl/06_bmg_flash_attention/06_xe_fmha_fwd_prefill_cached_kv_bfloat16_t_hdim128 \
            --seq_len_kv_cache=256
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
  using ElementQ = cutlass::float_e5m2_t;
  using ElementK = cutlass::float_e5m2_t;
  using ElementV = cutlass::float_e5m2_t;
#elif defined(IS_FLOAT_E4M3)
  using ElementQ = cutlass::float_e4m3_t;
  using ElementK = cutlass::float_e4m3_t;
  using ElementV = cutlass::float_e4m3_t;
#elif defined(IS_FLOAT_E2M1)
  using ElementQ = cutlass::float_e2m1_t;
  using ElementK = cutlass::float_e2m1_t;
  using ElementV = cutlass::bfloat16_t;
#else
  using ElementQ = bfloat16_t;
  using ElementK = bfloat16_t;
  using ElementV = bfloat16_t;
#endif

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
#if !(defined(SYCL_INTEL_TARGET) && (SYCL_INTEL_TARGET == 35))
  using ShapeQK = Shape<_256, _32, _32>;
  using ShapePV = Shape<_256, _32, _32>;
  using ShapeOut = Shape<_256, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#else
  using ShapeQK = Shape<_256, _64, _64>;
  using ShapePV = Shape<_256, _64, _64>;
  using ShapeOut = Shape<_256, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#endif
#elif HEAD_DIM == 192
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _192>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;

#endif
#elif defined(DECODE)

#define NUM_SG _8
#define KV_TILE_SIZE _512

#if HEAD_DIM == 16
  /* Tiny config for testing */
  using ShapeQK = Shape<_1, _16, _16>;       // (q,k,d)
  using ShapePV = Shape<_1, _16, _16>;       // (q,v,k)
  using ShapeOut = Shape<_1, _16>;           // (q,v)
  using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;

#elif HEAD_DIM == 64
    using ShapeQK = Shape<_1, KV_TILE_SIZE, _64>;
    using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
    using ShapeOut = Shape<_1, _64>;
    using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;

#elif HEAD_DIM == 96
    using ShapeQK = Shape<_1, KV_TILE_SIZE, _32>;
    using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
    using ShapeOut = Shape<_1, _96>;
    using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;

#elif HEAD_DIM == 128
    using ShapeQK = Shape<_1, KV_TILE_SIZE, _64>;
    using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
    using ShapeOut = Shape<_1, _128>;
    using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;

#elif HEAD_DIM == 192
    using ShapeQK = Shape<_1, KV_TILE_SIZE, _64>;
    using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
    using ShapeOut = Shape<_1, _192>;
    using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;
#endif
#else
#error Either DECODE or PREFILL should be defined.
#endif

#ifdef DECODE
  constexpr int PipelineStages = 1;
#else
  constexpr int PipelineStages = 2;
#endif

  // Directly instantiate only CachedKV=true, PagedKV=false kernels.
  // Causal and VarLen are dispatched at runtime.
  // BlockScale (mxfp) is not supported with CachedKV.
  using Scheduler = cutlass::fmha::kernel::XeFHMAIndividualTileScheduler<>;

  using FMHACausal    = FMHAConfig<true, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages, false, ElementQ, ElementK, ElementV>;
  using FMHANonCausal = FMHAConfig<false, false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages, false, ElementQ, ElementK, ElementV>;

  if (options.seq_len_kv_cache <= 0) {
    std::cerr << "Error: seq_len_kv_cache must be > 0 for the cached_kv binary." << std::endl;
    return -1;
  }

  if (options.is_causal) {
    if (options.use_paged_kv && options.varlen) {
      return FMHACausal::template run<true, true, true, Scheduler>(options);
    } else if (options.use_paged_kv && !options.varlen) {
      return FMHACausal::template run<false, true, true, Scheduler>(options);
    } else if (!options.use_paged_kv && options.varlen) {
      return FMHACausal::template run<true, true, false, Scheduler>(options);
    } else {
      return FMHACausal::template run<false, true, false, Scheduler>(options);
    }
  } else {
    if (options.use_paged_kv && options.varlen) {
      return FMHANonCausal::template run<true, true, true, Scheduler>(options);
    } else if (options.use_paged_kv && !options.varlen) {
      return FMHANonCausal::template run<false, true, true, Scheduler>(options);
    } else if (!options.use_paged_kv && options.varlen) {
      return FMHANonCausal::template run<true, true, false, Scheduler>(options);
    } else {
      return FMHANonCausal::template run<false, true, false, Scheduler>(options);
    }
  }
}
