/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
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
 *  \brief Unit tests for the Xe35 chunkwise Gated DeltaNet attention
 *  kernel. Exercises several control-flow regimes:
 *
 *    - seq_len=64  : one chunk per batch (single trip through the per-chunk
 *                    state-update loop in `chunk_fwd_o_kernel`).
 *    - seq_len=256 : four chunks per batch (state hand-off between chunks is
 *                    exercised, including the `has_prev = (c != 0)` branch).
 *    - multi-batch (batch>1) : multiple equal-length sequences, exercising
 *                    query_start_loc / cache_indices folding and per-batch
 *                    state slots (the example/benchmark batching).
 *    - kv_ratio=1/2 : num_v_heads / num_k_heads fan-out (identity and pairwise),
 *                    complementing the default kv_ratio=4 and the =64 probe.
 *    - alt seed    : default 4-chunk shape re-run with a different RNG seed.
 */

#include <gtest/gtest.h>

#include "cutlass/bfloat16.h"

#include "gdn_chunkwise_testbed.hpp"

namespace cutlass {

TEST(XE35_GDN_Chunkwise_bf16, seq_len_64) {
  test::gdn_attention::ChunkwiseTestbed<cutlass::bfloat16_t, float> tb;
  tb.num_v_heads = 16;
  tb.num_k_heads = 4;
  tb.seq_len = 64;
  EXPECT_TRUE(tb.run());
}

TEST(XE35_GDN_Chunkwise_bf16, seq_len_256) {
  test::gdn_attention::ChunkwiseTestbed<cutlass::bfloat16_t, float> tb;
  tb.seq_len = 256;
  EXPECT_TRUE(tb.run());
}

/* Xe-core occupancy probes on CRI (32 Xe-cores). Work unit = one
 * (chunk x v_head) tile = ceil(seq_len/64) * num_v_heads. The two shapes reach
 * a comparable tile count by opposite routes, covering both decompositions.
 *   head-parallel: 64 v_heads x 1 chunk = 64 tiles (single-chunk fast path). */
TEST(XE35_GDN_Chunkwise_bf16, occupancy_one_xe_core) {
  test::gdn_attention::ChunkwiseTestbed<cutlass::bfloat16_t, float> tb;
  tb.num_v_heads = 64;
  tb.num_k_heads = 1;
  tb.seq_len     = 64;
  EXPECT_TRUE(tb.run());
}



/* GQA grouping kv_ratio = 1 (num_v_heads == num_k_heads): every v-head reads
 * its own k-head, so the v_head_id / kv_ratio key-head folding is the identity.
 * Complements the kv_ratio = 4 / 64 shapes above.
 *
 * NOTE: removable -- if kv_ratio=1 is later found unsupported on the kernel,
 * this single TEST can be deleted with no impact on the others (it is the only
 * one that sets num_v_heads == num_k_heads). */
TEST(XE35_GDN_Chunkwise_bf16, kv_ratio_one) {
  test::gdn_attention::ChunkwiseTestbed<cutlass::bfloat16_t, float> tb;
  tb.num_v_heads = 8;
  tb.num_k_heads = 8;
  tb.seq_len     = 64;
  EXPECT_TRUE(tb.run());
}

/* GQA grouping kv_ratio = 2 (num_v_heads = 2 * num_k_heads): an intermediate
 * fan-out between the identity (kv_ratio=1) and the kv_ratio=4/64 shapes, so
 * the v_head_id / kv_ratio folding maps head pairs onto each k-head. */
TEST(XE35_GDN_Chunkwise_bf16, kv_ratio_two) {
  test::gdn_attention::ChunkwiseTestbed<cutlass::bfloat16_t, float> tb;
  tb.num_v_heads = 8;
  tb.num_k_heads = 4;
  tb.seq_len     = 64;
  EXPECT_TRUE(tb.run());
}


/* Uniform multi-batch: batch>1 with equal per-sequence lengths. Exercises the
 * query_start_loc / cache_indices folding and per-batch state slots with more
 * than one sequence -- the batch=1 shapes above never reach this. Equal,
 * chunk-aligned lengths, matching how the example/benchmark runners batch. */
TEST(XE35_GDN_Chunkwise_bf16, multi_batch_uniform) {
  test::gdn_attention::ChunkwiseTestbed<cutlass::bfloat16_t, float> tb;
  tb.batch       = 4;
  tb.num_v_heads = 16;
  tb.num_k_heads = 4;
  tb.seq_len     = 128;  // 2 chunks per sequence
  EXPECT_TRUE(tb.run());
}

/* Seed-variation robustness: the default shape (seq_len=256, 4 chunks) re-run
 * with a different RNG seed, to catch data-dependent failures that a single
 * fixed seed would miss. Cheap -- reuses an already-covered control-flow path
 * with fresh inputs. */
TEST(XE35_GDN_Chunkwise_bf16, seq_len_256_alt_seed) {
  test::gdn_attention::ChunkwiseTestbed<cutlass::bfloat16_t, float> tb;
  tb.seq_len = 256;
  tb.seed    = 0x1234ABCDu;
  EXPECT_TRUE(tb.run());
}



/* Two-sequence minimal batch: smallest batch> (the simplest cache_indices /
 * query_start_loc fold), default GQA ratio, single chunk per sequence. */
TEST(XE35_GDN_Chunkwise_bf16, batch_two_single_chunk) {
  test::gdn_attention::ChunkwiseTestbed<cutlass::bfloat16_t, float> tb;
  tb.batch       = 2;
  tb.num_v_heads = 4;
  tb.num_k_heads = 1;
  tb.seq_len     = 64;
  EXPECT_TRUE(tb.run());
}


}  // namespace cutlass
