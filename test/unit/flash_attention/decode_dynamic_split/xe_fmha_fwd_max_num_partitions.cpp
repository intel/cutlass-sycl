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
    \brief Host-side regression tests for compute_max_num_partitions().

    The persistent decode split-K kernel (XeFMHAFwdDynamicSplitKernel) reserves
    `max_num_partitions` partial-result slots per batch_head in its workspace.
    This value used to be a hardcoded `static const int = 8`, which overflows on
    large GPUs (e.g. PVC) when num_batch_heads is small, causing out-of-bounds
    workspace writes. It is now computed dynamically from sm_count and
    num_batch_heads. These tests guard that computation so the dynamic-partition
    path cannot silently regress back to a too-small constant.
*/

#include "cutlass_unit_test.h"

#include "cutlass/util/packed_stride.hpp"
#include "flash_attention_v2/kernel/xe_fmha_fwd_kernel.hpp"

using cutlass::fmha::kernel::compute_max_num_partitions;

// The old, buggy hardcoded value this change replaced.
static constexpr int kOldStaticMaxNumPartitions = 8;

// Formula under test: ceil_div(sm_count, max(1, num_batch_heads)) + 1.
// The +1 accounts for WG split boundaries not aligning to batch_head
// boundaries (a batch_head's KV blocks can span two WG allocation regions).

TEST(XE_FMHA_Fwd_MaxNumPartitions, PvcMax1550_OverflowsOldStaticBound) {
  // PVC Max 1550, single stack. CUTLASS sm_count comes from
  // gpu_slices * gpu_subslices_per_slice (an Xe-core / "subslice" count),
  // NOT max_compute_units (the EU count). A fully-enabled stack exposes 64
  // subslices; fused parts expose 56 (the value measured on our test card:
  // slices=1, subslices_per_slice=56). Both overflow the old static bound.
  // Using the measured 56 here with batch=1, num_heads_q=4:
  // ceil_div(56, 4) + 1 = 14 + 1 = 15.
  int const sm_count = 56;
  int const num_batch_heads = 1 * 4;
  int const max_parts = compute_max_num_partitions(sm_count, num_batch_heads);

  EXPECT_EQ(max_parts, 15);
  // This is exactly the bug: the old static bound of 8 is too small.
  EXPECT_GT(max_parts, kOldStaticMaxNumPartitions);

  // A fully-enabled (non-fused) PVC stack exposes 64 subslices:
  // ceil_div(64, 4) + 1 = 16 + 1 = 17. Still overflows the old static 8.
  EXPECT_EQ(compute_max_num_partitions(64, 4), 17);
  EXPECT_GT(compute_max_num_partitions(64, 4), kOldStaticMaxNumPartitions);
}

TEST(XE_FMHA_Fwd_MaxNumPartitions, BmgB580_SmallerThanPvc) {
  // BMG Arc B580: 20 XeCores, batch=1, num_heads_q=4.
  // ceil_div(20, 4) + 1 = 5 + 1 = 6. Demonstrates why a single static
  // constant cannot fit all hardware: it would overflow PVC or waste memory
  // here.
  EXPECT_EQ(compute_max_num_partitions(20, 4), 6);
}

TEST(XE_FMHA_Fwd_MaxNumPartitions, BoundaryCases) {
  // num_batch_heads == sm_count: every WG handles a distinct batch_head,
  // plus the +1 boundary slack => 2.
  EXPECT_EQ(compute_max_num_partitions(56, 56), 2);

  // num_batch_heads > sm_count (rejected by can_implement, but the math must
  // still be well-defined): ceil_div(56, 112) + 1 = 1 + 1 = 2.
  EXPECT_EQ(compute_max_num_partitions(56, 112), 2);

  // num_batch_heads == 0 is guarded by max(1, .): ceil_div(56, 1) + 1 = 57.
  EXPECT_EQ(compute_max_num_partitions(56, 0), 57);
}

TEST(XE_FMHA_Fwd_MaxNumPartitions, GeneralProperties) {
  for (int sm_count : {10, 20, 56, 64, 128}) {
    int prev = compute_max_num_partitions(sm_count, 1);
    // Must always reserve at least 2 slots (>=1 partition + boundary slack).
    EXPECT_GE(compute_max_num_partitions(sm_count, sm_count), 2);
    for (int nbh = 1; nbh <= sm_count; ++nbh) {
      int cur = compute_max_num_partitions(sm_count, nbh);
      // Always strictly positive and at least 2 (because of the +1).
      EXPECT_GE(cur, 2);
      // Non-increasing as the work is spread over more batch_heads.
      EXPECT_LE(cur, prev);
      prev = cur;
    }
  }
}
