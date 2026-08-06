/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*!
  \file xe35_chunk_gated_delta_rule_gemm.hpp
  \brief CuTe GEMM helpers for the five GDN attention kernels (namespace cutlass::gdn::detail).

  Each helper computes  C += op(A) * op(B)  using Intel Xe 2D-block loads and DPAS,
  with a software prefetch pipeline (3 tiles ahead) to hide global-memory latency.

  Naming convention -- each letter describes one operand's storage:
    T = Tensor  : lives in global memory, loaded tile-by-tile each call
    S = Sub-group fragment : already in registers from a prior GEMM result

  Variants:
    gemm_TTS         : A(gmem) * B(gmem)          -> accumulate into C(regs)
    gemm_STS         : A(regs, kept from prior)   * B(gmem) -> C(regs)
    gemm_TSS         : A(gmem) * B(regs, kept)    -> C(regs)
    gemm_TTS_k_multi : same as TTS, but each k-slice of A is pre-scaled by
                       a per-lane float from an SLM array (used for diagonal
                       scaling in compute_wu / fwd_o).

  All helpers accumulate into the caller's register fragment tCrC, so
  multiple calls can be chained (C += A1*B1 + A2*B2 ...) without intermediate
  global-memory stores.
*/

#pragma once

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace cutlass::gdn::detail {

using namespace cute;

template <
    class ATensor,
    class BTensor,
    class SGCTensor,
    class TiledMMA>
/* gemm_TTS: C += A(gmem, M×K) * B(gmem, N×K)^T  -- both operands from global memory. */
CUTE_DEVICE void gemm_TTS(
    ATensor const& A,  // (M,K)
    BTensor const& B,  // (N,K)
    SGCTensor& tCrC,   // (M,N)
    int wg_m,          // m tile start id
    int wg_n,          // n tile start id
    TiledMMA const& mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());

  auto wg_tile = mma.tile_mnk();

  Tensor gA = local_tile(
      cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(
      cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)

  auto copy_a = get_block_2d_copy_A<void>(mma, A);
  auto copy_b = get_block_2d_copy_B<void>(mma, B);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  const int prefetch_dist = 3;

  constexpr auto barrier_scope = SPIRVScope::ScopeWorkgroup;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }
}

template <
    class ASGCTensor,
    class BTensor,
    class CSGCTensor,
    class TiledMMA>
/* gemm_STS: C += A(regs, M×K) * B(gmem, N×K)^T  -- A is reused from a prior gemm_TTS call. */
CUTE_DEVICE void gemm_STS(
    ASGCTensor const& tCrA,  // (M,K)
    BTensor const& B,        // (N,K)
    CSGCTensor& tCrC,        // (M,N)
    int wg_m,                // m tile start id
    int wg_n,                // n tile start id
    TiledMMA const& mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();

  Tensor cB = make_identity_tensor(B.shape());

  auto wg_tile = mma.tile_mnk();

  Tensor gB = local_tile(
      cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)

  auto copy_b = get_block_2d_copy_B<void>(mma, B);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pBgB = thr_prefetch_B.partition_S(gB);

  const int prefetch_dist = 3;

  constexpr auto barrier_scope = SPIRVScope::ScopeWorkgroup;

  int k_tile_count = ceil_div(shape<1>(B), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tBrB, tCrB);

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }
}

template <
    class ATensor,
    class BSGCTensor,
    class CSGCTensor,
    class TiledMMA>
/* gemm_TSS: C += A(gmem, M×K) * B(regs, N×K)^T  -- B is reused from a prior call. */
CUTE_DEVICE void gemm_TSS(
    ATensor const& A,        // (M,K)
    BSGCTensor const& tCrB,  // (N,K)
    CSGCTensor& tCrC,        // (M,N)
    int wg_m,                // m tile start id
    int wg_n,                // n tile start id
    TiledMMA const& mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());

  auto wg_tile = mma.tile_mnk();

  Tensor gA = local_tile(
      cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)

  auto copy_a = get_block_2d_copy_A<void>(mma, A);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);

  auto prefetch_a = make_block_2d_prefetch(copy_a);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);

  const int prefetch_dist = 3;

  constexpr auto barrier_scope = SPIRVScope::ScopeWorkgroup;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }
}

template <
    class ATensor,
    class BTensor,
    class SGCTensor,
    class TiledMMA>
/* gemm_TTS_k_multi: C += diag(K_multi[k]) * A(gmem) * B(gmem)^T
 * Before each DPAS call the corresponding k-slice of A is element-wise
 * scaled by K_multi[k_tile * tile_k + k * 16 + lane_id], which lets the
 * caller fold a per-token diagonal weight (e.g. exp(a)*beta) into the
 * matrix multiply without a separate pass. */
CUTE_DEVICE void gemm_TTS_k_multi(
    ATensor const& A,  // (M,K)
    BTensor const& B,  // (N,K)
    SGCTensor& tCrC,   // (M,N)
    int wg_m,          // m tile start id
    int wg_n,          // n tile start id
    TiledMMA const& mma,
    float* K_multi) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();
  auto sg = item.get_sub_group();
  int sg_local_id = sg.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());

  auto wg_tile = mma.tile_mnk();

  Tensor gA = local_tile(
      cA, select<0, 2>(wg_tile), make_coord(wg_m, _));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(
      cB, select<1, 2>(wg_tile), make_coord(wg_n, _));  // (BLK_N,BLK_K,k)

  auto copy_a = get_block_2d_copy_A<void>(mma, A);
  auto copy_b = get_block_2d_copy_B<void>(mma, B);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  const int prefetch_dist = 3;

  constexpr auto barrier_scope = SPIRVScope::ScopeWorkgroup;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (k_tile_prefetch < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
      prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    using TA = typename ATensor::element_type;
    Tensor A_frag = make_tensor<TA>(tCrA.layout());
    static constexpr auto I = decltype(size<0>(A_frag))::value;
    static constexpr auto J = decltype(size<1>(A_frag))::value;
    static constexpr auto K = decltype(size<2>(A_frag))::value;
    static constexpr int mma_K = 16;

    CUTE_UNROLL
    for (int k = 0; k < K; ++k) {
      float scale = K_multi[k_tile * get<2>(wg_tile) + k * mma_K + sg_local_id];
      CUTE_UNROLL
      for (int e = 0; e < I * J; ++e) {
        tCrA[k * I * J + e] =
            static_cast<TA>(static_cast<float>(tCrA[k * I * J + e]) * scale);
      }
    }

    cute::gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }
}

}  // namespace cutlass::gdn::detail
