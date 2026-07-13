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
 **************************************************************************************************/
#pragma once

/*!
  \file xe35_chunk_gated_delta_rule.hpp
  \brief Public C++ API for the Xe35 chunkwise Gated DeltaNet (GDN) attention kernel.

  This is the lightweight API surface -- it pulls in no device code, so host-only
  consumers (e.g. the chunkwise host reference) can depend on it freely. It
  declares:
    - GDNArguments : a flat struct of all tensor shapes and device pointers
    - get_workspace_sizes() : computes the three intermediate buffer sizes
    - chunk_gated_delta_rule_launch<T,StateT>() : submits all five stages
      to an in-order SYCL queue without any host-side wait between stages.

  This header only *declares* chunk_gated_delta_rule_launch() -- it does not
  define it. Callers that actually launch the kernel must instead include
  xe35_chunk_gated_delta_rule_launch.hpp, which holds the launcher's definition
  (an inline function template instantiated at each call site) and, in turn,
  pulls in the device-kernel header. That is why this public header stays free
  of device code.

  There is no framework dependency -- all tensors are raw device pointers
*/

#include "cutlass/cutlass.h"
#include <sycl/sycl.hpp>

namespace cutlass::gdn {

/// Per-launch problem description.
struct GDNArguments {
  // ---- shape ----
  int batch_size;            ///< number of sequences (== query_start_loc.size - 1)
  int total_seqlen;          ///< total real tokens (unpadded)
  int total_virtual_seqlen;  ///< total tokens after padding each sequence to a chunk multiple
  int num_k_heads;
  int num_v_heads;
  int head_k_dim;
  int head_v_dim;
  int ssm_state_stride_0;    ///< stride between batch slots in ssm_state

  /* ---- inputs (device pointers) ----
   * NOTE: q, k, a are mutated IN PLACE by the prepare stage (L2-normalize q/k;
   * cumsum a). They must not be truly const on the caller side. */
  void* q;                   ///< [total_virtual_seqlen, num_k_heads, head_k_dim]   T  (in/out)
  void* k;                   ///< [total_virtual_seqlen, num_k_heads, head_k_dim]   T  (in/out)
  const void* v;             ///< [total_virtual_seqlen, num_v_heads, head_v_dim]   T
  const float* b;            ///< [num_v_heads, total_virtual_seqlen]               FP32
  float* a;                  ///< [num_v_heads, total_virtual_seqlen]               FP32 (in/out)
  const float* A_log;        ///< [num_v_heads]                                     FP32
  const void* dt_bias;       ///< [num_v_heads]                                     T
  const int* query_start_loc;///< [batch_size + 1] packed seq boundaries            int32
  const int* cache_indices;  ///< [batch_size] ssm_state slot per batch             int32
  /* has_initial_state[batch_id] selects whether the SSM-state load at the
   * start of stages 4-5 reads ssm_state[cache_indices[batch_id], ...] or
   * starts from zero. A nullptr pointer is treated
   * as "every batch has carry-over state" -- the caller is then responsible
   * for pre-zeroing the corresponding ssm_state slots before the launch
   * (this matches the Python-side `initial_state[~has_initial_state] = 0`
   * idiom in vllm/model_executor/layers/mamba/gdn). */
  const bool* has_initial_state;///< [batch_size] or nullptr                        bool

  // ---- outputs (device pointers) ----
  void* core_attn_out;       ///< [total_seqlen, num_v_heads, head_v_dim]           T
  /* Recurrent SSM state, loaded (gated by has_initial_state) and written back
   * per batch. The leading cache_batch extent is the caller's slot count (not
   * a GDNArguments field): batch_id picks slot cache_indices[batch_id], and
   * ssm_state_stride_0 is the element stride between slots. */
  void* ssm_state;           ///< [cache_batch, num_v_heads, head_v_dim, head_k_dim] StateT

  /* ---- workspace (device pointers, allocated by caller) ----
   * Size each buffer with get_workspace_sizes() -- do not hand-roll the math.
   * The middle extent is total_virtual_seqlen (the chunk-padded token count),
   * NOT total_seqlen, because that is the stride the kernels index them by. */
  void* A_workspace;         ///< [num_v_heads, total_virtual_seqlen, kChunkSize]   T
  void* w_workspace;         ///< [num_v_heads, total_virtual_seqlen, head_k_dim]   T
  void* u_workspace;         ///< [num_v_heads, total_virtual_seqlen, head_v_dim]   T
};

/// Chunk size baked into the Xe35 kernel. Single source of truth: the device
/// kernels alias this as cutlass::gdn::detail::chunk_size in
/// xe35_chunk_gated_delta_rule_kernels.hpp.
///
// It is a compile-time constant rather than a parameter
/// because every fixed-trip inner loop in the kernels (and the 4x4 grid of
/// 16x16 DPAS blocks used to invert the per-chunk transition matrix) is sized
/// against it.
constexpr int kChunkSize = 64;

/* Compute the per-tensor workspace sizes (in elements of T) required for a
 * given problem. Caller allocates A_workspace, w_workspace, u_workspace. */
struct GDNWorkspaceSizes {
  size_t A_elems;
  size_t w_elems;
  size_t u_elems;
};
inline GDNWorkspaceSizes get_workspace_sizes(GDNArguments const& args) {
  /* Use total_virtual_seqlen directly -- this is the exact stride the kernel
   * uses when indexing A/w/u (e.g., v_head_id * total_virtual_seqlen * chunk_size). */
  const size_t tvs = static_cast<size_t>(args.total_virtual_seqlen);
  GDNWorkspaceSizes out;
  out.A_elems = static_cast<size_t>(args.num_v_heads) * tvs * kChunkSize;
  out.w_elems = static_cast<size_t>(args.num_v_heads) * tvs * args.head_k_dim;
  out.u_elems = static_cast<size_t>(args.num_v_heads) * tvs * args.head_v_dim;
  return out;
}

/* Launch the 5-stage Xe35 chunkwise GDN kernel on an in-order queue.
 *   T       : activation dtype (bfloat16_t)
 *   StateT  : ssm_state dtype (always float in current builds)
 *
 * `queue` MUST be in-order: the five stages are submitted back-to-back with no
 * host wait, so an out-of-order queue would race. The call is asynchronous --
 * the caller waits on the queue. Returns:
 *   kSuccess              -- submitted (or a no-op for an empty problem:
 *                            batch_size <= 0 or total_virtual_seqlen <= 0).
 *   kErrorInvalidProblem  -- a head count / head dim is <= 0, or
 *                            num_v_heads % num_k_heads != 0 (GQA grouping).
 * Device pointers are NOT validated; see the per-pointer contract in
 * GDNArguments. To actually launch, include
 * xe35_chunk_gated_delta_rule_launch.hpp (this header only declares the entry). */
template <typename T, typename StateT>
cutlass::Status chunk_gated_delta_rule_launch(
    sycl::queue& queue,
    GDNArguments const& args);

} // namespace cutlass::gdn
