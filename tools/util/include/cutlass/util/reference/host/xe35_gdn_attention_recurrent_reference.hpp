/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*!
  \file xe35_gdn_attention_recurrent_reference.hpp
  \brief Single-shot per-token recurrent host reference -- the
         E2E oracle for the Xe35 GDN kernel.

  Implements the GDN SSM recurrence (ref_gdn_attention_spec). The Python conv1d
  front-end, qkvz extraction, and spec-decode scatter are out of scope here --
  the harness supplies q/k/v already extracted and b pre-sigmoided.

  Token-at-a-time eval (one O(seq * H_v * D_k * D_v) sweep, no C x C
  inverse), so E2E verify stays within the CI/simulator budget. Used by the unit
  test, example, and benchmark --verify. The per-stage chunkwise references
  (xe35_gdn_attention_stage_references.hpp) stay in place as the per-kernel
  oracles; this is the aggregate one.

  E2E checks the same per-element bound the rest of the repo uses (FMHA's
  BlockCompareRelativelyEqual at 5e-2): empirically the bf16 kernel matches this
  fp32 reference to ~0.05 per element.

  Recurrence (per v-head, per token t in order; S is [head_v_dim, head_k_dim]):
    alpha = exp(-exp(A_log) * softplus(a_raw[t] + dt_bias))   (gated decay, <=1)
    beta  = b_sigmoid[t]                                      (delta-rule strength)
    S    <- alpha * S
    pred  = S @ k_t                                           (kv_mem in vLLM)
    S    <- S + beta * outer(v_t - pred, k_t)
    o_t   = S @ q_t                                           (q pre-scaled)
  GQA: v-head vh reads k-head vh/(H_v/H_k), i.e. vLLM's repeat_interleave.

  Input contract (drop-in with run_chunkwise_gdn_attn_reference):
    q, k      : RAW; L2-normalized in place here (q also *= 1/sqrt(D_k)).
    a         : RAW per-token gate, read-only (NOT cumsummed).
    b_sigmoid : already sigmoided (caller applies apply_sigmoid_b).
    ssm_state : pre-kernel state (zeros when has_initial_state is false).
    ssm_state_stride_0 : per-batch stride between cache slots in ssm_state
                (GDNArguments::ssm_state_stride_0). 0 means the contiguous
                default num_v_heads * head_v_dim * head_k_dim.
  Header-only; depends only on the public cutlass/gdn API, no device code.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "gdn_attention/xe35_chunk_gated_delta_rule.hpp"  // cutlass::gdn::kChunkSize

#include <algorithm>  // std::fill
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cutlass::gdn::reference::recurrent {

inline constexpr int kChunkSize = cutlass::gdn::kChunkSize;

/* E2E per-element tolerance (atol == rtol), shared by all three verify paths.
 * Matches FMHA's bf16 tolerance; the kernel meets it vs. this fp32 ref. */
inline constexpr float kTolE2E = 5e-2f;

// softplus with the same x>=20 cutoff the prepare stage uses (overflow guard).
inline float softplus_ref(float x) {
  return (x < 20.0f) ? std::log1p(std::exp(x)) : x;
}

/* Single-shot recurrent GDN forward. Writes core_attn_out + ssm_state, and
 * normalizes q/k in place (see header for the full contract). T = activation
 * dtype (bfloat16_t), StateT = ssm_state dtype (float). */
template <typename T, typename StateT>
inline void run_recurrent_gdn_attn_reference(
    std::vector<T>&             core_attn_out,
    std::vector<StateT>&        ssm_state,
    std::vector<T>&             q,           // in/out (normalized)
    std::vector<T>&             k,           // in/out (normalized)
    const std::vector<T>&       v,
    const std::vector<float>&   b_sigmoid,
    const std::vector<float>&   a,           // raw per-token gate input (read-only)
    const std::vector<float>&   A_log,
    const std::vector<T>&       dt_bias,
    const std::vector<int>&     qsl,         // query_start_loc, length batch+1
    const std::vector<int>&     cache_indices,
    const std::vector<uint8_t>& initial_state,
    int batch, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int total_virtual_seqlen,
    // Per-batch stride between cache slots in ssm_state; see
    // GDNArguments::ssm_state_stride_0. 0 -> contiguous default below.
    int ssm_state_stride_0 = 0)
{
  const int C        = kChunkSize;
  const int kv_ratio = num_v_heads / num_k_heads;
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));
  constexpr float eps = 1e-6f;
  const size_t state_stride0 =
      ssm_state_stride_0 > 0
          ? static_cast<size_t>(ssm_state_stride_0)
          : static_cast<size_t>(num_v_heads) * head_v_dim * head_k_dim;

  // L2-normalize q (then *q_scale) and k in place, over the full padded extent
  // to match the kernel's prepare stage (vLLM: rsqrt(.+eps) then *scale).
  for (int t = 0; t < total_virtual_seqlen; ++t) {
    for (int kh = 0; kh < num_k_heads; ++kh) {
      T* qp = q.data() + (static_cast<size_t>(t) * num_k_heads + kh) * head_k_dim;
      T* kp = k.data() + (static_cast<size_t>(t) * num_k_heads + kh) * head_k_dim;
      float qs = 0.0f, ks = 0.0f;
      for (int d = 0; d < head_k_dim; ++d) {
        float qv = static_cast<float>(qp[d]), kv = static_cast<float>(kp[d]);
        qs += qv * qv; ks += kv * kv;
      }
      qs = std::sqrt(qs + eps); ks = std::sqrt(ks + eps);
      for (int d = 0; d < head_k_dim; ++d) {
        qp[d] = static_cast<T>(static_cast<float>(qp[d]) / qs * q_scale);
        kp[d] = static_cast<T>(static_cast<float>(kp[d]) / ks);
      }
    }
  }

  // Per-(v-head, batch) recurrence over real tokens.
  std::vector<float> S(static_cast<size_t>(head_v_dim) * head_k_dim);
  std::vector<float> pred(head_v_dim);

  for (int vh = 0; vh < num_v_heads; ++vh) {
    const int kh = vh / kv_ratio;
    const float A_log_exp_h = -std::exp(A_log[vh]);
    const float dt_bias_h   = static_cast<float>(dt_bias[vh]);

    int pre_chunks = 0;  // virtual-token base advances by padded chunks per batch
    for (int bi = 0; bi < batch; ++bi) {
      const int seq_len_b = qsl[bi + 1] - qsl[bi];
      const int n_chunks  = (seq_len_b + C - 1) / C;
      const int virtual_base = pre_chunks * C;     // first virtual row of this sequence
      const bool init = initial_state[bi] != 0;

      StateT* S_base = ssm_state.data()
                     + static_cast<size_t>(cache_indices[bi]) * state_stride0
                     + static_cast<size_t>(vh) * head_v_dim * head_k_dim;

      // Initial state: carry-over or zero.
      if (init) {
        for (size_t i = 0; i < S.size(); ++i) S[i] = static_cast<float>(S_base[i]);
      } else {
        std::fill(S.begin(), S.end(), 0.0f);
      }

      for (int t = 0; t < seq_len_b; ++t) {
        const int vt = virtual_base + t;  // virtual-token row

        const float a_raw = a[static_cast<size_t>(vh) * total_virtual_seqlen + vt];
        const float g_t   = softplus_ref(a_raw + dt_bias_h) * A_log_exp_h;
        const float alpha = std::exp(g_t);
        const float beta  = b_sigmoid[static_cast<size_t>(vh) * total_virtual_seqlen + vt];

        const T* q_t = q.data() + static_cast<size_t>(vt) * num_k_heads * head_k_dim + kh * head_k_dim;
        const T* k_t = k.data() + static_cast<size_t>(vt) * num_k_heads * head_k_dim + kh * head_k_dim;
        const T* v_t = v.data() + static_cast<size_t>(vt) * num_v_heads * head_v_dim + vh * head_v_dim;

        // S <- alpha * S  (gated decay)
        for (size_t i = 0; i < S.size(); ++i) S[i] *= alpha;

        // pred = S @ k_t
        for (int dv = 0; dv < head_v_dim; ++dv) {
          float acc = 0.0f;
          const float* S_row = S.data() + static_cast<size_t>(dv) * head_k_dim;
          for (int dk = 0; dk < head_k_dim; ++dk)
            acc += S_row[dk] * static_cast<float>(k_t[dk]);
          pred[dv] = acc;
        }

        // S <- S + beta * outer(v_t - pred, k_t)
        for (int dv = 0; dv < head_v_dim; ++dv) {
          const float w = beta * (static_cast<float>(v_t[dv]) - pred[dv]);
          if (w == 0.0f) continue;
          float* S_row = S.data() + static_cast<size_t>(dv) * head_k_dim;
          for (int dk = 0; dk < head_k_dim; ++dk)
            S_row[dk] += w * static_cast<float>(k_t[dk]);
        }

        // o_t = S @ q_t  (post-update state)
        T* o_t = core_attn_out.data()
               + static_cast<size_t>(qsl[bi] + t) * num_v_heads * head_v_dim
               + static_cast<size_t>(vh) * head_v_dim;
        for (int dv = 0; dv < head_v_dim; ++dv) {
          float acc = 0.0f;
          const float* S_row = S.data() + static_cast<size_t>(dv) * head_k_dim;
          for (int dk = 0; dk < head_k_dim; ++dk)
            acc += S_row[dk] * static_cast<float>(q_t[dk]);
          o_t[dv] = static_cast<T>(acc);
        }
      }

      // Write updated state back to its cache slot.
      for (size_t i = 0; i < S.size(); ++i) S_base[i] = static_cast<StateT>(S[i]);

      pre_chunks += n_chunks;
    }
  }
}

}  // namespace cutlass::gdn::reference::recurrent
