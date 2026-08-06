/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*!
  \file xe35_gdn_attention_stage_references.hpp
  \brief Templated host references for the 5 chunkwise GDN attention stages.

  Consumers
  ---------
  Shared by the two host-side verifiers of the GDN kernel:
    - examples/14_xe35_gdn_attention/xe35_gdn_attention_runner.hpp
    - test/unit/gdn_attention/gdn_chunkwise_testbed.hpp
  Both compose the 5 stage references in sequence (via
  run_chunkwise_gdn_attn_reference) and compare against the device output at
  atol = rtol = 5e-2.

  Two independent oracles (white-box chunkwise and black-box recurrent)
  -----------------------
  The kernel is validated against two host references that derive the same GDN
  math two different ways:
    - this file's 5-stage *chunkwise* reference, which mirrors the kernel's own
      per-stage decomposition (prepare / compute_A / inverse / compute_wu /
      fwd_o) one-to-one, and
    - a single-sweep per-token *recurrent* reference
      (xe35_gdn_attention_recurrent_reference.hpp).
  The two forms are algebraically equivalent but accumulate bf16/fp32 rounding
  in a different order, so requiring the kernel to agree with BOTH (each at the
  shared 5e-2 per-element tolerance) is a stronger check than either alone. The
  unit testbed runs both oracles; the example and benchmark --verify use the
  recurrent one.

  Algorithmic mapping
  -------------------
  These references mirror the device kernels one-to-one (prepare / compute_A /
  inverse / compute_wu / fwd_o), implementing the same chunkwise math.

  Functions are header-only static inline templates so multiple translation
  units can include them without ODR conflicts. This header depends only on the
  public GDN API (cutlass/gdn) -- never on the device-kernel internals -- so it
  is safe to host under cutlass/util/reference/host/.
*/

#pragma once

#include "cutlass/cutlass.h"

// Pull in only the lightweight public API for cutlass::gdn::kChunkSize. This
// host reference deliberately does NOT include the device-kernel header
// (xe35_chunk_gated_delta_rule_kernels.hpp) so it carries no dependency back into
// applications/ and no device code leaks into host translation units.
#include "gdn_attention/xe35_chunk_gated_delta_rule.hpp"  // for cutlass::gdn::kChunkSize

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cutlass::gdn::reference::stages {

inline constexpr int kChunkSize = cutlass::gdn::kChunkSize;

/* Elementwise sigmoid applied to the gate input `b`. The chunk kernel consumes
 * sigmoid(b) (a full GDN model produces in-range b via a conv1d/sigmoid
 * front-end the kernel does not include), and run_chunkwise_gdn_attn_reference
 * expects its `b_sigmoid` argument already sigmoided. Both the example runner
 * and the unit testbed apply this to the raw b before launch, so it lives here
 * as the single shared definition. */
inline std::vector<float> apply_sigmoid_b(const std::vector<float>& b_raw) {
  std::vector<float> out(b_raw.size());
  for (size_t i = 0; i < b_raw.size(); ++i)
    out[i] = 1.0f / (1.0f + std::exp(-b_raw[i]));
  return out;
}

/* ---------- Stage 1: prepare ------------------------------------------------
 * In-place L2-normalize q (scaled by 1/sqrt(head_k_dim)) and k, then write
 * per-chunk cumulative gate a[] = cumsum(softplus(a + dt_bias) * (-exp(A_log))). */
template <typename T>
inline void host_prepare_reference(
    std::vector<T>& q, std::vector<T>& k, std::vector<float>& a,
    const std::vector<float>& A_log, const std::vector<T>& dt_bias,
    const std::vector<int>& qsl,
    int batch, int num_k_heads, int num_v_heads,
    int head_k_dim, int total_virtual_seqlen)
{
  constexpr float eps = 1e-6f;
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(head_k_dim));
  for (int t = 0; t < total_virtual_seqlen; ++t) {
    for (int kh = 0; kh < num_k_heads; ++kh) {
      T* qp = q.data() + (static_cast<size_t>(t) * num_k_heads + kh) * head_k_dim;
      T* kp = k.data() + (static_cast<size_t>(t) * num_k_heads + kh) * head_k_dim;
      float qs = 0.0f, ks = 0.0f;
      for (int d = 0; d < head_k_dim; ++d) {
        float qv = static_cast<float>(qp[d]); float kv = static_cast<float>(kp[d]);
        qs += qv * qv; ks += kv * kv;
      }
      qs = std::sqrt(qs + eps); ks = std::sqrt(ks + eps);
      for (int d = 0; d < head_k_dim; ++d) {
        qp[d] = static_cast<T>(static_cast<float>(qp[d]) / qs * q_scale);
        kp[d] = static_cast<T>(static_cast<float>(kp[d]) / ks);
      }
    }
  }
  for (int vh = 0; vh < num_v_heads; ++vh) {
    const float A_log_exp_h = -std::exp(A_log[vh]);
    const float dt_bias_h   = static_cast<float>(dt_bias[vh]);
    int pre_chunks = 0;
    for (int b = 0; b < batch; ++b) {
      int seq_len_b = qsl[b + 1] - qsl[b];
      int n_chunks  = (seq_len_b + kChunkSize - 1) / kChunkSize;
      for (int c = 0; c < n_chunks; ++c) {
        int chunk_start = (pre_chunks + c) * kChunkSize;
        float buf[kChunkSize];
        for (int e = 0; e < kChunkSize; ++e) {
          float v = a[chunk_start + e + static_cast<size_t>(vh) * total_virtual_seqlen];
          v += dt_bias_h;
          float sp = (v < 20.0f) ? std::log1p(std::exp(v)) : v;
          buf[e] = sp * A_log_exp_h;
        }
        float run = 0.0f;
        for (int e = 0; e < kChunkSize; ++e) {
          run += buf[e];
          a[chunk_start + e + static_cast<size_t>(vh) * total_virtual_seqlen] = run;
        }
      }
      pre_chunks += n_chunks;
    }
  }
}

/* ---------- Stage 2: compute_A ---------------------------------------------
 * A[m,n] = 0 for m<n, 1 for m==n, else (K_m . K_n) * exp(a[m]-a[n]) * b[m] */
template <typename T>
inline void host_compute_A_reference(
    std::vector<T>& A,
    const std::vector<T>& k, const std::vector<float>& b, const std::vector<float>& a,
    const std::vector<int>& qsl,
    int batch, int num_k_heads, int num_v_heads,
    int head_k_dim, int total_virtual_seqlen)
{
  const int C = kChunkSize;
  const int kv_ratio = num_v_heads / num_k_heads;
  for (int vh = 0; vh < num_v_heads; ++vh) {
    const int kh = vh / kv_ratio;
    int pre_chunks = 0;
    for (int bi = 0; bi < batch; ++bi) {
      int seq_len_b = qsl[bi + 1] - qsl[bi];
      int n_chunks  = (seq_len_b + C - 1) / C;
      for (int c = 0; c < n_chunks; ++c) {
        int chunk_start = (pre_chunks + c) * C;
        const T*     K_chunk = k.data() + static_cast<size_t>(chunk_start) * num_k_heads * head_k_dim + kh * head_k_dim;
        const float* a_chunk = a.data() + static_cast<size_t>(vh) * total_virtual_seqlen + chunk_start;
        const float* b_chunk = b.data() + static_cast<size_t>(vh) * total_virtual_seqlen + chunk_start;
        T*           A_chunk = A.data() + (static_cast<size_t>(vh) * total_virtual_seqlen + chunk_start) * C;
        for (int m = 0; m < C; ++m) {
          for (int n = 0; n < C; ++n) {
            float out;
            if (m < n) out = 0.0f;
            else if (m == n) out = 1.0f;
            else {
              float s = 0.0f;
              const T* Km = K_chunk + static_cast<size_t>(m) * num_k_heads * head_k_dim;
              const T* Kn = K_chunk + static_cast<size_t>(n) * num_k_heads * head_k_dim;
              for (int d = 0; d < head_k_dim; ++d) s += static_cast<float>(Km[d]) * static_cast<float>(Kn[d]);
              out = s * std::exp(a_chunk[m] - a_chunk[n]) * b_chunk[m];
            }
            A_chunk[static_cast<size_t>(m) * C + n] = static_cast<T>(out);
          }
        }
      }
      pre_chunks += n_chunks;
    }
  }
}

/* ---------- Stage 3: inverse -----------------------------------------------
 * In-place strict-lower-triangular inverse of (I + L_strict), where L_strict
 * comes from compute_A. Diagonal/upper untouched. */
template <typename T>
inline void host_inverse_reference(
    std::vector<T>& A,
    const std::vector<int>& qsl,
    int batch, int num_v_heads, int total_virtual_seqlen)
{
  const int C = kChunkSize;
  std::vector<float> L(C * C), Inv(C * C);
  for (int vh = 0; vh < num_v_heads; ++vh) {
    int pre_chunks = 0;
    for (int bi = 0; bi < batch; ++bi) {
      int seq_len_b = qsl[bi + 1] - qsl[bi];
      int n_chunks  = (seq_len_b + C - 1) / C;
      for (int c = 0; c < n_chunks; ++c) {
        int chunk_start = (pre_chunks + c) * C;
        T* A_chunk = A.data() + (static_cast<size_t>(vh) * total_virtual_seqlen + chunk_start) * C;
        for (int i = 0; i < C; ++i)
          for (int j = 0; j < C; ++j)
            L[i * C + j] = static_cast<float>(A_chunk[i * C + j]);
        std::fill(Inv.begin(), Inv.end(), 0.0f);
        for (int i = 0; i < C; ++i) Inv[i * C + i] = 1.0f;
        for (int m = 1; m < C; ++m) {
          for (int n = 0; n < m; ++n) {
            float sum = L[m * C + n];
            for (int kk = n + 1; kk < m; ++kk) sum += Inv[kk * C + n] * L[m * C + kk];
            Inv[m * C + n] = -sum;
          }
        }
        for (int m = 1; m < C; ++m)
          for (int n = 0; n < m; ++n)
            A_chunk[m * C + n] = static_cast<T>(Inv[m * C + n]);
      }
      pre_chunks += n_chunks;
    }
  }
}

/* ---------- Stage 4: compute_wu --------------------------------------------
 * U[m,dv] = sum_n A[m,n] * V[n,dv] * b[n]
 * W[m,dk] = sum_n A[m,n] * K[n,dk] * exp(a[n]) * b[n]   (only if (c!=0)||init) */
template <typename T>
inline void host_compute_wu_reference(
    std::vector<T>& U, std::vector<T>& W,
    const std::vector<T>& A, const std::vector<T>& k, const std::vector<T>& v,
    const std::vector<float>& b, const std::vector<float>& a,
    const std::vector<int>& qsl,
    const std::vector<uint8_t>& initial_state,
    int batch, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int total_virtual_seqlen)
{
  const int C = kChunkSize;
  const int kv_ratio = num_v_heads / num_k_heads;
  for (int vh = 0; vh < num_v_heads; ++vh) {
    const int kh = vh / kv_ratio;
    int pre_chunks = 0;
    for (int bi = 0; bi < batch; ++bi) {
      int seq_len_b = qsl[bi + 1] - qsl[bi];
      int n_chunks  = (seq_len_b + C - 1) / C;
      const bool init = initial_state[bi] != 0;
      for (int c = 0; c < n_chunks; ++c) {
        int chunk_start = (pre_chunks + c) * C;
        const T*     A_chunk = A.data() + (static_cast<size_t>(vh) * total_virtual_seqlen + chunk_start) * C;
        const float* a_chunk = a.data() + static_cast<size_t>(vh) * total_virtual_seqlen + chunk_start;
        const float* b_chunk = b.data() + static_cast<size_t>(vh) * total_virtual_seqlen + chunk_start;
        T*           U_chunk = U.data() + (static_cast<size_t>(vh) * total_virtual_seqlen + chunk_start) * head_v_dim;
        T*           W_chunk = W.data() + (static_cast<size_t>(vh) * total_virtual_seqlen + chunk_start) * head_k_dim;
        for (int m = 0; m < C; ++m)
          for (int dv = 0; dv < head_v_dim; ++dv) {
            float acc = 0.0f;
            for (int n = 0; n < C; ++n) {
              float av = static_cast<float>(A_chunk[m * C + n]);
              if (av == 0.0f) continue;
              const T* Vn = v.data() + (static_cast<size_t>(chunk_start)+n)*num_v_heads*head_v_dim + vh*head_v_dim;
              acc += av * static_cast<float>(Vn[dv]) * b_chunk[n];
            }
            U_chunk[m * head_v_dim + dv] = static_cast<T>(acc);
          }
        if ((c != 0) || init) {
          for (int m = 0; m < C; ++m)
            for (int dk = 0; dk < head_k_dim; ++dk) {
              float acc = 0.0f;
              for (int n = 0; n < C; ++n) {
                float av = static_cast<float>(A_chunk[m * C + n]);
                if (av == 0.0f) continue;
                const T* Kn = k.data() + (static_cast<size_t>(chunk_start)+n)*num_k_heads*head_k_dim + kh*head_k_dim;
                float scale = std::exp(a_chunk[n]) * b_chunk[n];
                acc += av * static_cast<float>(Kn[dk]) * scale;
              }
              W_chunk[m * head_k_dim + dk] = static_cast<T>(acc);
            }
        }
      }
      pre_chunks += n_chunks;
    }
  }
}

/* ---------- Stage 5: fwd_o --------------------------------------------------
 * In-out U_mod (modified by Phase 1 when has_prev). Writes core_attn_out and
 * updates ssm_state in place. */
template <typename T, typename StateT>
inline void host_fwd_o_reference(
    std::vector<T>&      core_attn_out,
    std::vector<StateT>& ssm_state,
    std::vector<T>&      U_mod,
    const std::vector<T>&      W,
    const std::vector<T>&      q,
    const std::vector<T>&      k,
    const std::vector<float>&  a,
    const std::vector<int>&    qsl,
    const std::vector<int>&    cache_indices,
    const std::vector<uint8_t>& initial_state,
    int batch, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int total_virtual_seqlen,
    // Element stride between batch slots in ssm_state, matching the kernel ABI
    // (GDNArguments::ssm_state_stride_0 = ssm_state.stride(0) upstream). Passing
    // it explicitly -- rather than assuming the packed product below -- keeps
    // the reference correct for padded/aligned state layouts.
    int ssm_state_stride_0)
{
  const int C = kChunkSize;
  const int kv_ratio = num_v_heads / num_k_heads;
  const size_t state_stride0 = static_cast<size_t>(ssm_state_stride_0);

  std::vector<float> O2(C * C);

  for (int vh = 0; vh < num_v_heads; ++vh) {
    const int kh = vh / kv_ratio;
    int pre_chunks = 0;
    for (int bi = 0; bi < batch; ++bi) {
      int seq_len_b = qsl[bi + 1] - qsl[bi];
      int n_chunks  = (seq_len_b + C - 1) / C;
      const bool init = initial_state[bi] != 0;
      StateT* S_base = ssm_state.data()
                     + static_cast<size_t>(cache_indices[bi]) * state_stride0
                     + static_cast<size_t>(vh) * head_v_dim * head_k_dim;

      for (int c = 0; c < n_chunks; ++c) {
        int chunk_start    = (pre_chunks + c) * C;
        int out_chunk_off  = qsl[bi] + c * C;
        int ccs            = std::min(C, seq_len_b - c * C);
        const bool has_prev = (c != 0) || init;

        const float* a_chunk = a.data() + static_cast<size_t>(vh) * total_virtual_seqlen + chunk_start;
        float g_last     = a_chunk[ccs - 1];
        float g_last_exp = std::exp(g_last);

        T* U_chunk = U_mod.data() + (static_cast<size_t>(vh) * total_virtual_seqlen + chunk_start) * head_v_dim;
        const T* W_chunk = W.data() + (static_cast<size_t>(vh) * total_virtual_seqlen + chunk_start) * head_k_dim;

        std::vector<float> S_prev(head_v_dim * head_k_dim);
        for (size_t i = 0; i < S_prev.size(); ++i) S_prev[i] = static_cast<float>(S_base[i]);

        // Phase 1: U <- U - W @ S_prev^T
        if (has_prev) {
          for (int m = 0; m < ccs; ++m) {
            for (int dv = 0; dv < head_v_dim; ++dv) {
              float acc = 0.0f;
              for (int dk = 0; dk < head_k_dim; ++dk) {
                acc += static_cast<float>(W_chunk[m * head_k_dim + dk]) * S_prev[dv * head_k_dim + dk];
              }
              float u0 = static_cast<float>(U_chunk[m * head_v_dim + dv]);
              U_chunk[m * head_v_dim + dv] = static_cast<T>(u0 - acc);
            }
          }
        }

        // Phase 2: O2 (chunk x chunk), bf16 round-tripped to match kernel.
        std::fill(O2.begin(), O2.end(), 0.0f);
        for (int m = 0; m < ccs; ++m) {
          float gm = a_chunk[m];
          const T* Qm = q.data() + (static_cast<size_t>(chunk_start) + m) * num_k_heads * head_k_dim + kh * head_k_dim;
          for (int n = 0; n <= m; ++n) {
            float gn = a_chunk[n];
            const T* Kn = k.data() + (static_cast<size_t>(chunk_start) + n) * num_k_heads * head_k_dim + kh * head_k_dim;
            float s = 0.0f;
            for (int d = 0; d < head_k_dim; ++d) s += static_cast<float>(Qm[d]) * static_cast<float>(Kn[d]);
            O2[m * C + n] = static_cast<float>(static_cast<T>(s * std::exp(gm - gn)));
          }
        }

        // Phase 3 / 4b: write O
        T* O_chunk = core_attn_out.data() + static_cast<size_t>(out_chunk_off) * num_v_heads * head_v_dim + vh * head_v_dim;
        if (has_prev) {
          for (int m = 0; m < ccs; ++m) {
            float gm_exp = std::exp(a_chunk[m]);
            const T* Qm = q.data() + (static_cast<size_t>(chunk_start) + m) * num_k_heads * head_k_dim + kh * head_k_dim;
            for (int dv = 0; dv < head_v_dim; ++dv) {
              float acc = 0.0f;
              for (int d = 0; d < head_k_dim; ++d) acc += static_cast<float>(Qm[d]) * S_prev[dv * head_k_dim + d];
              acc *= gm_exp;
              for (int n = 0; n <= m; ++n)
                acc += O2[m * C + n] * static_cast<float>(U_chunk[n * head_v_dim + dv]);
              O_chunk[m * num_v_heads * head_v_dim + dv] = static_cast<T>(acc);
            }
          }
        } else {
          for (int m = 0; m < ccs; ++m)
            for (int dv = 0; dv < head_v_dim; ++dv) {
              float acc = 0.0f;
              for (int n = 0; n <= m; ++n)
                acc += O2[m * C + n] * static_cast<float>(U_chunk[n * head_v_dim + dv]);
              O_chunk[m * num_v_heads * head_v_dim + dv] = static_cast<T>(acc);
            }
        }

        // Phase 4: update S_base
        std::vector<float> S_new(head_v_dim * head_k_dim, 0.0f);
        if (has_prev) {
          for (size_t i = 0; i < S_new.size(); ++i) S_new[i] = g_last_exp * S_prev[i];
        }
        for (int n = 0; n < ccs; ++n) {
          float gmulti = std::exp(g_last - a_chunk[n]);
          const T* Kn = k.data() + (static_cast<size_t>(chunk_start) + n) * num_k_heads * head_k_dim + kh * head_k_dim;
          for (int dv = 0; dv < head_v_dim; ++dv) {
            float Un = static_cast<float>(U_chunk[n * head_v_dim + dv]);
            float u_scaled = Un * gmulti;
            if (u_scaled == 0.0f) continue;
            for (int dk = 0; dk < head_k_dim; ++dk) {
              S_new[dv * head_k_dim + dk] += u_scaled * static_cast<float>(Kn[dk]);
            }
          }
        }
        for (size_t i = 0; i < S_new.size(); ++i) S_base[i] = static_cast<StateT>(S_new[i]);
      }
      pre_chunks += n_chunks;
    }
  }
}

/* ---------- Compositional driver -------------------------------------------
 * Runs all 5 chunkwise stages in sequence, mirroring the kernel pipeline
 * (kernel_launcher in xe35_chunk_gated_delta_rule_kernels.hpp). Inputs match
 * the kernel ABI; outputs core_attn_out and ssm_state are written in place.
 * 
 * Required preconditions (mirror the kernel):
 *   - q, k, v are filled (q,k will be L2-normalized in place by stage 1).
 *   - b is already sigmoided (caller is responsible; the chunk kernel
 *     consumes sigmoid(b_raw) as produced by chunk_causal_conv1d_xe2).
 *   - a is the raw pre-cumsum gate input; stage 1 cumsum's it in place.
 *   - ssm_state holds the pre-kernel state (zeros for has_initial_state=false). */
template <typename T, typename StateT>
inline void run_chunkwise_gdn_attn_reference(
    std::vector<T>&     core_attn_out,
    std::vector<StateT>& ssm_state,
    std::vector<T>&     q,   // in/out (normalized)
    std::vector<T>&     k,   // in/out (normalized)
    const std::vector<T>&     v,
    const std::vector<float>& b_sigmoid,
    std::vector<float>&       a,   // in/out (cumsummed)
    const std::vector<float>& A_log,
    const std::vector<T>&     dt_bias,
    const std::vector<int>&   qsl,
    const std::vector<int>&   cache_indices,
    const std::vector<uint8_t>& initial_state,
    int batch, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int total_virtual_seqlen,
    // Forwarded to host_fwd_o_reference; see GDNArguments::ssm_state_stride_0.
    int ssm_state_stride_0)
{
  // Stage 1
  host_prepare_reference<T>(q, k, a, A_log, dt_bias, qsl,
                            batch, num_k_heads, num_v_heads,
                            head_k_dim, total_virtual_seqlen);

  // Stage 2: A workspace -- [num_v_heads, total_virtual_seqlen, kChunkSize]
  const size_t A_elems = static_cast<size_t>(num_v_heads) * total_virtual_seqlen * kChunkSize;
  std::vector<T> A_ws(A_elems, T{0});
  host_compute_A_reference<T>(A_ws, k, b_sigmoid, a, qsl,
                              batch, num_k_heads, num_v_heads,
                              head_k_dim, total_virtual_seqlen);

  // Stage 3
  host_inverse_reference<T>(A_ws, qsl, batch, num_v_heads, total_virtual_seqlen);

  // Stage 4: W [..., head_k_dim], U [..., head_v_dim]
  const size_t W_elems = static_cast<size_t>(num_v_heads) * total_virtual_seqlen * head_k_dim;
  const size_t U_elems = static_cast<size_t>(num_v_heads) * total_virtual_seqlen * head_v_dim;
  std::vector<T> W_ws(W_elems, T{0});
  std::vector<T> U_ws(U_elems, T{0});
  host_compute_wu_reference<T>(U_ws, W_ws, A_ws, k, v, b_sigmoid, a, qsl,
                               initial_state,
                               batch, num_k_heads, num_v_heads,
                               head_k_dim, head_v_dim, total_virtual_seqlen);

  // Stage 5
  host_fwd_o_reference<T, StateT>(core_attn_out, ssm_state, U_ws, W_ws,
                                  q, k, a, qsl, cache_indices, initial_state,
                                  batch, num_k_heads, num_v_heads,
                                  head_k_dim, head_v_dim, total_virtual_seqlen,
                                  ssm_state_stride_0);
}

}  // namespace cutlass::gdn::reference::stages
