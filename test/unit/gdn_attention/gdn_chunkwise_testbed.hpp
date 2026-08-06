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
 *  \brief Self-contained unit testbed for the Xe35 chunkwise GDN attention kernel.
 *
 *  Allocates a fixed-shape problem, launches `chunk_gated_delta_rule_launch`,
 *  and checks `core_attn_out` + `ssm_state` against the fast fp32 recurrent
 *  reference (run_recurrent_gdn_attn_reference) per-element at kTolE2E.
 *
 *  Intentionally does NOT depend on `GdnRunner` (CLI parsing, perf timers, and
 *  a profiling queue not appropriate for a gtest binary). */

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

#include "cutlass/util/device_memory.h"
#include "cutlass/util/initialize_block.hpp"

#include "gdn_attention/xe35_chunk_gated_delta_rule_launch.hpp"
// Chunkwise stage refs: used here only for apply_sigmoid_b.
#include "cutlass/util/reference/host/xe35_gdn_attention_stage_references.hpp"
// E2E oracle: recurrent reference + tolerance.
#include "cutlass/util/reference/host/xe35_gdn_attention_recurrent_reference.hpp"
#include "cutlass/util/reference/host/xe35_gdn_attention_compare.hpp"

namespace test::gdn_attention {

template <typename T, typename StateT>
struct ChunkwiseTestbed {
  // ----- fixed shape (caller-supplied) -----
  int batch       = 1;
  int num_v_heads = 64;
  int num_k_heads = 16;
  int head_k_dim  = 128;
  int head_v_dim  = 128;
  int seq_len     = 64;
  unsigned seed   = 0xC0FFEEu;

  // Per-element pass tolerance (like FMHA), default kTolE2E.
  float atol = cutlass::gdn::reference::recurrent::kTolE2E;
  float rtol = cutlass::gdn::reference::recurrent::kTolE2E;

  bool run() {
    constexpr int C = cutlass::gdn::kChunkSize;
    const int per_seq_padded   = ((seq_len + C - 1) / C) * C;
    const int total_seqlen     = batch * seq_len;
    const int total_virtual    = batch * per_seq_padded;

    sycl::queue queue{sycl::property::queue::in_order()};

    // -------- allocations --------
    cutlass::DeviceAllocation<T>       d_q {size_t(total_virtual) * num_k_heads * head_k_dim};
    cutlass::DeviceAllocation<T>       d_k {size_t(total_virtual) * num_k_heads * head_k_dim};
    cutlass::DeviceAllocation<T>       d_v {size_t(total_virtual) * num_v_heads * head_v_dim};
    cutlass::DeviceAllocation<float>   d_b {size_t(num_v_heads) * total_virtual};
    cutlass::DeviceAllocation<float>   d_a {size_t(num_v_heads) * total_virtual};
    cutlass::DeviceAllocation<float>   d_A_log {size_t(num_v_heads)};
    cutlass::DeviceAllocation<T>       d_dt_bias {size_t(num_v_heads)};
    cutlass::DeviceAllocation<int>     d_qsl {size_t(batch + 1)};
    cutlass::DeviceAllocation<int>     d_cache {size_t(batch)};
    cutlass::DeviceAllocation<uint8_t> d_has_init {size_t(batch)};
    cutlass::DeviceAllocation<T>       d_out {size_t(total_seqlen) * num_v_heads * head_v_dim};
    const size_t ssm_elems =
        size_t(batch) * num_v_heads * head_v_dim * head_k_dim;
    cutlass::DeviceAllocation<StateT>  d_ssm {ssm_elems};

    // workspaces (sizes obtained from the canonical helper so the testbed
    // tracks any future change to the kernel's intermediate layout)
    cutlass::gdn::GDNArguments ws_args{};
    ws_args.total_virtual_seqlen = total_virtual;
    ws_args.num_v_heads          = num_v_heads;
    ws_args.head_k_dim           = head_k_dim;
    ws_args.head_v_dim           = head_v_dim;
    const auto ws = cutlass::gdn::get_workspace_sizes(ws_args);
    cutlass::DeviceAllocation<T> d_A_ws {ws.A_elems};
    cutlass::DeviceAllocation<T> d_w_ws {ws.w_elems};
    cutlass::DeviceAllocation<T> d_u_ws {ws.u_elems};

    /* Zero-init workspaces. The kernel reads tiles after partial writes
     * across stages, so leftover USM contents can leak into the math.
     * Mirrors GdnRunner::initialize. */
    {
      std::vector<T> z(std::max({ws.A_elems, ws.w_elems, ws.u_elems}), T{0});
      d_A_ws.copy_from_host(z.data(), ws.A_elems);
      d_w_ws.copy_from_host(z.data(), ws.w_elems);
      d_u_ws.copy_from_host(z.data(), ws.u_elems);
    }

    // -------- initialize (mirrors GdnRunner::initialize, conservative ranges) --------
    cutlass::initialize_block(d_q,       static_cast<uint64_t>(seed) + 1);
    cutlass::initialize_block(d_k,       static_cast<uint64_t>(seed) + 2);
    cutlass::initialize_block(d_v,       static_cast<uint64_t>(seed) + 3);
    cutlass::initialize_block(d_b,       static_cast<uint64_t>(seed) + 4, -2.0f, 2.0f);
    cutlass::initialize_block(d_a,       static_cast<uint64_t>(seed) + 5, -2.0f, 2.0f);
    cutlass::initialize_block(d_A_log,   static_cast<uint64_t>(seed) + 6, -4.0f, 0.0f);
    cutlass::initialize_block(d_dt_bias, static_cast<uint64_t>(seed) + 7, T(-2.0f), T(2.0f));

    /* Apply sigmoid to b in place: the chunk launcher consumes `sigmoid(b)`
     * (see comment in xe35_gdn_attention_runner.hpp::initialize for why raw
     * b causes inverse-stage overflow). Keep both the raw and the sigmoided
     * host vectors: the device gets sigmoided b, and so does the host
     * reference -- `run_chunkwise_gdn_attn_reference`'s `b_sigmoid` parameter
     * is documented as already-sigmoided (caller-applied). */
    std::vector<float> h_b_raw(d_b.size());
    d_b.copy_to_host(h_b_raw.data(), h_b_raw.size());
    std::vector<float> h_b_sigmoid =
        cutlass::gdn::reference::stages::apply_sigmoid_b(h_b_raw);
    d_b.copy_from_host(h_b_sigmoid.data(), h_b_sigmoid.size());

    // ssm_state := 0, has_initial_state := 0 for every batch
    {
      std::vector<StateT> z(ssm_elems, StateT{0});
      d_ssm.copy_from_host(z.data(), z.size());
    }
    std::vector<int>     h_qsl(batch + 1, 0);
    std::vector<int>     h_cache(batch, 0);
    std::vector<uint8_t> h_has_init(batch, 0);
    for (int i = 0; i < batch; ++i) {
      h_qsl[i + 1] = h_qsl[i] + seq_len;
      h_cache[i]   = i;
    }
    d_qsl.copy_from_host(h_qsl.data(), h_qsl.size());
    d_cache.copy_from_host(h_cache.data(), h_cache.size());
    d_has_init.copy_from_host(h_has_init.data(), h_has_init.size());

    // snapshots of mutable inputs (kernel L2-normalizes q,k and cumsums a in place)
    std::vector<T>     h_q(d_q.size()), h_k(d_k.size()), h_v(d_v.size());
    std::vector<float> h_a(d_a.size());
    d_q.copy_to_host(h_q.data(), h_q.size());
    d_k.copy_to_host(h_k.data(), h_k.size());
    d_v.copy_to_host(h_v.data(), h_v.size());
    d_a.copy_to_host(h_a.data(), h_a.size());

    // dt_bias and A_log snapshots (read-only for kernel)
    std::vector<T>     h_dt_bias(d_dt_bias.size());
    std::vector<float> h_A_log(d_A_log.size());
    d_dt_bias.copy_to_host(h_dt_bias.data(), h_dt_bias.size());
    d_A_log.copy_to_host(h_A_log.data(), h_A_log.size());

    // -------- launch --------
    cutlass::gdn::GDNArguments args{};
    args.batch_size           = batch;
    args.total_seqlen         = total_seqlen;
    args.total_virtual_seqlen = total_virtual;
    args.num_k_heads          = num_k_heads;
    args.num_v_heads          = num_v_heads;
    args.head_k_dim           = head_k_dim;
    args.head_v_dim           = head_v_dim;
    args.ssm_state_stride_0   = num_v_heads * head_v_dim * head_k_dim;
    args.q                  = d_q.get();
    args.k                  = d_k.get();
    args.v                  = d_v.get();
    args.b                  = d_b.get();
    args.a                  = d_a.get();
    args.A_log              = d_A_log.get();
    args.dt_bias            = d_dt_bias.get();
    args.query_start_loc    = d_qsl.get();
    args.cache_indices      = d_cache.get();
    args.has_initial_state  = reinterpret_cast<bool const*>(d_has_init.get());
    args.core_attn_out      = d_out.get();
    args.ssm_state          = d_ssm.get();
    args.A_workspace        = d_A_ws.get();
    args.w_workspace        = d_w_ws.get();
    args.u_workspace        = d_u_ws.get();

    auto status = cutlass::gdn::chunk_gated_delta_rule_launch<T, StateT>(queue, args);
    if (status != cutlass::Status::kSuccess) return false;
    queue.wait_and_throw();

    // -------- Fetch device results once --------
    std::vector<T>      dev_out(d_out.size());
    std::vector<StateT> dev_ssm(ssm_elems);
    d_out.copy_to_host(dev_out.data(), dev_out.size());
    d_ssm.copy_to_host(dev_ssm.data(), dev_ssm.size());

    // -------- Reference 1: RECURRENT (token-by-token fp32) --------
    std::vector<T>      ref_recur_out(d_out.size(), T{0});
    std::vector<StateT> ref_recur_ssm(ssm_elems, StateT{0});
    // Snapshot h_q, h_k, h_a for second oracle (recurrent mutates q/k in place; h_a read-only)
    std::vector<T>     h_q_snap = h_q;
    std::vector<T>     h_k_snap = h_k;
    std::vector<float> h_a_snap = h_a;

    cutlass::gdn::reference::recurrent::run_recurrent_gdn_attn_reference<T, StateT>(
        ref_recur_out, ref_recur_ssm,
        h_q, h_k, h_v,
        h_b_sigmoid,   // already-sigmoided b
        h_a,
        h_A_log, h_dt_bias,
        h_qsl, h_cache, h_has_init,
        batch, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, total_virtual);

    auto stats_recur_out = cutlass::gdn::perf::compare_with_stats(ref_recur_out, dev_out, atol, rtol);
    auto stats_recur_ssm = cutlass::gdn::perf::compare_with_stats(ref_recur_ssm, dev_ssm, atol, rtol);
  /*
    To print result analysis histogram, uncomment the following:

    std::cout << "[RECURRENT oracle]\n";
    cutlass::gdn::perf::print_compare_stats("  core_attn_out", stats_recur_out);
    cutlass::gdn::perf::print_compare_stats("  ssm_state    ", stats_recur_ssm);
  */
    // -------- Reference 2: CHUNKWISE (5-stage, mirrors kernel decomposition) --------
    std::vector<T>      ref_chunk_out(d_out.size(), T{0});
    std::vector<StateT> ref_chunk_ssm(ssm_elems, StateT{0});
    // Restore q/k/a from snapshot (chunkwise mutates them all)
    h_q = h_q_snap;
    h_k = h_k_snap;
    h_a = h_a_snap;

    cutlass::gdn::reference::stages::run_chunkwise_gdn_attn_reference<T, StateT>(
        ref_chunk_out, ref_chunk_ssm,
        h_q, h_k, h_v,
        h_b_sigmoid,   // already-sigmoided b
        h_a,           // in/out (cumsummed)
        h_A_log, h_dt_bias,
        h_qsl, h_cache, h_has_init,
        batch, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, total_virtual,
        args.ssm_state_stride_0);

    auto stats_chunk_out = cutlass::gdn::perf::compare_with_stats(ref_chunk_out, dev_out, atol, rtol);
    auto stats_chunk_ssm = cutlass::gdn::perf::compare_with_stats(ref_chunk_ssm, dev_ssm, atol, rtol);
/*
    To print result analysis histogram, uncomment the following:

    std::cout << "[CHUNKWISE oracle]\n";
    cutlass::gdn::perf::print_compare_stats("  core_attn_out", stats_chunk_out);
    cutlass::gdn::perf::print_compare_stats("  ssm_state    ", stats_chunk_ssm);
*/
    // -------- Pass gate: BOTH oracles must pass (for now) --------
    bool recur_pass = stats_recur_out.passed() && stats_recur_ssm.passed();
    bool chunk_pass = stats_chunk_out.passed() && stats_chunk_ssm.passed();

    if (!recur_pass && chunk_pass) {
      std::cout << "*** DIAGNOSTIC: Recurrent oracle FAILED but chunkwise oracle PASSED.\n"
                << "    This indicates oracle divergence (not a kernel bug).\n";
    } else if (!recur_pass && !chunk_pass) {
      std::cout << "*** DIAGNOSTIC: BOTH oracles FAILED.\n"
                << "    This indicates a likely kernel bug.\n";
    }

    return recur_pass && chunk_pass;
  }
};

}  // namespace test::gdn_attention
