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

/*!
  \file benchmarks/gdn/benchmark_runner.hpp
  \brief Google Benchmark harness for the Xe35 chunkwise Gated DeltaNet (GDN)
         attention kernel.

  Mirrors the structure of benchmarks/flash_attention/benchmark_runner.hpp:
    - GDNBenchmarkOptions  : CLI options (parsed from the config .in file)
    - BenchmarkRunnerGDN<T,StateT> : owns device allocations, runs the timed loop
    - CUTLASS_CREATE_GDN_BENCHMARK / CUTLASS_GDN_BENCHMARK : registration macros

  Performance metrics emitted:
    avg_tflops      : rough arithmetic throughput estimate (GFLOPs / ms = TFLOPS/s)
    avg_throughput  : memory bandwidth estimate (MB/ms = GB/s)
    avg_runtime_ms  : trimmed mean latency (best + worst iterations removed)
    best_runtime_ms : minimum latency over all iterations
    worst_runtime_ms: maximum latency over all iterations
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/initialize_block.hpp"

#include "../common.hpp"

#include "gdn_attention/xe35_chunk_gated_delta_rule_launch.hpp"
// Chunkwise stage refs: used here only for apply_sigmoid_b.
#include "cutlass/util/reference/host/xe35_gdn_attention_stage_references.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <benchmark/benchmark.h>

namespace cutlass::benchmark::gdn {

// ---------------------------------------------------------------------------
// GDNBenchmarkOptions
//
// Parsed from one line of the benchmark configuration .in file, e.g.:
//   GdnConfig_BF16_FP32 --bm_name=gdn_bf16 --batch=1 --num_v_heads=64
//                       --num_k_heads=16 --head_k_dim=128 --head_v_dim=128
//                       --seq_len=4096
// ---------------------------------------------------------------------------

struct GDNBenchmarkOptions {
  bool error = false;

  int batch       = 1;
  int num_v_heads = 8;
  int num_k_heads = 1;    // GQA: num_v_heads must be a multiple of num_k_heads
  int head_k_dim  = 128;
  int head_v_dim  = 128;
  int seq_len     = 1024;

  std::string bm_name = "GDN";

  void parse(int argc, const char** args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("batch",       batch,       batch);
    cmd.get_cmd_line_argument("num_v_heads", num_v_heads, num_v_heads);
    cmd.get_cmd_line_argument("num_k_heads", num_k_heads, num_k_heads);
    cmd.get_cmd_line_argument("head_k_dim",  head_k_dim,  head_k_dim);
    cmd.get_cmd_line_argument("head_v_dim",  head_v_dim,  head_v_dim);
    cmd.get_cmd_line_argument("seq_len",     seq_len,     seq_len);
    cmd.get_cmd_line_argument("bm_name",     bm_name,     bm_name);

    // Reject non-positive shape parameters BEFORE the modulo below so a
    // user passing --num_k_heads=0 cannot trigger a divide-by-zero. Matches
    // the validation order in the example runner and launch wrapper.
    if (batch <= 0 || num_k_heads <= 0 || num_v_heads <= 0 ||
        head_k_dim <= 0 || head_v_dim <= 0 || seq_len <= 0) {
      std::cerr << "[GDN benchmark] Error: shape parameters must be positive\n";
      error = true;
    }
    if (!error && num_v_heads % num_k_heads != 0) {
      std::cerr << "[GDN benchmark] Error: num_v_heads (" << num_v_heads
                << ") must be a multiple of num_k_heads (" << num_k_heads << ")\n";
      error = true;
    }
    // The kernels tile the head dims into kChunkSize-wide 2D blocks
    // (`for (dv = 0; dv < head_v_dim / chunk_size; ++dv)` etc.), so a head dim
    // that is not a multiple of kChunkSize would silently drop its remainder.
    // Reject it here rather than produce wrong results. Matches the validation
    // in examples/14_xe35_gdn_attention's runner.
    constexpr int C = cutlass::gdn::kChunkSize;
    if (!error && (head_k_dim % C != 0 || head_v_dim % C != 0)) {
      std::cerr << "[GDN benchmark] Error: head_k_dim (" << head_k_dim
                << ") and head_v_dim (" << head_v_dim
                << ") must each be a multiple of the chunk size (" << C << ")\n";
      error = true;
    }
  }

  std::string benchmark_name() const {
    std::ostringstream ss;
    ss << bm_name << "/"
       << batch       << "x"
       << num_v_heads << "x"
       << num_k_heads << "x"
       << seq_len     << "x"
       << head_k_dim  << "x"
       << head_v_dim;
    return ss.str();
  }
};

// ---------------------------------------------------------------------------
// BenchmarkRunnerGDN<T, StateT>
//
// Owns the device allocations and workspace, and provides a run() method
// matching the BenchmarkRegistry callback signature:
//   void run(::benchmark::State&, GDNBenchmarkOptions const&,
//            cutlass::KernelHardwareInfo const&)
//
// Template parameters:
//   T      - activation dtype (bfloat16_t)
//   StateT - SSM state dtype  (always float in current builds)
// ---------------------------------------------------------------------------

template <typename T, typename StateT>
struct BenchmarkRunnerGDN {

  // ---- device allocations ----
  cutlass::DeviceAllocation<T>       d_q, d_k, d_v;
  cutlass::DeviceAllocation<T>       d_dt_bias;
  cutlass::DeviceAllocation<float>   d_b, d_a, d_A_log;
  cutlass::DeviceAllocation<int>     d_query_start_loc, d_cache_indices;
  cutlass::DeviceAllocation<uint8_t> d_has_initial_state; // bool stored as uint8_t
  cutlass::DeviceAllocation<T>       d_core_attn_out;
  cutlass::DeviceAllocation<StateT>  d_ssm_state;
  cutlass::DeviceAllocation<T>       d_A_ws, d_w_ws, d_u_ws;

  // ---- derived shape ----
  int batch                = 0;
  int num_v_heads          = 0;
  int num_k_heads          = 0;
  int head_k_dim           = 0;
  int head_v_dim           = 0;
  int total_seqlen         = 0;
  int total_virtual_seqlen = 0;  // padded per sequence to chunk multiples

  // ---- in-order queue (required by the GDN pipeline) ----
  // Profiling build: GPU_Clock sums device-side kernel timestamps (the launcher
  // registers the 5 stage events with EventManager), and get_profiling_info()
  // requires enable_profiling() on the queue. The default build times with
  // GPU_Clock's host wall-clock fallback and needs no extra property.
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  sycl::queue queue{sycl::property_list{
      sycl::property::queue::in_order(),
      sycl::property::queue::enable_profiling()}};
#else
  sycl::queue queue{sycl::property::queue::in_order()};
#endif

  // ---- setup / teardown ----

  void initialize(const GDNBenchmarkOptions& opts) {
    batch       = opts.batch;
    num_v_heads = opts.num_v_heads;
    num_k_heads = opts.num_k_heads;
    head_k_dim  = opts.head_k_dim;
    head_v_dim  = opts.head_v_dim;

    total_seqlen = batch * opts.seq_len;

    constexpr int C    = cutlass::gdn::kChunkSize;
    int per_seq_padded = ((opts.seq_len + C - 1) / C) * C;
    total_virtual_seqlen = batch * per_seq_padded;

    auto alloc = [](auto& da, size_t n, const char* tag) {
      try {
        da.reset(n);
      } catch (std::exception const& e) {
        std::cerr << "[GDN benchmark] alloc FAILED for " << tag
                  << " (n=" << n << "): " << e.what() << "\n";
        throw;
      }
    };

    alloc(d_q,   size_t(total_virtual_seqlen) * num_k_heads * head_k_dim, "d_q");
    alloc(d_k,   size_t(total_virtual_seqlen) * num_k_heads * head_k_dim, "d_k");
    alloc(d_v,   size_t(total_virtual_seqlen) * num_v_heads * head_v_dim, "d_v");
    alloc(d_b,   size_t(num_v_heads) * total_virtual_seqlen, "d_b");
    alloc(d_a,   size_t(num_v_heads) * total_virtual_seqlen, "d_a");
    alloc(d_A_log,   num_v_heads, "d_A_log");
    alloc(d_dt_bias, num_v_heads, "d_dt_bias");
    alloc(d_query_start_loc, batch + 1, "d_query_start_loc");
    alloc(d_cache_indices,   batch,     "d_cache_indices");
    alloc(d_has_initial_state, batch,   "d_has_initial_state");
    alloc(d_core_attn_out, size_t(total_seqlen) * num_v_heads * head_v_dim, "d_core_attn_out");
    alloc(d_ssm_state,
          size_t(batch) * num_v_heads * head_v_dim * head_k_dim, "d_ssm_state");

    auto ws = cutlass::gdn::get_workspace_sizes(make_shape_only_args());
    alloc(d_A_ws, ws.A_elems, "d_A_ws");
    alloc(d_w_ws, ws.w_elems, "d_w_ws");
    alloc(d_u_ws, ws.u_elems, "d_u_ws");

    // Zero-initialise workspaces (kernel reads after partial writes).
    std::vector<T> zeros_T(std::max({ws.A_elems, ws.w_elems, ws.u_elems}), T{0});
    d_A_ws.copy_from_host(zeros_T.data(), ws.A_elems);
    d_w_ws.copy_from_host(zeros_T.data(), ws.w_elems);
    d_u_ws.copy_from_host(zeros_T.data(), ws.u_elems);

    // Fill tensors with values in production-like ranges (matching the example runner).
    cutlass::initialize_block(d_q,       uint64_t(0xBEEF01));
    cutlass::initialize_block(d_k,       uint64_t(0xBEEF02));
    cutlass::initialize_block(d_v,       uint64_t(0xBEEF03));
    cutlass::initialize_block(d_b,       uint64_t(0xBEEF04), -2.0f, 2.0f);
    cutlass::initialize_block(d_a,       uint64_t(0xBEEF05), -2.0f, 2.0f);
    cutlass::initialize_block(d_A_log,   uint64_t(0xBEEF06), -4.0f, 0.0f);
    cutlass::initialize_block(d_dt_bias, uint64_t(0xBEEF07), T(-2.0f), T(2.0f));

    // Apply sigmoid to b (kernel needs b in (0,1)).
    {
      std::vector<float> h_b(d_b.size());
      d_b.copy_to_host(h_b.data(), h_b.size());
      std::vector<float> h_b_sigmoid =
          cutlass::gdn::reference::stages::apply_sigmoid_b(h_b);
      d_b.copy_from_host(h_b_sigmoid.data(), h_b_sigmoid.size());
    }

    // SSM state starts at zero; has_initial_state all-false.
    {
      std::vector<StateT> zeros_s(d_ssm_state.size(), StateT{0});
      d_ssm_state.copy_from_host(zeros_s.data(), zeros_s.size());
    }

    // query_start_loc and cache_indices: sequential layout.
    {
      const int seq_len = opts.seq_len;
      std::vector<int>     h_qsl(batch + 1, 0);
      std::vector<int>     h_cache(batch, 0);
      std::vector<uint8_t> h_has_init(batch, 0);
      for (int i = 0; i < batch; ++i) {
        h_qsl[i + 1] = h_qsl[i] + seq_len;
        h_cache[i]   = i;
      }
      d_query_start_loc.copy_from_host(h_qsl.data(),    h_qsl.size());
      d_cache_indices.copy_from_host(h_cache.data(),   h_cache.size());
      d_has_initial_state.copy_from_host(h_has_init.data(), h_has_init.size());
    }
  }

  cutlass::gdn::GDNArguments make_shape_only_args() const {
    cutlass::gdn::GDNArguments a{};
    a.batch_size           = batch;
    a.total_seqlen         = total_seqlen;
    a.total_virtual_seqlen = total_virtual_seqlen;
    a.num_k_heads          = num_k_heads;
    a.num_v_heads          = num_v_heads;
    a.head_k_dim           = head_k_dim;
    a.head_v_dim           = head_v_dim;
    a.ssm_state_stride_0   = num_v_heads * head_v_dim * head_k_dim;
    return a;
  }

  cutlass::gdn::GDNArguments make_arguments() const {
    auto a = make_shape_only_args();
    a.q                 = d_q.get();
    a.k                 = d_k.get();
    a.v                 = d_v.get();
    a.b                 = d_b.get();
    a.a                 = d_a.get();
    a.A_log             = d_A_log.get();
    a.dt_bias           = d_dt_bias.get();
    a.query_start_loc   = d_query_start_loc.get();
    a.cache_indices     = d_cache_indices.get();
    a.has_initial_state = reinterpret_cast<bool const*>(d_has_initial_state.get());
    a.core_attn_out     = d_core_attn_out.get();
    a.ssm_state         = d_ssm_state.get();
    a.A_workspace       = d_A_ws.get();
    a.w_workspace       = d_w_ws.get();
    a.u_workspace       = d_u_ws.get();
    return a;
  }

  // ---- performance model ----

  /* FLOP estimate for the chunkwise GDN forward pass.
   * Counts the algebraically required mul-add flops (1 MAC = 2 flops) of each
   * of the five stages; numeric constants in comments are dominant-term
   * derivations, sub-dominant scalar work (softplus, exp, eps adds) is
   * absorbed into small constants.
   *
   * Notation:
   *   C   = kChunkSize (= 64)
   *   D_k = head_k_dim,  D_v = head_v_dim
   *   H_k = num_k_heads, H_v = num_v_heads
   *   T   = total tokens after per-batch padding (== total_virtual_seqlen)
   *   N   = per-batch chunks summed across batches = ceil_sum(seq_len_b / C)
   *         (matches the stride the kernels use; for fixed seq_len:
   *          N = batch * ceil(seq_len / C))
   */
  double flop_estimate() const {
    constexpr int C    = cutlass::gdn::kChunkSize;
    const double T_tot = double(total_virtual_seqlen);
    const double N     = T_tot / double(C);
    const double Hk    = num_k_heads;
    const double Hv    = num_v_heads;
    const double Dk    = head_k_dim;
    const double Dv    = head_v_dim;

    /* Stage 1 -- chunk_prepare:
     *   per k-head per token: L2-norm of q and k each = 2*D_k mul-adds + 1 rsqrt + D_k scales
     *     ~= 5 * D_k flops (q and k together)
     *   per v-head per token: softplus(a+dt_bias) * -exp(A_log) + cumsum
     *     ~= 8 flops */
    const double prepare = 5.0 * T_tot * Hk * Dk + 8.0 * T_tot * Hv;

    /* Stage 2 -- chunk_compute_A:
     *   L[m,n] = (K_m . K_n) * exp(a[m]-a[n]) * b[m]  for m,n in [0,C)
     *   Counted as a full C x C x D_k GEMM (2 flops per MAC); the lower-tri
     *   masking is done post-hoc on the same flops the hardware computes.
     *   Plus the per-element exp-and-multiply scaling (~3 flops). */
    const double compute_A = N * Hv * (2.0 * C * C * Dk + 3.0 * C * C);

    /* Stage 3 -- chunk_inverse:
     *   Block forward substitution to invert a CxC lower-triangular matrix.
     *   Classical complexity is C^3/3 mul-adds = 2*C^3/3 flops. The
     *   DPAS-tiled path performs additional rearrange ops but the asymptotic
     *   flop count is the same. */
    const double inverse = N * Hv * (2.0 * C * C * C / 3.0);

    /* Stage 4 -- chunk_compute_wu:
     *   U = L^-1 * V * diag(b)         -> (CxC) * (CxD_v) GEMM = 2*C^2*D_v
     *   W = L^-1 * K * diag(exp(a)*b)  -> (CxC) * (CxD_k) GEMM = 2*C^2*D_k
     *   Per-element diag scaling adds ~C*(D_v + D_k) flops; negligible. */
    const double compute_wu = N * Hv * 2.0 * C * C * (Dk + Dv);

    /* Stage 5 -- chunk_fwd_o:
     *   O2  = Q * K^T                       -> (CxD_k) * (D_k x C) = 2*C^2*D_k
     *   O_intra = O2 * U                    -> (CxC) * (CxD_v)     = 2*C^2*D_v
     *   O_inter = Q * S^T * exp(g)          -> (CxD_k) * (D_k xD_v) = 2*C*D_k*D_v
     *                                          (S_prev contribution; counted
     *                                           every chunk; the leading chunk
     *                                           with no prev state is a small
     *                                           constant overcharge)
     *   S_out  = exp(g_last)*S_prev + U^T * K_scaled
     *                                       -> (D_v xC) * (CxD_k)   = 2*C*D_k*D_v
     *   Subdominant: per-element exp(g[m]-g[n]) masking & exp(g) scales,
     *   ~3*C^2 flops; absorbed. */
    const double fwd_o = N * Hv * (2.0 * C * C * Dk      // O2 = Q*K^T
                                 + 2.0 * C * C * Dv      // O2 * U
                                 + 2.0 * C * Dk * Dv     // Q * S^T * exp(g)
                                 + 2.0 * C * Dk * Dv);   // S update

    return (prepare + compute_A + inverse + compute_wu + fwd_o) * 1e-9; // GFLOPs
  }

  /* Memory-traffic estimate.
   *
   * Charges each tensor for every kernel-stage in which it is read or
   * written, rather than once over the whole pipeline. Workspaces dominate
   * and are NOT scratched in cache between stages on this hardware, so they
   * are paid for again in every consuming stage. Pre-launch host copies and
   * post-launch readbacks are NOT counted; the benchmark times only kernel
   * execution. */
  double bytes_estimate() const {
    constexpr double sT = sizeof(T);
    constexpr double sS = sizeof(StateT);
    constexpr int C = cutlass::gdn::kChunkSize;

    const double tvs = double(total_virtual_seqlen);
    const double ts  = double(total_seqlen);
    const double Hk  = num_k_heads;
    const double Hv  = num_v_heads;
    const double Dk  = head_k_dim;
    const double Dv  = head_v_dim;
    const double Bn  = batch;

    // ---- per-tensor sizes (in bytes) ----
    const double sz_q       = tvs * Hk * Dk * sT;
    const double sz_k       = tvs * Hk * Dk * sT;
    const double sz_v       = tvs * Hv * Dv * sT;
    const double sz_a       = Hv * tvs * sizeof(float);
    const double sz_b       = Hv * tvs * sizeof(float);
    const double sz_A_log   = Hv * sizeof(float);
    const double sz_dt_bias = Hv * sT;
    const double sz_O       = ts  * Hv * Dv * sT;
    const double sz_ssm     = Bn  * Hv * Dv * Dk * sS;
    const double sz_A_ws    = Hv  * tvs * C   * sT;
    const double sz_w_ws    = Hv  * tvs * Dk  * sT;
    const double sz_u_ws    = Hv  * tvs * Dv  * sT;

    // ---- per-stage traffic ----
    // Stage 1: reads q,k,a,A_log,dt_bias; writes q,k,a (in place).
    const double bytes_prepare    = 2.0*sz_q + 2.0*sz_k + 2.0*sz_a + sz_A_log + sz_dt_bias;
    // Stage 2: reads k,b,a; writes A_workspace.
    const double bytes_compute_A  = sz_k + sz_b + sz_a + sz_A_ws;
    // Stage 3: reads A_workspace; writes A_workspace (in place).
    const double bytes_inverse    = 2.0 * sz_A_ws;
    // Stage 4: reads A_workspace, q, k, v, b, a, A_log, dt_bias; writes w, u.
    const double bytes_compute_wu = sz_A_ws + sz_q + sz_k + sz_v + sz_b + sz_a
                                  + sz_A_log + sz_dt_bias + sz_w_ws + sz_u_ws;
    // Stage 5: reads q, k, a, w, u, A_workspace (as O2 scratch), ssm_state;
    //          writes core_attn_out, A_workspace (O2), ssm_state.
    const double bytes_fwd_o      = sz_q + sz_k + sz_a + sz_w_ws + sz_u_ws
                                  + 2.0 * sz_A_ws + sz_O + 2.0 * sz_ssm;

    const double bytes = bytes_prepare + bytes_compute_A + bytes_inverse
                       + bytes_compute_wu + bytes_fwd_o;
    return bytes * 1e-6;  // MB
  }

  // ---- Google Benchmark integration ----

  void run(::benchmark::State& state,
           const GDNBenchmarkOptions& opts,
           const cutlass::KernelHardwareInfo& /* hw_info */) {

    initialize(opts);

    auto args = make_arguments();

#ifndef CUTLASS_TEST_FOR_CRI
    // Warm-up run before timing (skipped on CRI simulator).
    auto warmup_status = cutlass::gdn::chunk_gated_delta_rule_launch<T, StateT>(queue, args);
    if (warmup_status != cutlass::Status::kSuccess) {
      state.SkipWithError("GDN kernel launch failed during warm-up");
      return;
    }
    queue.wait_and_throw();
#endif

    // Drain any default-queue work (tensor initialisation submitted by
    // initialize_block / copy_from_host) before the timed loop.
    // SYCLTimer::start() calls compat::get_default_queue().wait() internally;
    // if that drains initialisation work on the first iteration it inflates the
    // wait time to thousands of ms and leaves ms_elapsed ≈ 0.
    compat::get_default_queue().wait();

    state.counters["batch"]       = opts.batch;
    state.counters["num_v_heads"] = opts.num_v_heads;
    state.counters["num_k_heads"] = opts.num_k_heads;
    state.counters["head_k_dim"]  = opts.head_k_dim;
    state.counters["head_v_dim"]  = opts.head_v_dim;
    state.counters["seq_len"]     = opts.seq_len;

    initialize_timing_counters(state);

    const double gflop    = flop_estimate();
    const double mega_bytes = bytes_estimate();

    for (auto _ : state) {
      // Time with GPU_Clock (the repo-wide timing utility). Correct on both
      // builds: with CUTLASS_SYCL_PROFILING_ENABLED it sums the 5 stage events
      // that the launcher registers with EventManager (the queue above enables
      // profiling); without it, SYCLTimer falls back to a host wall-clock span
      // bracketed by `compat::get_default_queue().wait()`. The GDN kernel runs
      // on a private in-order queue, not the default queue -- the default queue
      // was drained above (and stays idle) so that internal wait is a no-op,
      // and `queue.wait_and_throw()` on the private queue before
      // `timer.milliseconds()` makes the wall-clock span cover exactly the
      // private-queue launch.
      GPU_Clock timer;
      timer.start();
      auto status = cutlass::gdn::chunk_gated_delta_rule_launch<T, StateT>(queue, args);
      queue.wait_and_throw();
      double ms_elapsed = timer.milliseconds();

      if (status != cutlass::Status::kSuccess) {
        state.SkipWithError("GDN kernel launch failed");
        return;
      }

      update_timing_counters(state, ms_elapsed);
      state.SetIterationTime(ms_elapsed / 1000.0);
    }

    finalize_timing_counters(state, gflop, mega_bytes);
  }

 private:
  static void initialize_timing_counters(::benchmark::State& state) {
    state.counters["total_runtime_ms"]  = 0.0;
    state.counters["avg_runtime_ms"]    = 0.0;
    state.counters["best_runtime_ms"]   = std::numeric_limits<double>::max();
    state.counters["worst_runtime_ms"]  = std::numeric_limits<double>::lowest();
  }

  static void update_timing_counters(::benchmark::State& state, double ms) {
    state.PauseTiming();
    state.counters["total_runtime_ms"] += ms;
    state.counters["best_runtime_ms"]   = std::min<double>(state.counters["best_runtime_ms"],  ms);
    state.counters["worst_runtime_ms"]  = std::max<double>(state.counters["worst_runtime_ms"], ms);
    state.ResumeTiming();
  }

  static void finalize_timing_counters(::benchmark::State& state,
                                       double gflop, double mega_bytes) {
    const auto iters = static_cast<double>(state.iterations());
    // Trimmed mean: remove best + worst if we have enough iterations.
    double denom = (iters > 2) ? (iters - 2) : iters;
    double trimmed_total = state.counters["total_runtime_ms"]
                         - state.counters["best_runtime_ms"]
                         - state.counters["worst_runtime_ms"];
    if (iters <= 2) trimmed_total = state.counters["total_runtime_ms"];
    state.counters["avg_runtime_ms"]   = trimmed_total / denom;
    state.counters["avg_tflops"]       = gflop      / state.counters["avg_runtime_ms"];
    state.counters["avg_throughput"]   = mega_bytes  / state.counters["avg_runtime_ms"];
    state.counters["best_tflop"]       = gflop      / state.counters["best_runtime_ms"];
    state.counters["best_bandwidth"]   = mega_bytes  / state.counters["best_runtime_ms"];
  }
};

}  // namespace cutlass::benchmark::gdn

// ---------------------------------------------------------------------------
// Registration macros (mirrors CUTLASS_CREATE_FMHA_BENCHMARK pattern)
//
// CUTLASS_CREATE_GDN_BENCHMARK(F) — define a static trampoline function for F.
// CUTLASS_GDN_BENCHMARK(F)        — register F with the BenchmarkRegistry.
// ---------------------------------------------------------------------------

#define CUTLASS_GDN_BENCHMARK(F) \
  cutlass::benchmark::BenchmarkRegistry<cutlass::benchmark::gdn::GDNBenchmarkOptions>::Register( \
      #F, &F##_func)

#define CUTLASS_CREATE_GDN_BENCHMARK(F)                                   \
  static void F##_func(                                                   \
      ::benchmark::State& state,                                          \
      cutlass::benchmark::gdn::GDNBenchmarkOptions const& options,        \
      cutlass::KernelHardwareInfo const& hw_info) {                       \
    auto bench = cutlass::benchmark::gdn::BenchmarkRunnerGDN<             \
        typename F::ElementT, typename F::StateT>();                      \
    bench.run(state, options, hw_info);                                   \
  }
