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

/*!
  \file xe35_gdn_attention_runner.hpp
  \brief Host harness for the Xe35 chunkwise Gated DeltaNet attention
         example. Allocates inputs, calls the kernel, optionally verifies
         against the fp32 recurrent host reference
         (cutlass/util/reference/host/xe35_gdn_attention_recurrent_reference.hpp)
         and reports timing -- modeled on examples/06_bmg_flash_attention's
         runner.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/initialize_block.hpp"

#include "sycl_common.hpp"

#include "gdn_attention/xe35_chunk_gated_delta_rule_launch.hpp"
#include "xe35_gdn_attention_perf.hpp"
// Per-stage chunkwise references (apply_sigmoid_b + per-kernel oracles).
#include "cutlass/util/reference/host/xe35_gdn_attention_stage_references.hpp"
// recurrent reference: the E2E verify oracle.
#include "cutlass/util/reference/host/xe35_gdn_attention_recurrent_reference.hpp"

using cutlass::DeviceAllocation;

/* ---------------------------------------------------------------------------
 * Options
 * --------------------------------------------------------------------------- */

struct Options {
  bool help = false;
  bool error = false;
  /* Set by parse() to true iff the user explicitly supplied any of
   * --batch / --num_v_heads / --num_k_heads / --head_k_dim / --head_v_dim /
   * --seq_len. The example driver (xe35_gdn_attention.cpp) uses this to decide
   * between iterating its hard-coded case table and running a single
   * CLI-driven case. */
  bool shape_overridden = false;

  /* Defaults sized for the CRI simulator: a single sub-chunk problem
   * (seq_len < kChunkSize=64) with minimal head count, no warmup, one
   * timed iteration. On BMG (real hardware) the ctest passes no shape
   * flags, so the driver iterates its built-in case table instead of this
   * single case; override on the command line to run an arbitrary shape on
   * either target. */
  int batch       = 1;
  int num_v_heads = 4;
  int num_k_heads = 1;        // GQA: num_v_heads must be a multiple of num_k_heads
  /* Head dims default to the compile-time HEAD_K_DIM / HEAD_V_DIM (128 / 128)
   * but are overridable via --head_k_dim / --head_v_dim. They must be positive
   * multiples of the kernel chunk size (kChunkSize=64): the kernels iterate
   * `head_*_dim / chunk_size` 2D blocks, so a non-multiple silently drops the
   * remainder. Only 128/128 is validated end-to-end by the unit test. */
  int head_k_dim  = HEAD_K_DIM;
  int head_v_dim  = HEAD_V_DIM;
  int seq_len     = 16;       // chunked kernel pads internally to kChunkSize=64
  int iterations  = 1;
  int warmup      = 0;
  int verify      = 1;
  unsigned seed   = 0xC0FFEEu;

  std::ostream& print_usage(std::ostream& os) const {
    os << "14_xe35_gdn_attention -- Xe35 chunkwise Gated DeltaNet attention\n\n"
       << "Defaults are sized for the CRI simulator (single sub-chunk);\n"
       << "with no shape flags the driver iterates its built-in case table.\n\n"
       << "Options:\n"
       << "  --help                   Show this help\n"
       << "  --batch=<int>            Number of sequences in the batch (default 1)\n"
       << "  --num_v_heads=<int>      Number of value heads (default 4)\n"
       << "  --num_k_heads=<int>      Number of key heads (default 1, GQA: must divide num_v_heads)\n"
       << "  --head_k_dim=<int>       Key/query head dim (default " << HEAD_K_DIM << "; must be a multiple of chunk size 64)\n"
       << "  --head_v_dim=<int>       Value head dim (default " << HEAD_V_DIM << "; must be a multiple of chunk size 64)\n"
       << "  --seq_len=<int>          Tokens per sequence (default 16; kernel chunk size is 64)\n"
       << "  --iterations=<int>       Timed iterations (default 1)\n"
       << "  --warmup=<int>           Warmup iterations (default 0)\n"
       << "  --verify=<0|1>           Run host reference and compare (default 1)\n"
       << "  --seed=<int>             RNG seed (default 0xC0FFEE)\n"
       << "\nDefault head dims (compile-time HEAD_K_DIM / HEAD_V_DIM): head_k_dim="
       << HEAD_K_DIM << " head_v_dim=" << HEAD_V_DIM << "\n";
    return os;
  }

  void parse(int argc, const char** argv) {
    cutlass::CommandLine cmd(argc, argv);
    if (cmd.check_cmd_line_flag("help")) { help = true; return; }
    /* Snapshot whether any shape flag was supplied BEFORE calling
     * get_cmd_line_argument; the cutlass CommandLine helper has no
     * "was value defaulted" return, so we probe the keys directly.
     * check_cmd_line_flag matches both bare `--name` and `--name=value`. */
    shape_overridden = cmd.check_cmd_line_flag("batch")       ||
                       cmd.check_cmd_line_flag("num_v_heads") ||
                       cmd.check_cmd_line_flag("num_k_heads") ||
                       cmd.check_cmd_line_flag("head_k_dim")  ||
                       cmd.check_cmd_line_flag("head_v_dim")  ||
                       cmd.check_cmd_line_flag("seq_len");
    cmd.get_cmd_line_argument("batch",       batch,       batch);
    cmd.get_cmd_line_argument("num_v_heads", num_v_heads, num_v_heads);
    cmd.get_cmd_line_argument("num_k_heads", num_k_heads, num_k_heads);
    cmd.get_cmd_line_argument("head_k_dim",  head_k_dim,  head_k_dim);
    cmd.get_cmd_line_argument("head_v_dim",  head_v_dim,  head_v_dim);
    cmd.get_cmd_line_argument("seq_len",     seq_len,     seq_len);
    cmd.get_cmd_line_argument("iterations",  iterations,  iterations);
    cmd.get_cmd_line_argument("warmup",      warmup,      warmup);
    cmd.get_cmd_line_argument("verify",      verify,      verify);
    int seed_int = static_cast<int>(seed);
    cmd.get_cmd_line_argument("seed", seed_int, seed_int);
    seed = static_cast<unsigned>(seed_int);

    // Reject non-positive shape parameters BEFORE the modulo below so a
    // user passing --num_k_heads=0 cannot trigger a divide-by-zero. Matches
    // the validation order in xe35_chunk_gated_delta_rule_launch.hpp.
    if (num_k_heads <= 0 || num_v_heads <= 0 || head_k_dim <= 0 ||
        head_v_dim <= 0 || seq_len <= 0 || batch <= 0) {
      std::cerr << "Error: shape parameters must be positive\n";
      error = true;
    }
    if (!error && num_v_heads % num_k_heads != 0) {
      std::cerr << "Error: num_v_heads (" << num_v_heads
                << ") must be a multiple of num_k_heads (" << num_k_heads << ")\n";
      error = true;
    }
    // The kernels tile the head dims into kChunkSize-wide 2D blocks
    // (`for (dv = 0; dv < head_v_dim / chunk_size; ++dv)` etc.), so a head dim
    // that is not a multiple of kChunkSize would silently drop its remainder.
    // Reject it here rather than produce wrong results.
    constexpr int C = cutlass::gdn::kChunkSize;
    if (!error && (head_k_dim % C != 0 || head_v_dim % C != 0)) {
      std::cerr << "Error: head_k_dim (" << head_k_dim << ") and head_v_dim ("
                << head_v_dim << ") must each be a multiple of the chunk size ("
                << C << ")\n";
      error = true;
    }
  }
};

/* ---------------------------------------------------------------------------
 * Runner
 * --------------------------------------------------------------------------- */

template <typename T, typename StateT>
struct GdnRunner {

  Options opt;
  /* In-order queue: kernel_launcher submits prepare -> compute_A -> inverse ->
   * compute_wu -> fwd_o back-to-back without explicit .wait() between them.
   * A default (out-of-order) queue lets those kernels race on the shared
   * workspaces A/w/u, which manifests as fwd_o reading partially-written
   * tiles ("only first 2D block populated, rest zero") on slow simulators
   * like CRI. Force in-order semantics to match the CUDA-stream model the
   * baseline implementation relies on.
   *
   * Profiling property: GPU_Clock/SYCLTimer sums each kernel's device-side
   * command_start/command_end timestamps (kernel_launcher registers the 5
   * stage events with EventManager), and get_profiling_info() requires the
   * queue to have been created with enable_profiling(). Add it only in the
   * profiling build; the default build times with GPU_Clock's host wall-clock
   * fallback (see perf::time_launches) and needs no extra property. */
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  sycl::queue queue{sycl::property_list{
      sycl::property::queue::in_order(),
      sycl::property::queue::enable_profiling()}};
#else
  sycl::queue queue{sycl::property_list{
      sycl::property::queue::in_order()}};
#endif

  // shape
  int batch = 0;
  int num_v_heads = 0;
  int num_k_heads = 0;
  int head_k_dim = 0;
  int head_v_dim = 0;
  int seq_len = 0;
  int total_seqlen = 0;
  int total_virtual_seqlen = 0;  // padded to multiples of kChunkSize per sequence
  int padded_seqlen_global = 0;  // total_seqlen + batch*(kChunkSize-1)

  // device allocations
  DeviceAllocation<T>     d_q, d_k, d_v;
  DeviceAllocation<T>     d_dt_bias;
  DeviceAllocation<float> d_b, d_a, d_A_log;
  DeviceAllocation<int>   d_query_start_loc, d_cache_indices;
  /* Stored as uint8_t (1 byte) -- cutlass::sizeof_bits<bool> is 1 bit, which
   * would make DeviceAllocation<bool>::reset(N) request floor(N/8) bytes
   * (= 0 for our small N) and trip the "Failed to allocate memory" path.
   * We reinterpret_cast to bool* at the kernel call boundary. */
  DeviceAllocation<uint8_t> d_has_initial_state;
  DeviceAllocation<T>     d_core_attn_out;
  DeviceAllocation<StateT> d_ssm_state;

  /* workspaces (intermediates produced by the prepare / compute_A /
   * inverse / compute_wu stages and consumed by fwd_o) */
  DeviceAllocation<T> d_A_ws, d_w_ws, d_u_ws;

  /* Raw (pre-sigmoid) snapshot of b. Both the kernel and the reference consume
   * sigmoid(b); the snapshot keeps a clean pre-sigmoid copy so each path
   * sigmoids exactly once:
   *   (1) initialize d_b with raw values in [-2, 2],
   *   (2) snapshot them here into h_b_raw,
   *   (3) sigmoid d_b in-place (stand-in for the conv1d front-end) to feed the kernel,
   *   (4) at verify time, sigmoid h_b_raw via apply_sigmoid_b and hand the result
   *       to the reference (which consumes sigmoid(b) directly, no further sigmoid).
   * Without the snapshot, verify would copy the already-sigmoided d_b back to
   * host and sigmoid it a second time. */
  std::vector<float> h_b_raw;

  /* Pre-kernel snapshots of q, k, v, a. The kernel's prepare stage mutates
   * q, k, a IN PLACE on the device (L2-normalize q/k; cumsum a). The recurrent
   * reference normalizes q/k itself, so it must receive the raw pre-kernel
   * values, not the device buffers read back after the launch. v is not mutated
   * but is captured too for a single source of truth. */
  std::vector<T>     h_q_raw, h_k_raw, h_v_raw;
  std::vector<float> h_a_raw;
  /* Pre-kernel ssm_state snapshot (used as the reference's starting state).
   * The kernel updates d_ssm_state in place, so we cannot read it back after
   * the kernel run and call that "the initial state". */
  std::vector<StateT> h_ssm_initial_raw;

  explicit GdnRunner(Options o) : opt(std::move(o)) {}

  size_t ssm_state_elems() const {
    return size_t(opt.batch) * opt.num_v_heads * opt.head_v_dim * opt.head_k_dim;
  }

  // ---------- setup ----------

  void initialize() {
    batch       = opt.batch;
    num_v_heads = opt.num_v_heads;
    num_k_heads = opt.num_k_heads;
    head_k_dim  = opt.head_k_dim;
    head_v_dim  = opt.head_v_dim;
    seq_len     = opt.seq_len;
    total_seqlen         = batch * seq_len;
    constexpr int C      = cutlass::gdn::kChunkSize;
    int per_seq_padded   = ((seq_len + C - 1) / C) * C;
    total_virtual_seqlen = batch * per_seq_padded;
    padded_seqlen_global = total_seqlen + batch * (C - 1);

    auto alloc = [](auto& da, size_t n, char const* tag) {
      try {
        da.reset(n);
      } catch (std::exception const& e) {
        std::cerr << "[14_gdn] alloc FAILED for " << tag
                  << " (n=" << n << "): " << e.what() << "\n";
        throw;
      }
    };

    alloc(d_q, size_t(total_virtual_seqlen) * num_k_heads * head_k_dim, "d_q");
    alloc(d_k, size_t(total_virtual_seqlen) * num_k_heads * head_k_dim, "d_k");
    alloc(d_v, size_t(total_virtual_seqlen) * num_v_heads * head_v_dim, "d_v");
    alloc(d_b, size_t(num_v_heads) * total_virtual_seqlen, "d_b");
    alloc(d_a, size_t(num_v_heads) * total_virtual_seqlen, "d_a");
    alloc(d_A_log,   num_v_heads, "d_A_log");
    alloc(d_dt_bias, num_v_heads, "d_dt_bias");
    alloc(d_query_start_loc, batch + 1, "d_query_start_loc");
    alloc(d_cache_indices,   batch, "d_cache_indices");
    alloc(d_has_initial_state, batch, "d_has_initial_state");
    alloc(d_core_attn_out, size_t(total_seqlen) * num_v_heads * head_v_dim, "d_core_attn_out");
    alloc(d_ssm_state, size_t(batch) * num_v_heads * head_v_dim * head_k_dim, "d_ssm_state");

    auto ws = cutlass::gdn::get_workspace_sizes(make_arguments_shape_only());
    alloc(d_A_ws, ws.A_elems, "d_A_ws");
    alloc(d_w_ws, ws.w_elems, "d_w_ws");
    alloc(d_u_ws, ws.u_elems, "d_u_ws");

    // Zero-init workspaces (kernel reads after partial writes).
    std::vector<T> zeros_T(std::max({ws.A_elems, ws.w_elems, ws.u_elems}), T{0});
    d_A_ws.copy_from_host(zeros_T.data(), ws.A_elems);
    d_w_ws.copy_from_host(zeros_T.data(), ws.w_elems);
    d_u_ws.copy_from_host(zeros_T.data(), ws.u_elems);

    /* Random fill for floating-point tensors using cutlass helpers.
     * 
     *   chunk_prepare_kernel computes per-token gate:
     *     g = softplus(a + dt_bias) * (-exp(A_log))
     *   then cumulative-sums g over a chunk into a[].
     * 
     *   The default `initialize_block(float)` scope is [-64, 64] (derived from
     *   max_for_test = 1 << ceil_div(digits<float>=24, 4) = 1 << 6 = 64).
     *   With A_log up to +64, exp(A_log) reaches ~6e27 -- the source of the
     *   "a max_abs=7.5e+20" we observed, which then makes compute_A overflow
     *   to +-inf and inverse / compute_wu / fwd_o produce NaN.
     * 
     * Use ranges that match production GDN model statistics:
     *   - A_log  = log(A) with A in (0, 1]  ->  A_log in (-inf, 0]; clip to [-4, 0]
     *   - a, dt_bias: modest pre-activation values, [-2, 2]
     *   - b: ranged [-2, 2] then sigmoided below to (0, 1)
     *   - q, k, v: bf16 default range from initialize_block (already modest) */
    cutlass::initialize_block(d_q,        static_cast<uint64_t>(opt.seed) + 1);
    cutlass::initialize_block(d_k,        static_cast<uint64_t>(opt.seed) + 2);
    cutlass::initialize_block(d_v,        static_cast<uint64_t>(opt.seed) + 3);
    cutlass::initialize_block(d_b,        static_cast<uint64_t>(opt.seed) + 4, -2.0f, 2.0f);
    cutlass::initialize_block(d_a,        static_cast<uint64_t>(opt.seed) + 5, -2.0f, 2.0f);
    cutlass::initialize_block(d_A_log,    static_cast<uint64_t>(opt.seed) + 6, -4.0f, 0.0f);
    cutlass::initialize_block(d_dt_bias,  static_cast<uint64_t>(opt.seed) + 7, T(-2.0f), T(2.0f));

    /* Apply sigmoid to b. Rationale:
     * 
     *   The chunk path of GDN attention assumes that `b` it receives is already
     *   in (0, 1) -- i.e., the output of a sigmoid activation. In a full GDN
     *   stack, raw `projected_states_ba` is funneled through a causal conv1d
     *   front-end before reaching the chunk kernel, and that front-end is
     *   what produces the in-range `b` that the chunk kernels consume. The
     *   non-chunk reference path makes this explicit via `act_sigmoid(b)`.
     * 
     *   This example bypasses the conv1d preprocessor and feeds the chunk
     *   launcher directly, so we must apply sigmoid ourselves to honor the
     *   kernel's input contract. Without it, raw `initialize_block` values
     *   (~[-2, 2]) make L_strict[m,n] = (K_m . K_n) * exp(a[m]-a[n]) * b[m] of
     *   order O(1), and the 64-step forward substitution in the inverse stage
     *   overflows to +/-inf, propagating NaN through compute_wu and fwd_o.
     * 
     * Snapshot the raw b BEFORE the sigmoid so the host reference (which
     * applies its own sigmoid internally) does not sigmoid twice. See the
     * comment on h_b_raw above. */
    h_b_raw.assign(d_b.size(), 0.0f);
    d_b.copy_to_host(h_b_raw.data(), h_b_raw.size());
    {
      std::vector<float> h_b_sigmoid =
          cutlass::gdn::reference::stages::apply_sigmoid_b(h_b_raw);
      d_b.copy_from_host(h_b_sigmoid.data(), h_b_sigmoid.size());
    }

    /* ------------------------------------------------------------------
     * Padded chunk slots intentionally left at their random initialization.
     * 
     * Earlier revisions zeroed q/k/v and pushed a sentinel into the padded
     * tail of `a` to keep the kernel's chunkwise math from leaking padded
     * contributions into the per-sequence final state. That masking pass is
     * now redundant: with has_initial_state=false and the strict-lower-tri
     * structure of the chunkwise recurrence, padded slots cannot affect
     * real-token outputs (confirmed by the E2E verify passing at 5e-2).
     * Removing the masking pass also removes four full device<->host round
     * trips from initialize().
     * ------------------------------------------------------------------ */

    /* ssm_state starts at zero on both device and reference -- has_initial_state
     * is all-false in this example, so the kernel must ignore the buffer's prior
     * contents. Seeding it to zero keeps device and reference inputs identical
     * (avoids the trap of random device init vs. zero-default host vector). */
    {
      std::vector<StateT> zeros_state(ssm_state_elems(), StateT{0});
      d_ssm_state.copy_from_host(zeros_state.data(), zeros_state.size());
    }

    // host-built integer/bool tensors
    std::vector<int>  h_qsl(batch + 1, 0);
    std::vector<int>  h_cache(batch, 0);
    std::vector<uint8_t> h_has_init(batch, 0); // start with no initial state
    for (int i = 0; i < batch; ++i) {
      h_qsl[i + 1] = h_qsl[i] + seq_len;
      h_cache[i]   = i;
    }
    d_query_start_loc.copy_from_host(h_qsl.data(), h_qsl.size());
    d_cache_indices.copy_from_host(h_cache.data(), h_cache.size());
    d_has_initial_state.copy_from_host(h_has_init.data(), h_has_init.size());

    /* ------------------------------------------------------------------
     * Snapshot raw inputs for the host reference. The kernel mutates q, k, a,
     * ssm_state in place; the recurrent reference needs the pre-kernel values. */
    h_q_raw.assign(d_q.size(), T{0});
    h_k_raw.assign(d_k.size(), T{0});
    h_v_raw.assign(d_v.size(), T{0});
    h_a_raw.assign(d_a.size(), 0.0f);
    h_ssm_initial_raw.assign(d_ssm_state.size(), StateT{0});
    d_q.copy_to_host(h_q_raw.data(), h_q_raw.size());
    d_k.copy_to_host(h_k_raw.data(), h_k_raw.size());
    d_v.copy_to_host(h_v_raw.data(), h_v_raw.size());
    d_a.copy_to_host(h_a_raw.data(), h_a_raw.size());
    d_ssm_state.copy_to_host(h_ssm_initial_raw.data(), h_ssm_initial_raw.size());
  }

  cutlass::gdn::GDNArguments make_arguments_shape_only() const {
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
    auto a = make_arguments_shape_only();
    a.q                  = d_q.get();
    a.k                  = d_k.get();
    a.v                  = d_v.get();
    a.b                  = d_b.get();
    a.a                  = d_a.get();
    a.A_log              = d_A_log.get();
    a.dt_bias            = d_dt_bias.get();
    a.query_start_loc    = d_query_start_loc.get();
    a.cache_indices      = d_cache_indices.get();
    a.has_initial_state  = reinterpret_cast<bool const*>(d_has_initial_state.get());
    a.core_attn_out      = d_core_attn_out.get();
    a.ssm_state          = d_ssm_state.get();
    a.A_workspace        = d_A_ws.get();
    a.w_workspace        = d_w_ws.get();
    a.u_workspace        = d_u_ws.get();
    return a;
  }

  // ---------- run + verify ----------

  // E2E verify against the recurrent reference
  bool verify_against_reference() {
    using cutlass::gdn::reference::recurrent::run_recurrent_gdn_attn_reference;
    namespace ref = cutlass::gdn::reference::recurrent;

    // ------- collect host-side inputs (raw snapshots where mutated) ------
    const size_t ba_n  = d_b.size();
    const size_t ssm_n = d_ssm_state.size();
    const size_t out_n = d_core_attn_out.size();

    std::vector<float> h_A_log(num_v_heads);
    std::vector<T>     h_dt_bias(num_v_heads);
    std::vector<int>   h_qsl(batch + 1), h_cache(batch);
    std::vector<uint8_t> h_has_init(batch);
    std::vector<T>      h_out_dev(out_n);
    std::vector<StateT> h_ssm_dev(ssm_n);

    d_A_log.copy_to_host(h_A_log.data(), h_A_log.size());
    d_dt_bias.copy_to_host(h_dt_bias.data(), h_dt_bias.size());
    d_query_start_loc.copy_to_host(h_qsl.data(), h_qsl.size());
    d_cache_indices.copy_to_host(h_cache.data(), h_cache.size());
    d_has_initial_state.copy_to_host(h_has_init.data(), h_has_init.size());
    d_ssm_state.copy_to_host(h_ssm_dev.data(), ssm_n);
    d_core_attn_out.copy_to_host(h_out_dev.data(), out_n);

    // The reference consumes sigmoid(b); h_b_raw is the pre-sigmoid snapshot.
    assert(h_b_raw.size() == ba_n);
    std::vector<float> h_b_sigmoid =
        cutlass::gdn::reference::stages::apply_sigmoid_b(h_b_raw);

    // Mutable copies: the reference normalizes q,k in place; a stays raw.
    std::vector<T>     h_q = h_q_raw;
    std::vector<T>     h_k = h_k_raw;
    std::vector<float> h_a = h_a_raw;
    std::vector<T>     h_out_ref(out_n, T{0});
    std::vector<StateT> h_ssm_ref = h_ssm_initial_raw;  // updated in place

    // ------- run fast recurrent reference -------
    run_recurrent_gdn_attn_reference<T, StateT>(
        h_out_ref, h_ssm_ref,
        h_q, h_k, h_v_raw,
        h_b_sigmoid, h_a,
        h_A_log, h_dt_bias,
        h_qsl, h_cache, h_has_init,
        batch, num_k_heads, num_v_heads,
        head_k_dim, head_v_dim, total_virtual_seqlen);

    // per-element compare at kTolE2E (like FMHA).
    auto compare = [&](char const* tag, auto const& r, auto const& dev) {
      auto s = cutlass::gdn::perf::compare_with_stats(r, dev, ref::kTolE2E, ref::kTolE2E);
      cutlass::gdn::perf::print_compare_stats(tag, s);
      return s.passed();
    };

    bool ok_out = compare("core_attn_out", h_out_ref, h_out_dev);
    bool ok_ssm = compare("ssm_state    ", h_ssm_ref, h_ssm_dev);
    return ok_out && ok_ssm;
  }

  int run() {
    initialize();

    auto args = make_arguments();

    auto status = cutlass::gdn::chunk_gated_delta_rule_launch<T, StateT>(
        queue, args);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "[error] kernel launch failed: status=" << int(status) << "\n";
      return -1;
    }
    queue.wait_and_throw();

    if (opt.verify) {
      if (!verify_against_reference()) {
        std::cout << "[verify] FAIL  device output does not match host reference\n";
        return -2;
      }
      std::cout << "[verify] PASS\n";
    }

    // Warmup + timing (matches the pattern used by 01_bmg_gemm_with_collective_builder).
    const double ms = cutlass::gdn::perf::time_launches(
        queue, opt.iterations, opt.warmup,
        [&]{ cutlass::gdn::chunk_gated_delta_rule_launch<T, StateT>(queue, args); });

    std::cout << "Avg kernel time: " << ms << " ms over "
              << opt.iterations << " iterations\n"
              << "  batch=" << batch
              << " seq_len=" << seq_len
              << " num_v_heads=" << num_v_heads
              << " num_k_heads=" << num_k_heads
              << " head_k_dim=" << head_k_dim
              << " head_v_dim=" << head_v_dim << "\n";
    return 0;
  }
};
