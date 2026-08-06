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
    \brief Google-Benchmark harness for the hand-written MoE grouped-GEMM kernel
           (applications/moe_grouped_gemm/MoEGEMM), i.e. the same kernel that
           example 12 launches.

    The existing benchmarks/grouped_gemm harness is built around the standard
    GemmUniversalAdapter (can_implement / initialize / run). The MoE kernel is a
    raw parallel_for launch with a custom tile scheduler + packed-A + a device
    num_rows_per_expert table, with no GemmUniversal interface, so it needs its
    own runner. It still reuses the shared scaffolding from benchmarks/common.hpp
    (BenchmarkRegistry, register_benchmarks, benchmark_main) and Google Benchmark.

    All the kernel machinery (choose_tiled_mma, the Config structs, fill_scale,
    MoE::MoEGEMM, the PersistentTileSchedulerXeMoE setup) is reused directly from
    example 12's joint header. We only benchmark ONE GEMM of the exact N/K/experts
    /M described in the config line (no up-gate 2x / down-proj expansion).
*/

#pragma once

// LEAN benchmark TU: this header (and main.cpp) compiles Google Benchmark +
// common.hpp scaffolding but NOT the cute / MoE kernel, which lives only in
// moe_kernel_launch.cpp behind the thin moe_kernel_launch.hpp interface. See
// that header for the TU-split rationale (avoids the IGC ICE on CRI).
#include "../common.hpp"
#include <benchmark/benchmark.h>

// THIN interface to the kernel launch. No cute / MoE / SYCL / oneMKL — just an
// opaque handle + 3 free functions. The chosen Config (Bf16Config / ...) is
// baked into moe_kernel_launch.cpp at build time via -DMOE_BENCH_CONFIG, so this
// benchmark TU never names a cute Config type.
#include "moe_kernel_launch.hpp"

#include <algorithm>
#include <cfloat>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

namespace cutlass::benchmark {

///////////////////////////////////////////////////////////////////////////////////////////////////

// Fixed seed for --random MoE routing so a given config line is reproducible.
inline constexpr unsigned kRoutingSeed = 0x4D6F45u; // "MoE"

// Whether a benchmark line runs verification. `false` (default) is perf-only;
// `true` routes to the example's VerificationHelper.

// Command line options for one MoE grouped-GEMM benchmark line. Builds a
// per-expert M vector from one of three shape sources (highest precedence first):
//   1. --m_per_expert=<csv>   explicit per-expert M list
//   2. --moe_mode (--m/--topk/--num_experts/--ep_size)  MoE routing math,
//      following PR #687: experts_per_gpu = num_experts/ep_size,
//      tokens_per_gpu = (m*topk)/ep_size spread uniformly over experts_per_gpu.
//   3. --uniform_m            legacy uniform M per expert
struct MoEBenchmarkOptions {

  bool error;

  int n, k, num_experts, uniform_m;
  // MoE routing parameters (PR #687 semantics). moe_mode selects source #2.
  bool moe_mode;
  int m, topk, ep_size;
  bool random_mode;
  bool verify;
  std::string m_per_expert; // comma-separated per-expert M list
  std::string bm_name;

  // ByteDance reference-API vocabulary (Yanfei/Wei thread, requirement #3:
  // "changed to the MoE format, i.e., configured using
  // experts_token_count/offset"). These are the canonical names; the legacy
  // --m/--n/--k/--m_per_expert remain as silent aliases so the committed .in
  // files keep working. BD name wins when both are given.
  //   num_tokens          <- total routed tokens (== legacy --m)
  //   hidden_size         <- model hidden dim
  //   new_hidden_size     <- intermediate / expert dim
  //   proj = up|down      <- which projection, decides the N/K assignment:
  //                            up   : K=hidden_size,     N=new_hidden_size
  //                            down : K=new_hidden_size, N=hidden_size
  //   experts_token_count <- per-expert token counts (== legacy --m_per_expert)
  //   experts_token_offset<- per-expert prefix-sum; if given, validated to be
  //                          exactly the running sum of experts_token_count
  //   num_experts_per_rank<- experts on this rank (== num_experts/ep_size); if
  //                          given, overrides num_experts for the per-rank GEMM
  int hidden_size, new_hidden_size, num_experts_per_rank;
  std::string proj; // "up" | "down" | "" (raw n/k)
  std::string experts_token_offset; // comma-separated prefix sum (optional)

  // Per-expert M list (the "groups").
  std::vector<int> rows_per_expert;

  MoEBenchmarkOptions()
      : error(false), n(2880), k(2880), num_experts(8), uniform_m(128),
        moe_mode(false), m(4096), topk(1), ep_size(1), random_mode(false),
        verify(false), m_per_expert(""), bm_name("MoEGEMM"),
        hidden_size(0), new_hidden_size(0), num_experts_per_rank(0), proj(""),
        experts_token_offset("") {
    build_rows();
  }

  void build_rows() {
    rows_per_expert.clear();
    if (!m_per_expert.empty()) {
      std::stringstream ss(m_per_expert);
      std::string token;
      while (std::getline(ss, token, ',')) {
        if (token.empty())
          continue;
        rows_per_expert.push_back(std::stoi(token));
      }
      // num_experts follows the csv length.
      num_experts = static_cast<int>(rows_per_expert.size());
    } else if (moe_mode) {
      // PR #687 MoE routing math: distribute the per-GPU token budget over the
      // per-GPU experts. ep_size>1 shards experts across GPUs (expert
      // parallelism); we benchmark one GPU's share.
      const int experts_per_gpu = std::max(1, num_experts / ep_size);
      const int tokens_per_gpu = (m * topk) / ep_size;
      num_experts = experts_per_gpu;
      if (random_mode) {
        // Randomized routing: assign each token to a random expert, modelling
        // the load imbalance of real top-k routing. Deterministically seeded so
        // a given config line is reproducible across runs.
        rows_per_expert.assign(experts_per_gpu, 0);
        std::mt19937 rng(kRoutingSeed);
        std::uniform_int_distribution<int> pick(0, experts_per_gpu - 1);
        for (int t = 0; t < tokens_per_gpu; ++t)
          rows_per_expert[pick(rng)] += 1;
      } else {
        // Uniform routing: even split, remainder spread so total M is exact.
        const int base = tokens_per_gpu / experts_per_gpu;
        const int rem = tokens_per_gpu % experts_per_gpu;
        rows_per_expert.assign(experts_per_gpu, base);
        for (int i = 0; i < rem && i < experts_per_gpu; ++i)
          rows_per_expert[i] += 1;
      }
    } else {
      int mm = (uniform_m > 0) ? uniform_m : 128;
      rows_per_expert.assign(num_experts, mm);
    }
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    cmd.get_cmd_line_argument("n", n, 2880);
    cmd.get_cmd_line_argument("k", k, 2880);
    cmd.get_cmd_line_argument("num_experts", num_experts, 8);
    cmd.get_cmd_line_argument("uniform_m", uniform_m, 128);
    cmd.get_cmd_line_argument("m_per_expert", m_per_expert, std::string(""));
    cmd.get_cmd_line_argument("bm_name", bm_name, std::string("MoEGEMM"));

    // PR #687 MoE routing parameters.
    moe_mode = cmd.check_cmd_line_flag("moe_mode");
    cmd.get_cmd_line_argument("m", m, 4096);
    cmd.get_cmd_line_argument("topk", topk, 1);
    cmd.get_cmd_line_argument("ep_size", ep_size, 1);
    random_mode = cmd.check_cmd_line_flag("random");

    // ---- ByteDance reference-API names (requirement #3). Read after the
    // legacy flags so a BD name, when present, overrides its alias. ----
    int num_tokens = 0;
    cmd.get_cmd_line_argument("num_tokens", num_tokens, 0);
    int num_of_tokens = 0;
    cmd.get_cmd_line_argument("num_of_tokens", num_of_tokens, 0);
    if (num_of_tokens > 0)
      m = num_of_tokens; // alias used by CRI .in files
    else if (num_tokens > 0)
      m = num_tokens; // total routed tokens == legacy --m

    cmd.get_cmd_line_argument("hidden_size", hidden_size, 0);
    cmd.get_cmd_line_argument("new_hidden_size", new_hidden_size, 0);
    cmd.get_cmd_line_argument("proj", proj, std::string(""));
    // Map (hidden_size, new_hidden_size, proj) -> raw (n, k). Up-projection
    // reads hidden and writes new_hidden (K=hidden, N=new_hidden); down swaps.
    if (hidden_size > 0 && new_hidden_size > 0) {
      if (proj == "down") {
        k = new_hidden_size;
        n = hidden_size;
      } else {
        // default + "up": K=hidden, N=new_hidden.
        if (!proj.empty() && proj != "up") {
          std::cerr << "Error: --proj must be 'up' or 'down' (got '" << proj
                    << "').\n";
          error = true;
        }
        k = hidden_size;
        n = new_hidden_size;
      }
    }

    // experts_token_count is the BD name for the per-expert M list. It aliases
    // --m_per_expert; BD name wins if both are present.
    std::string experts_token_count;
    cmd.get_cmd_line_argument("experts_token_count", experts_token_count,
                              std::string(""));
    if (!experts_token_count.empty())
      m_per_expert = experts_token_count;

    // num_experts_per_rank (BD) == experts on this rank. When given without the
    // routing math, it sets num_experts directly for the per-rank GEMM.
    cmd.get_cmd_line_argument("num_experts_per_rank", num_experts_per_rank, 0);
    if (num_experts_per_rank > 0 && !moe_mode)
      num_experts = num_experts_per_rank;

    cmd.get_cmd_line_argument("experts_token_offset", experts_token_offset,
                              std::string(""));

    std::string verify_str;
    cmd.get_cmd_line_argument("verify", verify_str, std::string("false"));
    verify = (verify_str == "true" || verify_str == "1");

    if (moe_mode && (topk <= 0 || ep_size <= 0 || num_experts <= 0)) {
      std::cerr << "Error: --moe_mode requires positive --topk/--ep_size/"
                   "--num_experts.\n";
      error = true;
    }

    build_rows();

    // If the BD experts_token_offset was supplied, validate it is exactly the
    // running prefix-sum of the per-expert counts (the reference computes
    // cur_token_start = experts_token_offset[i]; we derive the same internally,
    // so this just guards a mismatched hand-written .in line).
    if (!experts_token_offset.empty()) {
      std::vector<int> off;
      std::stringstream ss(experts_token_offset);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        if (tok.empty())
          continue;
        off.push_back(std::stoi(tok));
      }
      if (off.size() != rows_per_expert.size()) {
        std::cerr << "Error: experts_token_offset has " << off.size()
                  << " entries but experts_token_count has "
                  << rows_per_expert.size() << ".\n";
        error = true;
      } else {
        int running = 0;
        for (size_t i = 0; i < off.size(); ++i) {
          if (off[i] != running) {
            std::cerr << "Error: experts_token_offset[" << i << "]=" << off[i]
                      << " != prefix-sum " << running
                      << " of experts_token_count.\n";
            error = true;
            break;
          }
          running += rows_per_expert[i];
        }
      }
    }
  }

  int total_m() const {
    int t = 0;
    for (int m : rows_per_expert)
      t += m;
    return t;
  }

  /// Compute performance in TFLOP/s over all per-expert problems.
  double tflops(double runtime_s) const {
    uint64_t fmas = 0;
    for (int m : rows_per_expert) {
      fmas += static_cast<uint64_t>(m) * static_cast<uint64_t>(n) *
              static_cast<uint64_t>(k);
    }
    uint64_t flop = static_cast<uint64_t>(2) * fmas;
    double tflop = double(flop) / double(1.0e12);
    return tflop / runtime_s;
  }

  std::string benchmark_name() const {
    std::stringstream full_name;
    full_name << bm_name << "/" << std::to_string(n) << "x"
              << std::to_string(k) << "x" << std::to_string(num_experts) << "x"
              << std::to_string(total_m());
    return full_name.str();
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Runner for the one config this binary compiles. All device work — A/B/D +
// scale allocation, scheduler/grid/mma setup, the MoE::MoEGEMM parallel_for, and
// the GPU_Clock timing — lives in the LEAN moe_kernel_launch.cpp, reached via
// moe_setup / moe_launch_once / moe_teardown. This runner only drives the
// Google-Benchmark loop + counters; it instantiates NO cute / MoE / SYCL device
// code, and is NOT templated on Config (the config is baked into the .cpp at
// build time via -DMOE_BENCH_CONFIG), so this TU never names a cute Config type.
// Perf only — no verification.
struct MoEBenchmarkRunner {

  void run(::benchmark::State &state, MoEBenchmarkOptions const &options,
           cutlass::KernelHardwareInfo const & /*hw_info*/,
           const char *config_name) {
    const int num_experts = options.num_experts;
    const int N = options.n;
    const int K = options.k;

    std::vector<int> M_per_expert = options.rows_per_expert;
    if (static_cast<int>(M_per_expert.size()) != num_experts) {
      state.SkipWithError("num_experts does not match per-expert M list size.");
      return;
    }
    int num_tokens = options.total_m();

    // Map the verify toggle to the TU-boundary int (kVerifyNone/kVerifyOn).
    int verify_kind =
        options.verify ? moe_bench::kVerifyOn : moe_bench::kVerifyNone;

    // Allocate + warm up the kernel ONCE in the lean TU. The returned handle is
    // opaque; this TU never sees the kernel type. If verify_kind != none, the
    // lean TU also runs the example's VerificationHelper against this shape.
    std::string error;
    moe_bench::MoeRunHandle *handle =
        moe_bench::moe_setup_by_name(config_name, N, K, num_experts,
                                     M_per_expert, &error, verify_kind);
    if (!handle) {
      state.SkipWithError(error.empty() ? "moe_setup failed" : error.c_str());
      return;
    }

    // FLOP count over all per-expert problems.
    uint64_t fmas = 0;
    for (int m : M_per_expert) {
      fmas += static_cast<uint64_t>(m) * static_cast<uint64_t>(N) *
              static_cast<uint64_t>(K);
    }
    double gflop = double(static_cast<uint64_t>(2) * fmas) / double(1.0e9);

    state.counters["n"] = N;
    state.counters["k"] = K;
    state.counters["num_experts"] = num_experts;
    state.counters["total_m"] = num_tokens;

    initialize_counters(state);
    for (auto _ : state) {
      // Timed body lives entirely in the lean .cpp; returns elapsed ms.
      double ms_elapsed = moe_bench::moe_launch_once(handle);
      update_counters(state, ms_elapsed);
      state.SetIterationTime(ms_elapsed / 1000);
    }
    finalize_counters(state, gflop);

    moe_bench::moe_teardown(handle);
  }

private:
  static void initialize_counters(::benchmark::State &state) {
    state.counters["avg_runtime_ms"] = 0;
    state.counters["best_runtime_ms"] = std::numeric_limits<double>::max();
    state.counters["worst_runtime_ms"] = std::numeric_limits<double>::lowest();
    state.counters["total_runtime_ms"] = 0;
  }

  static void update_counters(::benchmark::State &state, double ms_elapsed) {
    state.PauseTiming();
    state.counters["total_runtime_ms"] += ms_elapsed;
    state.counters["best_runtime_ms"] =
        std::min<double>(state.counters["best_runtime_ms"], ms_elapsed);
    state.counters["worst_runtime_ms"] =
        std::max<double>(state.counters["worst_runtime_ms"], ms_elapsed);
    state.ResumeTiming();
  }

  static void finalize_counters(::benchmark::State &state, double gflop) {
    auto iters = static_cast<double>(state.iterations());
    if (iters > 2) {
      state.counters["avg_runtime_ms"] =
          (state.counters["total_runtime_ms"] -
           state.counters["best_runtime_ms"] -
           state.counters["worst_runtime_ms"]) /
          (iters - 2);
    } else {
      state.counters["avg_runtime_ms"] =
          state.counters["total_runtime_ms"] / iters;
    }
    state.counters["avg_tflops"] = gflop / state.counters["avg_runtime_ms"];
    state.counters["best_tflop"] = gflop / state.counters["best_runtime_ms"];
  }
};

} // namespace cutlass::benchmark

///////////////////////////////////////////////////////////////////////////////////////////////////

// Two-level indirection so a MACRO argument (MOE_BENCH_NAME) is expanded BEFORE
// the # / ## operators act on it. Using # / ## on the bare parameter would
// stringize/paste the literal token "MOE_BENCH_NAME" instead of its value
// (e.g. MoE_BF16) — which silently registers the benchmark under the wrong name
// and yields "Benchmark not found" at run time.
#define CUTLASS_MOE_STR_IMPL(s) #s
#define CUTLASS_MOE_STR(s) CUTLASS_MOE_STR_IMPL(s)
#define CUTLASS_MOE_CAT_IMPL(a, b) a##b
#define CUTLASS_MOE_CAT(a, b) CUTLASS_MOE_CAT_IMPL(a, b)

#define CUTLASS_GROUPED_GEMM_BENCHMARK(Name)                                   \
  cutlass::benchmark::BenchmarkRegistry<                                       \
      cutlass::benchmark::MoEBenchmarkOptions>::Register(                      \
      CUTLASS_MOE_STR(Name), &CUTLASS_MOE_CAT(Name, _func))

// The runner is no longer templated on Config (the config is baked into
// moe_kernel_launch.cpp via -DMOE_BENCH_CONFIG), so this macro just builds the
// registration thunk under the requested name.
#define CUTLASS_CREATE_GROUPED_GEMM_BENCHMARK(Name)                            \
  static void CUTLASS_MOE_CAT(Name, _func)(                                    \
      ::benchmark::State &state,                                               \
      cutlass::benchmark::MoEBenchmarkOptions const &options,                  \
      cutlass::KernelHardwareInfo const &hw_info) {                            \
    auto bench = cutlass::benchmark::MoEBenchmarkRunner();                     \
    bench.run(state, options, hw_info, CUTLASS_MOE_STR(Name));                 \
  }

///////////////////////////////////////////////////////////////////////////////////////////////////

// ONE config compiled per binary, selected by -DMOE_BENCH_CONFIG=<name> at
// build time (consumed in moe_kernel_launch.cpp). Compiling all configs into a
// single device image triggers an IGC internal compiler error on CRI, so — like
// example 12, which is one .cpp/target per dtype — each benchmark executable
// holds a single dtype. The registered benchmark name (the config-file token)
// is fixed below per binary so the .in files stay stable regardless of which
// binary is built. This benchmark TU only needs MOE_BENCH_NAME.
// TILE SWEEP: this binary holds ALL tiles for its dtype (selected by
// -DMOE_DTYPE_<TAG>). moe_tile_list.hpp expands to X(NAME,CONFIG) entries; we
// create a registration thunk per NAME and register them all. The .in line's
// first token selects which tile runs (grouped_gemm-style). moe_setup_by_name
// in the .cpp maps the same NAME back to the right cute config.
#include "moe_tile_list.hpp"

#ifdef MOE_TILE_X_LIST
#define X(NAME, CONFIG) CUTLASS_CREATE_GROUPED_GEMM_BENCHMARK(NAME)
MOE_TILE_X_LIST
#undef X
#endif

static void register_grouped_gemm_benchmarks() {
#ifdef MOE_TILE_X_LIST
#define X(NAME, CONFIG) CUTLASS_GROUPED_GEMM_BENCHMARK(NAME);
  MOE_TILE_X_LIST
#undef X
#endif
}
