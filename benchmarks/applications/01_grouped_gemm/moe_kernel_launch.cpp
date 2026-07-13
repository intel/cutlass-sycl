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
    \brief Device-codegen translation unit for the MoE grouped-GEMM benchmark:
           the ONLY TU that instantiates the MoE::MoEGEMM kernel (and so the only
           emitter of device SPIR-V). It includes the runner machinery + this
           file's thin interface, but NOT benchmark/oneMKL/common.hpp. See
           moe_kernel_launch.hpp for the TU-split rationale (avoids the IGC ICE).

    Allocation, scale setup, warmup, and the timed submission live here; the
    timed core (moe_launch_timed) and config/scale/verify machinery come from
    moe_grouped_gemm/runner/moe_gemm_runner.hpp. Everything is wrapped behind an
    opaque handle so the benchmark TU never sees cute / MoE / SYCL types.
*/

#include "moe_kernel_launch.hpp"

// The shared runner pulls in everything: cute, MoE::MoEGEMM, the persistent
// tile scheduler, choose_tiled_mma, fill_scale, ScaleKind, and the Config
// structs (Bf16Config / MxFp8E4m3Config / ...). It also includes GPU_Clock and
// the SYCL event manager, and consumes the kernel API under
// applications/moe_grouped_gemm/ directly (no dependency on examples/). NO
// benchmark / oneMKL headers here. Reached via ${CUTLASS_DIR}/applications on
// the include path.
#include "moe_grouped_gemm/runner/moe_gemm_runner.hpp"

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <stdexcept>
// The shared runner lives in namespace cutlass::moe; pull it in unqualified so
// the existing references (Config structs, moe_launch_timed, VerificationHelper,
// fill_scale, ScaleKind, ...) resolve without per-symbol qualification.
using namespace cutlass::moe;

// ONE config compiled per binary, selected by -DMOE_BENCH_CONFIG=<name> at
// build time (matches the runner header default). Bf16Config / MxFp8E4m3Config /
// ... are defined in the shared runner included above.
#ifndef MOE_BENCH_CONFIG
#define MOE_BENCH_CONFIG Bf16Config
#endif

namespace moe_bench {

// NOTE: the SYCL kernel-name tag is owned by the shared core (GemmCuteName)
// now that this TU delegates the launch to moe_launch_timed. Each benchmark
// binary compiles exactly one Config (-DMOE_BENCH_CONFIG, the IGC-ICE
// one-config-per-binary split), so there is a single kernel per binary and no
// cross-config name aliasing to guard against.

// Pull a config's scale-storage element type (only meaningful for scaled
// configs; Plain stays float and the pointers stay null).
template <class C, class = void> struct ConfigScale {
  using type = float;
};
template <class C>
struct ConfigScale<C, cute::void_t<typename C::ElementScale>> {
  using type = std::conditional_t<std::is_void_v<typename C::ElementScale>,
                                  float, typename C::ElementScale>;
};

// The opaque handle, fully defined here where the kernel types are known. It
// owns all device allocations (so they stay alive across the whole timed loop)
// and a type-erased `launch` closure that does ONE GPU_Clock-timed submission +
// wait and returns elapsed milliseconds. The benchmark TU only ever holds a
// MoeRunHandle* and calls the three free functions.
struct MoeRunHandle {
  // Type-erased owners for the device allocations. Each config has different
  // element types, so we hide them behind a vector of generic deleters captured
  // by the setup closure; simplest is to keep them alive inside the launch
  // closure's capture. To make teardown explicit we also keep a `release`
  // closure.
  std::function<double()> launch; // timed single launch -> elapsed ms
  std::function<void()> release;  // frees device allocations
};

} // namespace moe_bench

namespace moe_bench {

// Config-templated implementation. The public non-template moe_setup (below)
// dispatches to this for the single MOE_BENCH_CONFIG this binary compiles, so
// the cute/MoE kernel is instantiated exactly once, only in this TU.
template <class Config>
MoeRunHandle *moe_setup_impl(int N, int K, int num_experts,
                             std::vector<int> const &M_per_expert,
                             std::string *error, int verify) {
  using ElementInput = typename Config::Element;
  // Output (D) element type is declared per Config (Config::ElementOutput);
  // all current paths emit bf16. The 16-bit XE_STORE_2D store atom is element-agnostic.
  using ElementOutput = typename Config::ElementOutput;

  const uint64_t seed = 2023;

  if (static_cast<int>(M_per_expert.size()) != num_experts) {
    if (error)
      *error = "num_experts does not match per-expert M list size.";
    return nullptr;
  }

  int num_tokens = 0;
  for (int m : M_per_expert)
    num_tokens += m;

  // Shared device state held alive by the handle. Allocated on the heap so it
  // outlives this function and survives until moe_teardown.
  struct DeviceState {
    cutlass::DeviceAllocation<int32_t> num_rows_per_expert_device;
    cutlass::DeviceAllocation<ElementInput> activations_data;
    cutlass::DeviceAllocation<ElementInput> weights_data;
    cutlass::DeviceAllocation<ElementOutput> output_data;
    using ElementScaleStore =
        std::conditional_t<Config::scale_kind == ScaleKind::Plain, float,
                           typename ConfigScale<Config>::type>;
    cutlass::DeviceAllocation<ElementScaleStore> scaleA_data;
    cutlass::DeviceAllocation<ElementScaleStore> scaleB_data;
  };
  using ElementScaleStore = typename DeviceState::ElementScaleStore;

  auto *st = new DeviceState();

  try {
    // ---- Per-expert M table (host -> device) ----
    st->num_rows_per_expert_device.reset(num_experts);
    st->num_rows_per_expert_device.copy_from_host(M_per_expert.data());

    // ---- A / B / D allocation (row-major packed, matches launcher<Config>) ----
    int64_t A_size = int64_t(num_tokens) * K;
    int64_t B_size = int64_t(num_experts) * N * K;
    int64_t D_size = int64_t(num_tokens) * N;
    st->activations_data.reset(A_size);
    st->weights_data.reset(B_size);
    st->output_data.reset(D_size);
    initialize_block(st->activations_data, seed + 2023);
    initialize_block(st->weights_data, seed + 2022);
    initialize_block(st->output_data, seed + 2021);
  } catch (std::exception const &e) {
    if (error)
      *error = e.what();
    delete st;
    return nullptr;
  }

  // ---- Scale allocation (padded MX layout) — identical to launcher<Config> ----
  // For Plain (BF16) the scale pointers stay null and group_n/group_k = 0.
  int group_n = 0;
  int group_k = 0;
  const ElementScaleStore *ptr_sA = nullptr;
  const ElementScaleStore *ptr_sB = nullptr;

  if constexpr (Config::scale_kind != ScaleKind::Plain) {
    constexpr bool is_tensor = (Config::scale_kind == ScaleKind::Tensor);
    group_k = is_tensor ? K : Config::group_k;
    group_n = is_tensor ? N : Config::group_n;
    const int scale_k = (K + group_k - 1) / group_k;
    const int scale_n = (N + group_n - 1) / group_n;

    int64_t scaleA_size, scaleB_size;
    if constexpr (is_tensor) {
      scaleA_size = int64_t(num_tokens) * scale_k;
      scaleB_size = int64_t(num_experts) * scale_n * scale_k;
    } else {
      // Block (MX): per-expert rows padded to 64 so the 2D scale load surface
      // is aligned (must match launcher<Config> / the kernel mainloop).
      constexpr int kScaleAlign = 64;
      int64_t padded_M_total = 0;
      for (int i = 0; i < num_experts; i++) {
        padded_M_total +=
            ((M_per_expert[i] + kScaleAlign - 1) / kScaleAlign) *
            int64_t(kScaleAlign);
      }
      const int padded_scale_n =
          ((scale_n + kScaleAlign - 1) / kScaleAlign) * kScaleAlign;
      scaleA_size = padded_M_total * scale_k;
      scaleB_size = int64_t(num_experts) * padded_scale_n * scale_k;
    }
    try {
      st->scaleA_data.reset(scaleA_size);
      st->scaleB_data.reset(scaleB_size);
    } catch (std::exception const &e) {
      if (error)
        *error = e.what();
      delete st;
      return nullptr;
    }
    fill_scale(st->scaleA_data, seed + 2020, is_tensor);
    fill_scale(st->scaleB_data, seed + 2019, is_tensor);
    ptr_sA = st->scaleA_data.get();
    ptr_sB = st->scaleB_data.get();
  }

  // Raw device pointers — the DeviceAllocation wrappers are not device-copyable,
  // so only plain pointers cross into the timed launch.
  const ElementInput *ptr_A = st->activations_data.get();
  const ElementInput *ptr_B = st->weights_data.get();
  ElementOutput *ptr_D = st->output_data.get();
  const int32_t *ptr_rows = st->num_rows_per_expert_device.get();
  const ElementScaleStore *cap_sA = ptr_sA;
  const ElementScaleStore *cap_sB = ptr_sB;
  const int cap_group_n = group_n;
  const int cap_group_k = group_k;

  // The single timed launch closure — the one source of truth for the
  // submission geometry + parallel_for body, delegating to the shared core
  // moe_launch_timed. Warmup, the optional verify pre-run, and the benchmark
  // loop all reuse this exact closure, so the launch can never drift and setup
  // never pays for a duplicate launch. This TU stays the only one that
  // instantiates the device kernel (the IGC-ICE lean-TU isolation).
  //
  // The Plain (BF16) path passes typed-null void* scale pointers so ElementS
  // deduces to void and the kernel takes the non-scaled path; the scaled path
  // passes the real typed scale pointers.
  auto launch = [=]() -> double {
    if constexpr (Config::scale_kind == ScaleKind::Plain) {
      return moe_launch_timed<'R', 'R', Config>(
          ptr_A, ptr_B, static_cast<const void *>(nullptr),
          static_cast<const void *>(nullptr), ptr_D, N, K, ptr_rows,
          num_experts, cap_group_n, cap_group_k);
    } else {
      return moe_launch_timed<'R', 'R', Config>(
          ptr_A, ptr_B, cap_sA, cap_sB, ptr_D, N, K, ptr_rows, num_experts,
          cap_group_n, cap_group_k);
    }
  };

  // Optional correctness check: when enabled, run the kernel once via the shared
  // launch closure, then the VerificationHelper against this exact shape — no
  // separate reference implementation. Off by default (perf runs). The Plain
  // path routes to verify(), the scaled paths to verify_scaled(). Kept inside
  // this heavy TU (which already includes VerificationHelper) so the benchmark
  // runner TU stays free of cute/MoE.
  if (verify != kVerifyNone) {
    (void)launch();
    VerificationHelper helper;
    helper.parse(num_experts, M_per_expert.data(), N, K);
    bool ok = true;
    if constexpr (Config::scale_kind == ScaleKind::Plain) {
      ok = helper.verify(ptr_A, ptr_B, ptr_D);
    } else {
      constexpr bool kIsTensor = (Config::scale_kind == ScaleKind::Tensor);
      sycl::queue Q = compat::get_default_queue();
      ok = helper.template verify_scaled<kIsTensor>(
          Q, ptr_A, ptr_B, ptr_sA, ptr_sB, ptr_D, group_n, group_k);
    }
    std::cerr << "[MoE bench verify] " << (ok ? "passed" : "FAILED")
              << std::endl;
    if (!ok) {
      if (error)
        // Match the grouped_gemm benchmark marker so run-harvesting scripts that
        // grep for a failed disposition treat a MoE verify mismatch identically.
        *error = "Disposition Failed.";
      delete st;
      return nullptr;  // Return error to trigger state.SkipWithError()
    }
  }

#ifndef CUTLASS_TEST_FOR_CRI
  // Warmup (perf only). Disabled on the CRI simulator since it is time-consuming.
  (void)launch();
#endif

  auto *handle = new MoeRunHandle();
  handle->launch = launch;
  // Release the device allocations. The launch closure only holds raw pointers
  // into st, so freeing st after the benchmark loop is correct.
  handle->release = [st]() { delete st; };
  return handle;
}

// (legacy single-config moe_setup removed: the tile-sweep registry
// moe_setup_by_name supersedes it; keeping it would duplicate a kernel
// instantiation and collide on the GemmCuteName mangled name.)


// ---- TILE SWEEP: name -> setup-fn registry (all cute types stay in this TU) ----
// Each (dtype,tile) config registers moe_setup_impl<Config> under its string
// name. moe_setup_by_name() looks it up so the benchmark TU can pick a tile at
// runtime from the .in line's first token, with NO cute type crossing the TU
// boundary. Multiple configs in ONE binary is now safe because GemmCuteName
// encodes the tile (unique SYCL kernel name per tile).
namespace {
using SetupFn = MoeRunHandle *(*)(int, int, int, std::vector<int> const &,
                                  std::string *, int);
std::map<std::string, SetupFn> &moe_setup_registry() {
  static std::map<std::string, SetupFn> r;
  return r;
}
struct MoeSetupRegistrar {
  MoeSetupRegistrar(const char *name, SetupFn fn) {
    moe_setup_registry().emplace(name, fn);
  }
};
} // namespace

// Register one (name, Config). Used by the per-binary tile list below.
#define MOE_REGISTER_TILE(NAME, CONFIG)                                          static MoeRunHandle *moe_setup_##NAME(                                             int N, int K, int ne, std::vector<int> const &M, std::string *e,               int v) {                                                                     return moe_setup_impl<CONFIG>(N, K, ne, M, e, v);                            }                                                                              static MoeSetupRegistrar moe_reg_##NAME(#NAME, &moe_setup_##NAME);

// The per-binary tile list comes from moe_tile_list.hpp, selected by the
// -DMOE_DTYPE_<TAG> this binary is built with. Expand each X(NAME,CONFIG) into a
// MOE_REGISTER_TILE registration so every tile this binary holds is in the
// name->setup table.
#include "moe_tile_list.hpp"
#ifdef MOE_TILE_X_LIST
#define X(NAME, CONFIG) MOE_REGISTER_TILE(NAME, CONFIG)
MOE_TILE_X_LIST
#undef X
#endif

MoeRunHandle *moe_setup_by_name(const char *name, int N, int K, int num_experts,
                                std::vector<int> const &M_per_expert,
                                std::string *error, int verify) {
  auto &r = moe_setup_registry();
  auto it = r.find(name);
  if (it == r.end()) {
    if (error)
      *error = std::string("no registered tile config named '") + name + "'";
    return nullptr;
  }
  return it->second(N, K, num_experts, M_per_expert, error, verify);
}

double moe_launch_once(MoeRunHandle *handle) { return handle->launch(); }

void moe_teardown(MoeRunHandle *handle) {
  if (!handle)
    return;
  handle->release();
  delete handle;
}

} // namespace moe_bench
