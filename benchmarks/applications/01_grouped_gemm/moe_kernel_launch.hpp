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
    \brief THIN, device-code-free interface between the Google-Benchmark harness
           (moe_benchmark_runner.hpp / main.cpp) and the MoE grouped-GEMM kernel
           launch.

    The whole point of this header is isolation of translation units: the
    benchmark TU (which compiles <benchmark/benchmark.h> + benchmarks/common.hpp,
    pulling in oneMKL / std::function / the BenchmarkRegistry) must NOT also
    instantiate the MoE::MoEGEMM cute device kernel. Doing both in one SPIR-V
    module ICEs IGC during AOT device codegen on CRI (ocloc -device cri).

    So this header exposes only:
      - an OPAQUE handle (moe_bench::MoeRunHandle, defined entirely inside
        moe_kernel_launch.cpp where the cute/MoE/SYCL kernel types are known), and
      - three plain free functions (moe_setup / moe_launch_once / moe_teardown).

    The benchmark TU includes ONLY this header and never sees cute, MoE, SYCL, or
    the kernel name tag. All heavy template instantiation (and therefore the
    device SPIR-V) lives exclusively in moe_kernel_launch.cpp.

    moe_setup is NON-template across this TU boundary: the config is baked in at
    build time via -DMOE_BENCH_CONFIG, so the benchmark runner TU never names a
    cute/MoE Config type. Internally moe_kernel_launch.cpp forwards to the
    Config-templated moe_setup_impl<MOE_BENCH_CONFIG>; no other config is ever
    referenced. The returned handle is type-erased, so moe_launch_once /
    moe_teardown are likewise plain (non-template) functions the benchmark TU can
    call without knowing the config.
*/

#pragma once

#include <string>
#include <vector>

namespace moe_bench {

// Verification toggle, kept as a plain int across the TU boundary (the
// benchmark runner TU must not depend on the example header where the real
// VerificationHelper lives). 0=off, 1=on. The actual verify runs INSIDE
// moe_kernel_launch.cpp (the heavy TU that already includes the example header +
// VerificationHelper).
enum VerifyKind { kVerifyNone = 0, kVerifyOn = 1 };

// Opaque handle. Fully defined only inside moe_kernel_launch.cpp; the benchmark
// TU manipulates it solely through the pointer + the free functions below.
struct MoeRunHandle;

// Allocate device buffers (A/B/D + padded scaleA/scaleB for scaled configs),
// build the tile-scheduler params / grid / TiledMMA, and run a single warmup
// launch. If verify != kVerifyNone, run the example's VerificationHelper once
// against this shape (inside the lean .cpp) and report pass/fail to stderr.
// Returns a heap-allocated handle (nullptr on failure, with *error set to a
// human-readable message).
//
// NON-template across the TU boundary: which Config to use is baked in at build
// time via -DMOE_BENCH_CONFIG (each binary compiles exactly one). The .cpp's
// definition resolves to MOE_BENCH_CONFIG, so the benchmark TU never needs to
// name a cute/MoE Config type at all — keeping it free of the example header.
// (Removed: moe_setup() had no definition; use moe_setup_by_name instead.)

// Launch the kernel ONCE and block until it finishes. Returns the elapsed time
// in milliseconds (GPU_Clock timed INSIDE the lean .cpp), so the benchmark TU
// stays free of any SYCL / cute timing machinery.
// TILE SWEEP: pick the (dtype,tile) config by its registered string name (the
// .in line's first token). Returns nullptr + sets *error if the name is unknown
// to this binary. cute types stay inside the .cpp.
MoeRunHandle *moe_setup_by_name(const char *name, int N, int K, int num_experts,
                                std::vector<int> const &M_per_expert,
                                std::string *error, int verify = kVerifyNone);

double moe_launch_once(MoeRunHandle *handle);

// Release all device allocations owned by the handle.
void moe_teardown(MoeRunHandle *handle);

} // namespace moe_bench
