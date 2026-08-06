/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*!
  \file xe35_gdn_attention_perf.hpp
  \brief Timed-launch helper for the example runner, plus a re-export of the
         shared verification comparator.

  The `compare_with_stats` / `print_compare_stats` comparator now lives in
  cutlass/util/reference/host/xe35_gdn_attention_compare.hpp so the benchmark
  runner can reuse it too; it is pulled in below and stays in the same
  `cutlass::gdn::perf` namespace, so existing call sites are unchanged. This
  header keeps only `time_launches`, which is specific to the example's
  single-shape timing path.
*/

#pragma once

#include "sycl_common.hpp"

#include "cutlass/util/reference/host/xe35_gdn_attention_compare.hpp"
#include "cutlass/util/GPU_Clock.hpp"

namespace cutlass::gdn::perf {

// ---------------------------------------------------------------------------
// Timed launch helper.
//
// Usage:
//   auto t_ms = time_launches(queue, options.iterations, options.warmup, [&]{
//     submit_my_kernel(queue, ...);
//   });
//
// Performs an optional warmup pass, then `iterations` timed launches measured
// with cutlass::GPU_Clock (the repo-wide timing utility, as in the GEMM and
// flash-attention examples). Correct on both build configurations:
//
//   * CUTLASS_SYCL_PROFILING_ENABLED ON: GPU_Clock/SYCLTimer sums each kernel's
//     device-side command_start/command_end span. kernel_launcher registers its
//     5 stage events with EventManager, and the GDN queue is created with
//     enable_profiling(), so `timer.start()` records the event index, the timed
//     loop pushes 5*iterations events, and `timer.milliseconds()` sums exactly
//     those -- the host-side drains/waits below are then redundant but harmless.
//
//   * Profiling OFF: SYCLTimer falls back to a host wall-clock span bracketed by
//     `compat::get_default_queue().wait()` at both ends. The GDN kernel runs on
//     a private in-order queue, not the default queue, so we drain the default
//     queue before `timer.start()` (making that wait a no-op) and call
//     `queue.wait_and_throw()` on the private queue before `timer.milliseconds()`
//     so the measured span covers exactly the private-queue work.
// ---------------------------------------------------------------------------

template <typename Launcher>
inline double time_launches(sycl::queue& queue, int iterations,
                            int warmup, Launcher&& launcher) {
  if (iterations <= 0) return 0.0;
  for (int i = 0; i < warmup; ++i) launcher();
  queue.wait_and_throw();

  // Drain the default queue so SYCLTimer's internal default-queue waits are
  // no-ops and the measured span reflects only the private-queue launches.
  compat::get_default_queue().wait();

  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < iterations; ++i) launcher();
  queue.wait_and_throw();
  return double(timer.milliseconds()) / iterations;  // ms / iter
}

}  // namespace cutlass::gdn::perf
