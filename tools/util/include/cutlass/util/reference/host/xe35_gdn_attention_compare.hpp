/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

/*!
  \file xe35_gdn_attention_compare.hpp
  \brief Host-side verification comparator shared by the GDN attention
         consumers (example runner and benchmark runner).

  `compare_with_stats` is an `is_close` comparator that also counts NaN / +-Inf
  in both the reference and device buffers, so a failing verify immediately
  tells you "is the kernel producing garbage" vs "is the reference producing
  garbage" vs "small numerical disagreement". `print_compare_stats` renders the
  result, including bucketed mismatch counts at increasingly loose tolerances.

  Consumers
  ---------
    - examples/14_xe35_gdn_attention/xe35_gdn_attention_perf.hpp (re-exports these)
    - benchmarks/applications/03_gdn/benchmark_runner.hpp (optional --verify path)
    - test/unit/gdn_attention/gdn_chunkwise_testbed.hpp

  This header is self-contained: it carries its own `is_close` so it does not
  depend on the examples-only sycl_common.hpp (the benchmark does not include
  that). It pulls in no device code, matching the host-only contract of the
  reference headers it sits beside.
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>

namespace cutlass::gdn::perf {

// Absolute+relative closeness test: |a - b| <= atol + rtol*|b|. A namespace-
// local copy of the examples/common sycl_common.hpp helper so this header is
// usable from translation units (e.g. the benchmark) that do not include it.
template <typename T>
inline bool is_close(T a, T b, float atol, float rtol) {
  return std::abs(static_cast<float>(a) - static_cast<float>(b))
         <= atol + rtol * std::abs(static_cast<float>(b));
}

// ---------------------------------------------------------------------------
// Comparison with NaN / Inf counts and bucketed tolerances.
// ---------------------------------------------------------------------------

struct compare_stats_t {
  size_t total = 0;
  size_t mismatches = 0;   // !is_close at the primary (atol, rtol)
  size_t first_bad  = 0;
  float  first_bad_ref = 0.0f;
  float  first_bad_dev = 0.0f;
  float  max_abs_diff  = 0.0f;
  // Hardening counters: anything NaN/+-Inf in either side is a hard failure.
  size_t nan_dev = 0, inf_dev = 0;
  size_t nan_ref = 0, inf_ref = 0;
  // Bucketed mismatch counts at increasingly loose tolerances; useful for
  // diagnosing "are we off by 5%, 10%, or hopelessly?".
  size_t bad_5e2 = 0, bad_1e1 = 0, bad_2e1 = 0, bad_5e1 = 0;

  // Per-element pass (like FMHA's BlockCompareRelativelyEqual): every element
  // within (atol, rtol), no NaN/Inf on either side.
  bool passed() const {
    return mismatches == 0 && nan_dev == 0 && inf_dev == 0
                           && nan_ref == 0 && inf_ref == 0;
  }
};

// Element-wise compare of a reference buffer against a device buffer.
template <typename RefVec, typename DevVec>
inline compare_stats_t compare_with_stats(
    const RefVec& ref, const DevVec& dev,
    float atol, float rtol)
{
  compare_stats_t s{};
  s.total = ref.size();
  if (ref.size() != dev.size()) {
    // Treat size mismatch as a hard failure -- inflate counters so the caller
    // notices. Use the max of the two sizes: keying off ref.size() alone would
    // leave total==mismatches==0 (a false "pass") when ref is empty but dev is
    // not.
    const size_t n = std::max(ref.size(), dev.size());
    s.total = n;
    s.mismatches = n;
    return s;
  }
  bool first_bad_set = false;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float r = static_cast<float>(ref[i]);
    const float d = static_cast<float>(dev[i]);
    if (std::isnan(d)) ++s.nan_dev;
    else if (std::isinf(d)) ++s.inf_dev;
    if (std::isnan(r)) ++s.nan_ref;
    else if (std::isinf(r)) ++s.inf_ref;
    const float ad = std::fabs(r - d);
    if (std::isfinite(ad)) s.max_abs_diff = std::max(s.max_abs_diff, ad);
    if (!is_close(d, r, atol, rtol)) {
      if (!first_bad_set) {
        s.first_bad = i; s.first_bad_ref = r; s.first_bad_dev = d;
        first_bad_set = true;
      }
      ++s.mismatches;
    }
    if (!is_close(d, r, 5e-2f, 5e-2f)) ++s.bad_5e2;
    if (!is_close(d, r, 1e-1f, 1e-1f)) ++s.bad_1e1;
    if (!is_close(d, r, 2e-1f, 2e-1f)) ++s.bad_2e1;
    if (!is_close(d, r, 5e-1f, 5e-1f)) ++s.bad_5e1;
  }
  return s;
}

// One histogram row: the fraction of elements WITHIN a tolerance bucket
// (= total - bad), as a proportional bar. `bad` is the count outside it.
inline void print_bucket_bar(const char* label, size_t bad, size_t total) {
  constexpr int kBarWidth = 30;
  const size_t within = (total >= bad) ? (total - bad) : 0;
  const double frac = total ? double(within) / double(total) : 0.0;
  int fill = static_cast<int>(frac * kBarWidth + 0.5);
  std::cout << "            within " << label << " |";
  for (int i = 0; i < kBarWidth; ++i) std::cout << (i < fill ? '#' : '.');
  std::cout << "| " << within << "/" << total
            << "  (" << (frac * 100.0) << "%)\n";
}

// Verify report: leads with max_abs_diff + per-element mismatch count (the pass
// criterion), then a bucketed histogram of how the error is distributed.
inline void print_compare_stats(const char* tag, const compare_stats_t& s) {
  std::cout << "[verify] " << tag
            << "  max_abs_diff="          << s.max_abs_diff
            << "  mismatches="            << s.mismatches << "/" << s.total << "\n";
  print_bucket_bar(" 5%", s.bad_5e2, s.total);
  print_bucket_bar("10%", s.bad_1e1, s.total);
  print_bucket_bar("20%", s.bad_2e1, s.total);
  print_bucket_bar("50%", s.bad_5e1, s.total);
  // NaN/Inf are always fatal; show only when present.
  if (s.nan_dev || s.inf_dev || s.nan_ref || s.inf_ref) {
    std::cout << "            !! nan_dev=" << s.nan_dev
              << " inf_dev="              << s.inf_dev
              << " nan_ref="              << s.nan_ref
              << " inf_ref="              << s.inf_ref << "\n";
  }
  // First failing element, for a quick peek.
  if (s.mismatches) {
    std::cout << "            first bad idx=" << s.first_bad
              << " ref=" << s.first_bad_ref
              << " dev=" << s.first_bad_dev << "\n";
  }
}

}  // namespace cutlass::gdn::perf
