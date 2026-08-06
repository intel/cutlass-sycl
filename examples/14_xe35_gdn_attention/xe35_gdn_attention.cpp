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
/*! \file
    \brief Baseline naive Chunkwise Gated DeltaNet (GDN) attention
           example for Intel Xe GPUs.

    Driver for the baseline naive Xe35 implementation of the chunkwise GDN
    attention kernels. The host harness, options, initialization, host
    reference, and verification all live in the associated
    `xe35_gdn_attention_runner.hpp`.

    The SSM recurrent state (StateT) is fixed at fp32 to avoid drift
    across long sequences.  The activation dtype T (Q/K/V/O and dt_bias)
    is bfloat16_t.  One executable is produced:

      14_xe35_gdn_attention_bfloat16 -- bf16 activations, fp32 SSM state

    Build & run (from your build dir):

      $ ninja 14_xe35_gdn_attention_bfloat16
      $ ./examples/14_xe35_gdn_attention/14_xe35_gdn_attention_bfloat16 --help
*/

#include "xe35_gdn_attention_runner.hpp"

#include "cutlass/bfloat16.h"

#ifndef ACT_TYPE
#define ACT_TYPE cutlass::bfloat16_t
#endif

/* ---------------------------------------------------------------------------
 * Per-case launcher
 *
 * Mirrors the pattern used by examples/12_xe20_moe_gemm_cute_interface: a
 * thin wrapper that materialises one Options struct from a shape tuple and
 * the shared CLI-driven knobs (iterations / warmup / verify / seed), then
 * dispatches a single GdnRunner.run() invocation.
 * --------------------------------------------------------------------------- */
template <typename T, typename StateT>
int launcher(int batch, int num_v_heads, int num_k_heads, int seq_len,
             Options const& base) {
  Options opt = base;
  opt.batch       = batch;
  opt.num_v_heads = num_v_heads;
  opt.num_k_heads = num_k_heads;
  opt.seq_len     = seq_len;
  /* head_k_dim / head_v_dim keep the values already in `base` — their
   * HEAD_K_DIM / HEAD_V_DIM compile-time defaults unless overridden on the CLI
   * via --head_k_dim / --head_v_dim. Every entry in the case table and in
   * input_gdn_bf16.in uses 128/128. */

  std::cout << "\n========================================================\n"
            << " GDN case: batch=" << batch
            << " num_v_heads=" << num_v_heads
            << " num_k_heads=" << num_k_heads
            << " seq_len=" << seq_len
            << "\n========================================================\n";

  GdnRunner<T, StateT> runner(opt);
  return runner.run();
}

int main(int argc, const char** argv) {
  /* CLI parses the shared knobs (iterations/warmup/verify/seed/help/error)
   * AND the per-case shape fields (batch/num_v_heads/num_k_heads/seq_len).
   * If any shape flag is supplied, we run a single CLI-driven case below;
   * otherwise we iterate the hard-coded sweep table. */
  Options base_options;
  base_options.parse(argc, argv);

  if (base_options.help) {
    base_options.print_usage(std::cout) << std::endl;
    std::cout << "\nThis driver iterates over the BF16 cases from\n"
              << "  benchmarks/device/{cri,bmg}/input_files/input_gdn_bf16.in\n"
              << "calling launcher() once per case. Supply any of\n"
              << "  --batch / --num_v_heads / --num_k_heads / --head_k_dim /\n"
              << "  --head_v_dim / --seq_len\n"
              << "to skip the table and run a single CLI-driven case instead.\n";
    return 0;
  }
  if (base_options.error) {
    return -1;
  }

  /* Activation dtype: bfloat16_t (set via -DACT_TYPE in CMakeLists.txt).
   * SSM state dtype:  always fp32 to avoid accumulation drift across chunks. */
  using T      = ACT_TYPE;
  using StateT = float;

  /* If the user supplied any shape flag on the CLI, honor it and run a
   * single case with the parsed Options. Otherwise iterate the hard-coded
   * sweep table below (mirroring the BF16 entries from
   * benchmarks/config_files/03_gdn/cri/bf16.in). */
  if (base_options.shape_overridden) {
    std::cout << "\n========================================================\n"
              << " GDN case (CLI): batch=" << base_options.batch
              << " num_v_heads=" << base_options.num_v_heads
              << " num_k_heads=" << base_options.num_k_heads
              << " seq_len=" << base_options.seq_len
              << "\n========================================================\n";
    GdnRunner<T, StateT> runner(base_options);
    return runner.run();
  }

  /* Default case table -- kept light so the example stays fast on the CRI
   * simulator. Columns: { batch, num_v_heads, num_k_heads, seq_len };
   * head dims use HEAD_K_DIM / HEAD_V_DIM (128/128), overridable on the CLI.
   * seq_len is a multiple of kChunkSize (64) so --verify is valid. */
  struct GdnCase { int batch, num_v_heads, num_k_heads, seq_len; };
  static constexpr GdnCase kCases[] = {
      { 1,  8, 1,   64 },
      { 4,  8, 1,   64 },
      { 1,  16, 1,   64 },
      { 1, 32, 1,   64 },
      { 1, 16, 1,   256 },
      { 1, 4, 1,   2048 },
      };

  int overall = 0;
  for (auto const& c : kCases) {
    int rc = launcher<T, StateT>(c.batch, c.num_v_heads, c.num_k_heads,
                                 c.seq_len, base_options);
    if (rc != 0 && overall == 0) overall = rc;
  }
  return overall;
}
