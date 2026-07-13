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
#pragma once

/*!
  \file xe35_chunk_gated_delta_rule_launch.hpp
  \brief Header-only launcher that translates the public GDNArguments struct
         into the internal kernel_launcher() call.

  Responsibilities:
    - Validate inputs
    - Cast void* device pointers to the concrete template type T.
    - Submit the five chunkwise stages to the in-order SYCL queue.

  The launcher is an inline function template instantiated at each call site
  (matching the header-only convention of the other applications/ kernels),
  so it pulls in the device-kernel header. Host-only consumers that just need
  the API surface (GDNArguments, get_workspace_sizes) should include the
  lightweight public header xe35_chunk_gated_delta_rule.hpp instead.

  Supported instantiation in current builds:
    <bfloat16_t, float>  -- bf16 activations, fp32 SSM state.
*/

#include "gdn_attention/xe35_chunk_gated_delta_rule.hpp"
#include "gdn_attention/xe35_chunk_gated_delta_rule_kernels.hpp"

namespace cutlass::gdn {

template <typename T, typename StateT>
inline cutlass::Status chunk_gated_delta_rule_launch(
    sycl::queue& queue,
    GDNArguments const& args) {

  // Empty problem: nothing to launch. Return kSuccess (not an error) -- a
  // zero-token / zero-batch problem is a valid no-op, matching how callers
  // batch variable-length sequences (a batch may legitimately have no work).
  if (args.batch_size <= 0 || args.total_virtual_seqlen <= 0) {
    return cutlass::Status::kSuccess;
  }
  // Reject non-positive head / head-dim values before the modulo below so a
  // zero num_k_heads cannot trigger a divide-by-zero, and so downstream grid
  // computations in kernel_launcher (which assume strictly positive
  // num_v_heads / head_*_dim) never see a degenerate problem. Mirrors the
  // shape checks done in the public-API runners in
  // examples/14_xe35_gdn_attention and benchmarks/applications/03_gdn.
  if (args.num_k_heads <= 0 || args.num_v_heads <= 0 ||
      args.head_k_dim <= 0 || args.head_v_dim <= 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }
  // has_initial_state == nullptr is intentionally accepted here: per the
  // GDNArguments docstring it means "every batch has carry-over state".
  // The kernels null-check the pointer at their two load sites.
  if (args.num_v_heads % args.num_k_heads != 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  // NOTE: device pointers (q/k/v, the A/w/u workspaces, core_attn_out,
  // ssm_state, the gate/bias arrays) are NOT null-checked here. They are part
  // of the caller's contract (see GDNArguments): all required pointers must be
  // valid device allocations, and the workspaces must be sized per
  // get_workspace_sizes(). has_initial_state is the one documented nullable
  // pointer. A null required pointer will fault inside the kernels rather than
  // being diagnosed here -- by design
  detail::kernel_launcher<T, StateT>(
      queue,
      static_cast<T*>(args.core_attn_out),
      static_cast<T*>(args.q),
      static_cast<T*>(args.k),
      static_cast<const T*>(args.v),
      static_cast<T*>(args.A_workspace),
      static_cast<T*>(args.w_workspace),
      static_cast<T*>(args.u_workspace),
      args.b,
      args.a,
      args.A_log,
      static_cast<const T*>(args.dt_bias),
      static_cast<StateT*>(args.ssm_state),
      args.ssm_state_stride_0,
      args.query_start_loc,
      args.cache_indices,
      args.has_initial_state,
      args.batch_size,
      args.total_virtual_seqlen,
      args.num_k_heads,
      args.head_k_dim,
      args.num_v_heads,
      args.head_v_dim);

  return cutlass::Status::kSuccess;
}

} // namespace cutlass::gdn
