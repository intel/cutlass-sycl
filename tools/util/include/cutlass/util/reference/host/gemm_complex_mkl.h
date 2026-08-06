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
    \brief BLAS-based host reference GEMM.

    This is a host-side reference GEMM intended as an alternative to
    cutlass::reference::device::GemmComplex for correctness checks that must not
    depend on the target device.

    It is computed on the host with a BLAS xGEMM (cblas_sgemm / cblas_dgemm from
    oneMKL), or if no oneMKL is available, a self-contained loop reference in
    cutlass::reference::host::GemmComplex.
*/

#pragma once

#include <iostream>
#include <vector>

#include "cutlass/coord.h"
#include "cutlass/complex.h"
#include "cutlass/numeric_types.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/platform/platform.h"

#include "cutlass/tensor_view.h"

#include "cutlass/gemm/gemm.h"

// Loop reference used as the portable fallback when no CBLAS is available.
#include "cutlass/util/reference/host/gemm_complex.h"

// Only use CBLAS when the build system guarantees oneMKL is linked
// (CUTLASS_ENABLE_CBLAS). CUTLASS_ENABLE_CBLAS is defined only when the build
// finds and links system oneMKL (MKL::MKL), which provides <mkl.h>. Relying on
// header availability alone is unsafe: an unrelated CBLAS header could be on the
// include path without the corresponding library being linked, producing
// undefined references to cblas_* at link time.
#if defined(CUTLASS_ENABLE_CBLAS)
#  include <mkl.h>
#endif

namespace cutlass {
namespace reference {
namespace host {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a general matrix product on the host using a BLAS xGEMM.
///
/// The result is `D = alpha * (A @ B) + beta * C`, written through `tensor_d`.
/// Operands and output are converted to/from the BLAS compute type (float, or
/// double when `ComputeType` is double), so low-precision types such as bf16,
/// fp16 and fp8 are supported.
///
/// The signature matches cutlass::reference::device::GemmComplex so this can be
/// used as a drop-in host substitute. `initial_accum` is folded into the output
/// as `alpha * initial_accum` (it is normally zero); complex conjugation is not
/// supported (operands are expected to be real).
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename ComputeType,
  typename ElementD = ElementC,
  typename LayoutD = LayoutC
>
bool GemmComplexMkl(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRef<ElementA, LayoutA> tensor_a,
  ComplexTransform transform_a,
  TensorRef<ElementB, LayoutB> tensor_b,
  ComplexTransform transform_b,
  ScalarType beta,
  TensorRef<ElementC, LayoutC> tensor_c,
  TensorRef<ElementD, LayoutD> tensor_d,
  ComputeType initial_accum,
  int batch_count = 1,
  int64_t batch_stride_A = 0,
  int64_t batch_stride_B = 0,
  int64_t batch_stride_C = 0,
  int64_t batch_stride_D = 0) {

  static_assert(
    LayoutA::kRank == 2 &&
    LayoutB::kRank == 2 &&
    LayoutC::kRank == 2 &&
    LayoutD::kRank == 2, "Tensors must be of rank 2");

  (void)transform_a;
  (void)transform_b;

  bool use_mkl_based_impl;

#if !defined(CUTLASS_ENABLE_CBLAS)
  // No host CBLAS available: fall back to the portable loop reference.
  use_mkl_based_impl = false;
  GemmComplex<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ScalarType, ComputeType, ElementD, LayoutD>(
      problem_size, alpha, tensor_a, transform_a, tensor_b, transform_b,
      beta, tensor_c, tensor_d, initial_accum,
      batch_count, batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D);
   return use_mkl_based_impl;
#else
  // CBLAS xGEMM only supports single/double precision. Accumulate in float
  // unless the caller explicitly requested double.
  use_mkl_based_impl = true;
  using BlasType = typename platform::conditional<
      platform::is_same<ComputeType, double>::value, double, float>::type;

  int const M = problem_size.m();
  int const N = problem_size.n();
  int const K = problem_size.k();

  // Column-major operands are passed to CBLAS (row-major) as transposed.
  constexpr bool trans_a = platform::is_same<LayoutA, layout::ColumnMajor>::value;
  constexpr bool trans_b = platform::is_same<LayoutB, layout::ColumnMajor>::value;

  CBLAS_TRANSPOSE const op_a = trans_a ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE const op_b = trans_b ? CblasTrans : CblasNoTrans;

  int const lda = trans_a ? M : K;
  int const ldb = trans_b ? K : N;
  int const ldc = N;

  NumericConverter<ElementD, BlasType> convert_op;

  BlasType const alpha_blas = static_cast<BlasType>(alpha);
  BlasType const beta_blas = static_cast<BlasType>(beta);

  // initial_accum is part of the (alpha-scaled) accumulator in the reference
  // semantics: D = alpha * (initial_accum + A @ B) + beta * C. CBLAS computes
  // alpha * A @ B + beta * C, so the alpha * initial_accum term is added after.
  BlasType const accum_bias = alpha_blas * static_cast<BlasType>(initial_accum);

  std::vector<BlasType> host_a(static_cast<size_t>(M) * K);
  std::vector<BlasType> host_b(static_cast<size_t>(K) * N);
  std::vector<BlasType> host_c(static_cast<size_t>(M) * N);

  for (int batch_idx = 0; batch_idx < batch_count; ++batch_idx) {

    ElementA const* ptr_a = tensor_a.data() + static_cast<int64_t>(batch_idx) * batch_stride_A;
    ElementB const* ptr_b = tensor_b.data() + static_cast<int64_t>(batch_idx) * batch_stride_B;
    ElementC const* ptr_c = tensor_c.data() + static_cast<int64_t>(batch_idx) * batch_stride_C;
    ElementD* ptr_d = tensor_d.data() + static_cast<int64_t>(batch_idx) * batch_stride_D;

    for (size_t i = 0; i < host_a.size(); ++i) { host_a[i] = static_cast<BlasType>(ptr_a[i]); }
    for (size_t i = 0; i < host_b.size(); ++i) { host_b[i] = static_cast<BlasType>(ptr_b[i]); }
    // CBLAS overwrites C in place with alpha * A @ B + beta * C, so seed it with C.
    for (size_t i = 0; i < host_c.size(); ++i) { host_c[i] = static_cast<BlasType>(ptr_c[i]); }

    if constexpr (platform::is_same<BlasType, double>::value) {
      cblas_dgemm(CblasRowMajor, op_a, op_b, M, N, K,
                  alpha_blas, host_a.data(), lda, host_b.data(), ldb,
                  beta_blas, host_c.data(), ldc);
    } else {
      cblas_sgemm(CblasRowMajor, op_a, op_b, M, N, K,
                  alpha_blas, host_a.data(), lda, host_b.data(), ldb,
                  beta_blas, host_c.data(), ldc);
    }

    for (size_t i = 0; i < host_c.size(); ++i) {
      ptr_d[i] = convert_op(host_c[i] + accum_bias);
    }
  }

  return use_mkl_based_impl;
#endif // CUTLASS_ENABLE_CBLAS
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a general matrix product on the host using a BLAS xGEMM,
/// accumulating in `float`.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename ElementD = ElementC,
  typename LayoutD = LayoutC
>
void GemmComplexMkl(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRef<ElementA, LayoutA> tensor_a,
  ComplexTransform transform_a,
  TensorRef<ElementB, LayoutB> tensor_b,
  ComplexTransform transform_b,
  ScalarType beta,
  TensorRef<ElementC, LayoutC> tensor_c,
  TensorRef<ElementD, LayoutD> tensor_d) {

  GemmComplexMkl<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
    ScalarType, float, ElementD, LayoutD>(
    problem_size, alpha, tensor_a, transform_a, tensor_b, transform_b,
    beta, tensor_c, tensor_d, float(0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass
