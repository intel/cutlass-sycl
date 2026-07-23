#pragma once

namespace sycl {
#ifdef __SYCL_DEVICE_ONLY__
  template <class T, int N> using vec_t = T __attribute__((ext_vector_type(N)));
#else
  template <class T, int N> using vec_t = sycl::marray<T, N>;
#endif
}

namespace cute {
namespace detail {

template <typename T, class... Ts, size_t... I>
sycl::vec_t<T, sizeof...(Ts)>
to_vec_impl(const cute::tuple<Ts...>& t, std::index_sequence<I...>) {
    return sycl::vec_t<T, sizeof...(Ts)>{ static_cast<T>(get<I>(t))... };
}

}

template <typename T, class... Ts>
auto to_vec(const cute::tuple<Ts...>& t) {
    return detail::to_vec_impl<T>(t, std::index_sequence_for<Ts...>{});
}

namespace detail {

/*
 * When the destination or source operand is in global memory, the address must be data-element aligned.
 * The address operand form [var + reg] or [reg + reg] is not supported by this instruction.
 * toff is a 32-bit signed integer vector. The vector size must match tensor dimension specified in the
 * instruction qualifier.
 * When ds is set to 4b or 6b, only tiled layout and type1 matrix is supported. For copy from global memory
 * to SLM, fm must set to zero.
 * Only type 1 and type 3 matrices are supported for copy from SLM to global memory.
 * This instruction can only be issued by one elected work-item in a subgroup.
 */

enum CacheCtrl{
  L2uc_L3uc = 0,
  L2uc_L3c, L2uc_L3wb = L2uc_L3c,
  L2c_L3uc, L2wb_L3uc = L2c_L3uc,
  L2c_L3c, L2wb_L3wb = L2c_L3c
};

enum FillMethod {
  Zero = 0, Nan
};

/*
 * Tensor Descriptor will contain global memory data-type
 */

template <typename DataType, CacheCtrl CacheType, FillMethod FM> struct AsyncTensorGlobal2SLM;

template <>
struct AsyncTensorGlobal2SLM<sycl::half, CacheCtrl::L2c_L3uc, FillMethod::Zero> {
  using DataType = sycl::half;

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
      const sycl::vec_t<int32_t, 1>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_workgroup.global.1d.16b.fp.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 2>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_workgroup.global.2d.16b.fp.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 3>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_workgroup.global.3d.16b.fp.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 4>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_workgroup.global.4d.16b.fp.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 5>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_workgroup.global.5d.16b.fp.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
      const sycl::vec_t<int32_t, 1>& coord, uint32_t wg_mask) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_cluster.global.1d.16b.fp.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4, %5;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord), "r"(wg_mask));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 2>& coord, uint32_t wg_mask) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_cluster.global.2d.16b.fp.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4, %5;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord), "r"(wg_mask));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 3>& coord, uint32_t wg_mask) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_cluster.global.3d.16b.fp.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4, %5;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord), "r"(wg_mask));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 4>& coord, uint32_t wg_mask) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_cluster.global.4d.16b.fp.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4, %5;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord), "r"(wg_mask));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 5>& coord, uint32_t wg_mask) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_cluster.global.5d.16b.fp.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4, %5;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord), "r"(wg_mask));
#endif
  }
};

template <>
struct AsyncTensorGlobal2SLM<sycl::ext::oneapi::bfloat16, CacheCtrl::L2c_L3uc, FillMethod::Zero> {
  using DataType = sycl::ext::oneapi::bfloat16;

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
      const sycl::vec_t<int32_t, 1>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_workgroup.global.1d.16b.bf.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 2>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_workgroup.global.2d.16b.bf.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 3>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_workgroup.global.3d.16b.bf.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 4>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_workgroup.global.4d.16b.bf.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 5>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_workgroup.global.5d.16b.bf.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
      const sycl::vec_t<int32_t, 1>& coord, uint32_t wg_mask) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_cluster.global.1d.16b.bf.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4, %5;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord), "r"(wg_mask));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 2>& coord, uint32_t wg_mask) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_cluster.global.2d.16b.bf.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4, %5;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord), "r"(wg_mask));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 3>& coord, uint32_t wg_mask) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_cluster.global.3d.16b.bf.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4, %5;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord), "r"(wg_mask));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 4>& coord, uint32_t wg_mask) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_cluster.global.4d.16b.bf.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4, %5;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord), "r"(wg_mask));
#endif
  }

  static inline void
  copy(MatrixDescriptor Mat, const DataType *GmemPtr, uint64_t* pAbar, TensorPayload* pTDesc,
    const sycl::vec_t<int32_t, 5>& coord, uint32_t wg_mask) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.shared_cluster.global.5d.16b.bf.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], %4, %5;\n"
          ::"r"(Mat), "r"(GmemPtr), "r"(pAbar), "r"(pTDesc), "r"(coord), "r"(wg_mask));
#endif
  }
};

template <int BitWidth, CacheCtrl CacheType> struct AsyncTensorSLM2Global;

template <>
struct AsyncTensorSLM2Global<16, CacheCtrl::L2wb_L3uc> {
  static inline void
  copy(const void* GmemPtr, MatrixDescriptor Mat, uint64_t* pAbar, TensorPayload* pTDesc,
      const sycl::vec_t<int32_t, 1>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.global.shared_workgroup.1d.16b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;\n"
          ::"r"(GmemPtr), "r"(Mat), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }
  static inline void
  copy(const void* GmemPtr, MatrixDescriptor Mat, uint64_t* pAbar, TensorPayload* pTDesc,
      const sycl::vec_t<int32_t, 2>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.global.shared_workgroup.2d.16b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;\n"
          ::"r"(GmemPtr), "r"(Mat), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }
  static inline void
  copy(const void* GmemPtr, MatrixDescriptor Mat, uint64_t* pAbar, TensorPayload* pTDesc,
      const sycl::vec_t<int32_t, 3>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.global.shared_workgroup.3d.16b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;\n"
          ::"r"(GmemPtr), "r"(Mat), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }
  static inline void
  copy(const void* GmemPtr, MatrixDescriptor Mat, uint64_t* pAbar, TensorPayload* pTDesc,
      const sycl::vec_t<int32_t, 4>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.global.shared_workgroup.4d.16b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;\n"
          ::"r"(GmemPtr), "r"(Mat), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }
  static inline void
  copy(const void* GmemPtr, MatrixDescriptor Mat, uint64_t* pAbar, TensorPayload* pTDesc,
      const sycl::vec_t<int32_t, 5>& coord) {
#if defined (__SYCL_DEVICE_ONLY__)
    asm volatile (
          "async_tensor_copy.global.shared_workgroup.5d.16b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;\n"
          ::"r"(GmemPtr), "r"(Mat), "r"(pAbar), "r"(pTDesc), "r"(coord));
#endif
  }
};

}
}
