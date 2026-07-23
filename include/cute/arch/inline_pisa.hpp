#pragma once

#include "xe4_util.hpp"

#ifdef AMMA_GENERATED
#include "async_gmma.hpp"
#endif

#ifdef TP_GENERATED
#include "generated_headers/tensorpipe_inline.hpp"
#endif

#ifdef VC_WA
#define ALLOCATE_ABAR(reg_name, abar_name, abar_bytes)
#define ALLOCATE_TDESC(reg_name, tdesc_name, ret)
#else
#define ALLOCATE_ABAR(reg_name, abar_name, abar_bytes) \
  INLINE_PISA(".abarrier .align 8 .8b @" #abar_name "[%0];" ::"i"(abar_bytes)); \
  INLINE_PISA(".reg .32b %%" #reg_name \
              ";\n\t" \
              "addrof.32b %%" #reg_name ", @" #abar_name \
              ";\n\t" \
              "addrcast.generic.abarrier  %0, %%" #reg_name ";" \
              : "=r"(ret) \
              :)

#define ALLOCATE_TDESC(reg_name, tdesc_name, ret) \
  INLINE_PISA(".tensordesc .align 64 .8b @" #tdesc_name \
              "[64];\n\t" \
              ".reg .32b %%" #reg_name \
              ";\n\t" \
              "addrof.32b %%" #reg_name ", @" #tdesc_name \
              ";\n\t" \
              "addrcast.generic.tensordesc  %0, %%" #reg_name ";" \
              : "=r"(ret) \
              :)
#endif

// Each N can be call only once
template <uint32_t IDX, uint32_t abar_bytes>
inline uint8_t *allocate_abar_bytes() {
  uint8_t *ret;
  if constexpr (IDX == 0) {
    ALLOCATE_ABAR(abar0, ABAR0, abar_bytes);
  } else if constexpr (IDX == 1) {
    ALLOCATE_ABAR(abar1, ABAR1, abar_bytes);
  } else if constexpr (IDX == 2) {
    ALLOCATE_ABAR(abar2, ABAR2, abar_bytes);
  } else if constexpr (IDX == 3) {
    ALLOCATE_ABAR(abar3, ABAR3, abar_bytes);
  } else if constexpr (IDX == 4) {
    ALLOCATE_ABAR(abar4, ABAR4, abar_bytes);
  } else {
    static_assert(IDX >= 0 && IDX <= 4, "Unsupported IDX for allocate_abar");
  }
  return ret;
}

// Each N can be call only once
template <uint32_t IDX, uint32_t abar_count = 1, typename abar_ptr_t = uint64_t *>
inline abar_ptr_t allocate_abar() {
  auto ret = allocate_abar_bytes<IDX, abar_count * 8>();
  return reinterpret_cast<abar_ptr_t>(ret);
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_init(abar_ptr_t abar_ptr, uint32_t total_arrive_cnt) {
  INLINE_PISA("abarrier.init.64b [%0], %1;" ::"r"(abar_ptr), "r"(total_arrive_cnt));
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_workgroup_expect_tx(abar_ptr_t abar_ptr, int32_t tx_cnt) {
  INLINE_PISA("abarrier.expect_tx.abarrier_workgroup.relaxed.64b [%0], %1;" ::"r"(abar_ptr), "r"(tx_cnt));
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_workgroup_arrive(abar_ptr_t abar_ptr, uint32_t arrive_cnt) {
  uint64_t temp;
  // sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
  INLINE_PISA("fence.shared.workgroup;");
  INLINE_PISA("abarrier.arrive.abarrier_workgroup.release.64b %0, [%1], %2;"
              : "=r"(temp)
              : "r"(abar_ptr), "r"(arrive_cnt));
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_workgroup_arrive_and_drop(abar_ptr_t abar_ptr, uint32_t arrive_cnt) {
  uint64_t temp;
  INLINE_PISA("fence.shared.workgroup;");
  INLINE_PISA("abarrier.arrive_drop.abarrier_workgroup.release.64b %0, [%1], %2;"
              : "=r"(temp)
              : "r"(abar_ptr), "r"(arrive_cnt));
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_workgroup_arrive_drop_expect_tx(abar_ptr_t abar_ptr, uint32_t tx_cnt) {
  uint64_t temp;
  INLINE_PISA("fence.shared.workgroup;");
  INLINE_PISA("abarrier.arrive_drop.expect_tx.abarrier_workgroup.release.64b %0, [%1], %2;"
              : "=r"(temp)
              : "r"(abar_ptr), "r"(tx_cnt));
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_workgroup_arrives(abar_ptr_t abar_ptr, uint32_t arrive_cnt) {
  uint64_t temp;
  INLINE_PISA("abarrier.arrive.abarrier_workgroup.release.64b %0, [%1], %2;"
              : "=r"(temp)
              : "r"(abar_ptr), "r"(arrive_cnt));
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_workgroup_arrive_expect_tx(abar_ptr_t abar_ptr, int32_t tx_cnt) {
  uint64_t temp;
  INLINE_PISA("abarrier.arrive.expect_tx.abarrier_workgroup.release.64b %0, [%1], %2;"
              : "=r"(temp)
              : "r"(abar_ptr), "r"(tx_cnt));
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_workgroup_complete_tx(abar_ptr_t abar_ptr, int32_t tx_cnt) {
  INLINE_PISA("abarrier.complete_tx.abarrier_workgroup.relaxed.64b  [%0], %1;" ::"r"(abar_ptr), "r"(tx_cnt));
}

template <typename abar_ptr_t = uint64_t *>
inline abar_ptr_t get_remote_abar_address(abar_ptr_t local_abar, uint32_t wg_rank) {
  abar_ptr_t remote_abar;
  INLINE_PISA("abarrier.getaddr.64b %0, %1, %2;" ::"=r"(remote_abar), "r"(local_abar), "r"(wg_rank));
  return remote_abar;
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_cluster_arrive(abar_ptr_t abar_ptr, uint32_t arrive_cnt) {
  INLINE_PISA("abarrier.arrive.abarrier_cluster.cluster.64b [%0], %1;" ::"r"(abar_ptr), "r"(arrive_cnt));
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_cluster_expect_tx(abar_ptr_t abar_ptr, int32_t tx_cnt) {
  INLINE_PISA("abarrier.expect_tx.abarrier_cluster.relaxed.64b [%0], %1;" ::"r"(abar_ptr), "r"(tx_cnt));
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_cluster_complete_tx(abar_ptr_t abar_ptr, int32_t tx_cnt) {
  INLINE_PISA("abarrier.complete_tx.abarrier_cluster.relaxed.64b  [%0], %1;" ::"r"(abar_ptr), "r"(tx_cnt));
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_cluster_arrive_expect_tx(abar_ptr_t abar_ptr, int32_t tx_cnt) {
  INLINE_PISA("abarrier.arrive.expect_tx.abarrier_cluster.cluster.64b [%0], %1;" ::"r"(abar_ptr), "r"(tx_cnt));
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_try_wait(abar_ptr_t abar_ptr, uint32_t phase_bit) {
  INLINE_PISA("abarrier.try_wait.parity.acquire.64b  [%0], %1;" ::"r"(abar_ptr), "r"(phase_bit));
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_try(abar_ptr_t abar_ptr, uint32_t phase_bit) {
  INLINE_PISA("abarrier.try.parity.64b  [%0], %1;" ::"r"(abar_ptr), "r"(phase_bit));
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_wait(abar_ptr_t abar_ptr) {
  INLINE_PISA("abarrier.wait.acquire.64b  [%0];" ::"r"(abar_ptr));
}

template <typename abar_ptr_t = uint64_t *>
inline void abarrier_inval(abar_ptr_t abar_ptr) {
  INLINE_PISA("abarrier.inval.64b [%0];" ::"r"(abar_ptr));
}

// Each IDX can be call only once
template <int IDX, typename tdesc_ptr_t = uint64_t *>
inline tdesc_ptr_t allocate_tdesc() {
  tdesc_ptr_t ret;

  if constexpr (IDX == 0) {
    ALLOCATE_TDESC(tdesc0, TDESC0, ret);
  } else if constexpr (IDX == 1) {
    ALLOCATE_TDESC(tdesc1, TDESC1, ret);
  } else if constexpr (IDX == 2) {
    ALLOCATE_TDESC(tdesc2, TDESC2, ret);
  } else if constexpr (IDX == 3) {
    ALLOCATE_TDESC(tdesc3, TDESC3, ret);
  } else if constexpr (IDX == 4) {
    ALLOCATE_TDESC(tdesc4, TDESC4, ret);
  } else if constexpr (IDX == 5) {
    ALLOCATE_TDESC(tdesc5, TDESC5, ret);
  } else if constexpr (IDX == 6) {
    ALLOCATE_TDESC(tdesc6, TDESC6, ret);
  } else if constexpr (IDX == 7) {
    ALLOCATE_TDESC(tdesc7, TDESC7, ret);
  } else {
    static_assert(IDX >= 0 && IDX <= 7, "Unsupported IDX for allocate_tdesc");
  }

  return ret;
}

template <uint32_t dim_num, typename dims_t, typename tdesc_ptr_t = uint64_t *>
inline void tensordesc_fill_dim_size(tdesc_ptr_t tdesc_ptr, const dims_t &dim_sizes) {
  static_assert(dim_num > 0 && dim_num <= 5, "Unsupported dimention size for tensor descriptor dim_size");
  static_assert(sizeof(dims_t {}[0]) == sizeof(uint32_t), "Invalid dim size type");

  INLINE_PISA("tensordesc.fill.dim_size.32b [%0], 0, %1;" ::"r"(tdesc_ptr), "r"(dim_sizes[0] - 1));
  if constexpr (dim_num > 1) {
    INLINE_PISA("tensordesc.fill.dim_size.32b [%0], 1, %1;" ::"r"(tdesc_ptr), "r"(dim_sizes[1] - 1));
  }
  if constexpr (dim_num > 2) {
    INLINE_PISA("tensordesc.fill.dim_size.32b [%0], 2, %1;" ::"r"(tdesc_ptr), "r"(dim_sizes[2] - 1));
  }
  if constexpr (dim_num > 3) {
    INLINE_PISA("tensordesc.fill.dim_size.32b [%0], 3, %1;" ::"r"(tdesc_ptr), "r"(dim_sizes[3] - 1));
  }
  if constexpr (dim_num > 4) {
    INLINE_PISA("tensordesc.fill.dim_size.32b [%0], 4, %1;" ::"r"(tdesc_ptr), "r"(dim_sizes[4] - 1));
  }
}

template <uint32_t dim_num, typename tdesc_ptr_t = uint64_t *>
inline void tensordesc_set_dim_size(tdesc_ptr_t tdesc_ptr, uint32_t dim_size) {
  static_assert(dim_num > 0 && dim_num <= 5, "Unsupported dimention size for tensor descriptor dim_size");

  if constexpr (dim_num == 1) {
    INLINE_PISA("tensordesc.fill.dim_size.32b [%0], 0, %1;" ::"r"(tdesc_ptr), "r"(dim_size - 1));
  } else if constexpr (dim_num == 2) {
    INLINE_PISA("tensordesc.fill.dim_size.32b [%0], 1, %1;" ::"r"(tdesc_ptr), "r"(dim_size - 1));
  } else if constexpr (dim_num == 3) {
    INLINE_PISA("tensordesc.fill.dim_size.32b [%0], 2, %1;" ::"r"(tdesc_ptr), "r"(dim_size - 1));
  } else if constexpr (dim_num == 4) {
    INLINE_PISA("tensordesc.fill.dim_size.32b [%0], 3, %1;" ::"r"(tdesc_ptr), "r"(dim_size - 1));
  } else if constexpr (dim_num == 5) {
    INLINE_PISA("tensordesc.fill.dim_size.32b [%0], 4, %1;" ::"r"(tdesc_ptr), "r"(dim_size - 1));
  }
}

template <uint32_t dim_num, typename strides_t, typename tdesc_ptr_t = uint64_t *>
inline void tensordesc_fill_dim_stride(tdesc_ptr_t tdesc_ptr, const strides_t &dim_strides) {
  static_assert(dim_num >= 2 && dim_num <= 5, "Unsupported dimention size fortensor descriptor dim_stride");
  static_assert(sizeof(strides_t {}[0]) == sizeof(uint64_t), "Invalid stride type");

  INLINE_PISA("tensordesc.fill.dim_stride.64b [%0], 1, %1;" ::"r"(tdesc_ptr), "r"(dim_strides[0]));
  if constexpr (dim_num > 2) {
    INLINE_PISA("tensordesc.fill.dim_stride.64b [%0], 2, %1;" ::"r"(tdesc_ptr), "r"(dim_strides[1]));
  }
  if constexpr (dim_num > 3) {
    INLINE_PISA("tensordesc.fill.dim_stride.64b [%0], 3, %1;" ::"r"(tdesc_ptr), "r"(dim_strides[2]));
  }
  if constexpr (dim_num > 4) {
    INLINE_PISA("tensordesc.fill.dim_stride.64b [%0], 4, %1;" ::"r"(tdesc_ptr), "r"(dim_strides[3]));
  }
}

template <uint32_t dim_num, typename dims_t, typename tdesc_ptr_t = uint64_t *>
inline void tensordesc_fill_roitensor_dim_size(tdesc_ptr_t tdesc_ptr, const dims_t &roitensor_sizes) {
  static_assert(dim_num > 0 && dim_num <= 5, "Unsupported dimention size for tensor descriptor roitensor_size");
  static_assert(sizeof(dims_t {}[0]) == sizeof(uint32_t), "Invalid roitensor size type");

  INLINE_PISA("tensordesc.fill.roitensor_dim_size.32b [%0], 0, %1;" ::"r"(tdesc_ptr), "r"(roitensor_sizes[0] - 1));
  if constexpr (dim_num > 1) {
    INLINE_PISA("tensordesc.fill.roitensor_dim_size.32b [%0], 1, %1;" ::"r"(tdesc_ptr), "r"(roitensor_sizes[1] - 1));
  }
  if constexpr (dim_num > 2) {
    INLINE_PISA("tensordesc.fill.roitensor_dim_size.32b [%0], 2, %1;" ::"r"(tdesc_ptr), "r"(roitensor_sizes[2] - 1));
  }
  if constexpr (dim_num > 3) {
    INLINE_PISA("tensordesc.fill.roitensor_dim_size.32b [%0], 3, %1;" ::"r"(tdesc_ptr), "r"(roitensor_sizes[3] - 1));
  }
  if constexpr (dim_num > 4) {
    INLINE_PISA("tensordesc.fill.roitensor_dim_size.32b [%0], 4, %1;" ::"r"(tdesc_ptr), "r"(roitensor_sizes[4] - 1));
  }
}

template <uint32_t dim_num, typename strides_t, typename tdesc_ptr_t = uint64_t *>
inline void tensordesc_fill_element_stride(tdesc_ptr_t tdesc_ptr, const strides_t &element_stride) {
  static_assert(dim_num > 0 && dim_num <= 5, "Unsupported dimention size for tensor descriptor element_stride");
  static_assert(sizeof(strides_t {}[0]) == sizeof(uint32_t), "Invalid element stride type");

  INLINE_PISA("tensordesc.fill.element_stride.32b [%0], 0, %1;" ::"r"(tdesc_ptr), "r"(element_stride[0] - 1));
  if constexpr (dim_num > 1) {
    INLINE_PISA("tensordesc.fill.element_stride.32b [%0], 1, %1;" ::"r"(tdesc_ptr), "r"(element_stride[1] - 1));
  }
  if constexpr (dim_num > 2) {
    INLINE_PISA("tensordesc.fill.element_stride.32b [%0], 2, %1;" ::"r"(tdesc_ptr), "r"(element_stride[2] - 1));
  }
  if constexpr (dim_num > 3) {
    INLINE_PISA("tensordesc.fill.element_stride.32b [%0], 3, %1;" ::"r"(tdesc_ptr), "r"(element_stride[3] - 1));
  }
  if constexpr (dim_num > 4) {
    INLINE_PISA("tensordesc.fill.element_stride.32b [%0], 4, %1;" ::"r"(tdesc_ptr), "r"(element_stride[4] - 1));
  }
}

template <size_t dim, typename dtype, typename tdesc_t, typename abar_t, typename mat_desc_t>
inline void async_tensor_load(tdesc_t tdesc, mat_desc_t mat_desc, dtype *gmem_ptr,
                              const sycl::marray<int32_t, dim> &coord, abar_t abar) {
  if constexpr (dim == 2) {
    if constexpr (sizeof_bits<dtype>() == 4) {
      INLINE_PISA(
          "async_tensor_copy.shared_workgroup.global.2d.4b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 6) {
      INLINE_PISA(
          "async_tensor_copy.shared_workgroup.global.2d.6b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 8 || sizeof(dtype) == 1) {
      INLINE_PISA(
          "async_tensor_copy.shared_workgroup.global.2d.8b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 16) {
      INLINE_PISA(
          "async_tensor_copy.shared_workgroup.global.2d.16b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 32) {
      INLINE_PISA(
          "async_tensor_copy.shared_workgroup.global.2d.32b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 64) {
      INLINE_PISA(
          "async_tensor_copy.shared_workgroup.global.2d.64b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else {
      static_assert(false, "Unsupported data size");
    }
  } else if constexpr (dim == 3) {
    if constexpr (sizeof(dtype) == 1) {
      INLINE_PISA(
          "async_tensor_copy.shared_workgroup.global.3d.8b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof(dtype) == 2) {
      INLINE_PISA(
          "async_tensor_copy.shared_workgroup.global.3d.16b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof(dtype) == 4) {
      INLINE_PISA(
          "async_tensor_copy.shared_workgroup.global.3d.32b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof(dtype) == 8) {
      INLINE_PISA(
          "async_tensor_copy.shared_workgroup.global.3d.64b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else {
      static_assert(false, "Unsupported data size");
    }
  } else if constexpr (dim == 4) {
    if constexpr (sizeof(dtype) == 1) {
      INLINE_PISA(
          "async_tensor_copy.shared_workgroup.global.4d.8b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof(dtype) == 2) {
      INLINE_PISA(
          "async_tensor_copy.shared_workgroup.global.4d.16b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof(dtype) == 4) {
      INLINE_PISA(
          "async_tensor_copy.shared_workgroup.global.4d.32b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof(dtype) == 8) {
      INLINE_PISA(
          "async_tensor_copy.shared_workgroup.global.4d.64b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else {
      static_assert(false, "Unsupported data size");
    }
  } else {
    static_assert(false, "Unsupported dim");
  }
}

template <size_t dim, typename dtype, typename tdesc_t, typename abar_t, typename mat_desc_t>
inline void async_tensor_load(tdesc_t tdesc, mat_desc_t mat_desc, dtype *gmem_ptr,
                              const sycl::marray<int32_t, dim> &coord, abar_t abar, uint32_t wg_mask) {
  if constexpr (dim == 2) {
    if constexpr (sizeof_bits<dtype>() == 4) {
      INLINE_PISA(
          "async_tensor_copy.shared_cluster.global.2d.4b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4, %5;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)), "r"(wg_mask));
    } else if constexpr (sizeof_bits<dtype>() == 8) {
      INLINE_PISA(
          "async_tensor_copy.shared_cluster.global.2d.8b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4, %5;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)), "r"(wg_mask));
    } else if constexpr (sizeof_bits<dtype>() == 16) {
      INLINE_PISA(
          "async_tensor_copy.shared_cluster.global.2d.16b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4, %5;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)), "r"(wg_mask));
    } else if constexpr (sizeof_bits<dtype>() == 32) {
      INLINE_PISA(
          "async_tensor_copy.shared_cluster.global.2d.32b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4, %5;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)), "r"(wg_mask));
    } else if constexpr (sizeof_bits<dtype>() == 64) {
      INLINE_PISA(
          "async_tensor_copy.shared_cluster.global.2d.64b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4, %5;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)), "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported data size");
    }
  } else if constexpr (dim == 3) {
    if constexpr (sizeof_bits<dtype>() == 4) {
      INLINE_PISA(
          "async_tensor_copy.shared_cluster.global.3d.4b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4, %5;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)), "r"(wg_mask));
    } else if constexpr (sizeof_bits<dtype>() == 8) {
      INLINE_PISA(
          "async_tensor_copy.shared_cluster.global.3d.8b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4, %5;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)), "r"(wg_mask));
    } else if constexpr (sizeof_bits<dtype>() == 16) {
      INLINE_PISA(
          "async_tensor_copy.shared_cluster.global.3d.16b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4, %5;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)), "r"(wg_mask));
    } else if constexpr (sizeof_bits<dtype>() == 32) {
      INLINE_PISA(
          "async_tensor_copy.shared_cluster.global.3d.32b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4, %5;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)), "r"(wg_mask));
    } else if constexpr (sizeof_bits<dtype>() == 64) {
      INLINE_PISA(
          "async_tensor_copy.shared_cluster.global.3d.64b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4, %5;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)), "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported data size");
    }
  } else if constexpr (dim == 4) {
    if constexpr (sizeof_bits<dtype>() == 4) {
      INLINE_PISA(
          "async_tensor_copy.shared_cluster.global.4d.4b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4, %5;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)), "r"(wg_mask));
    } else if constexpr (sizeof_bits<dtype>() == 8) {
      INLINE_PISA(
          "async_tensor_copy.shared_cluster.global.4d.8b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4, %5;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)), "r"(wg_mask));
    } else if constexpr (sizeof_bits<dtype>() == 16) {
      INLINE_PISA(
          "async_tensor_copy.shared_cluster.global.4d.16b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4, %5;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)), "r"(wg_mask));
    } else if constexpr (sizeof_bits<dtype>() == 32) {
      INLINE_PISA(
          "async_tensor_copy.shared_cluster.global.4d.32b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4, %5;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)), "r"(wg_mask));
    } else if constexpr (sizeof_bits<dtype>() == 64) {
      INLINE_PISA(
          "async_tensor_copy.shared_cluster.global.4d.64b.uint.zero.L2c.L3uc.abarrier %0, [%1], [%2], [%3], "
          "%4, %5;" ::"r"(mat_desc),
          "r"(gmem_ptr), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)), "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported data size");
    }
  } else {
    static_assert(false, "Unsupported dim");
  }
}

template <uint32_t dim, typename dtype, typename tdesc_ptr_t = uint64_t *>
inline void async_tensor_prefetch(tdesc_ptr_t tdesc_ptr, dtype *gmem_ptr, const sycl::marray<int32_t, dim> &toff) {
  if constexpr (dim == 2) {
    if constexpr (sizeof(dtype) == 1) {
      INLINE_PISA("async_tensor_prefetch.2d.8b.L2c.L3uc.global [%0], [%1], %2;" ::"r"(gmem_ptr), "r"(tdesc_ptr),
                  "r"(as_vector_t(toff)));
    } else if constexpr (sizeof(dtype) == 2) {
      INLINE_PISA("async_tensor_prefetch.2d.16b.L2c.L3uc.global [%0], [%1], %2;" ::"r"(gmem_ptr), "r"(tdesc_ptr),
                  "r"(as_vector_t(toff)));
    } else if constexpr (sizeof(dtype) == 4) {
      INLINE_PISA("async_tensor_prefetch.2d.32b.L2c.L3uc.global [%0], [%1], %2;" ::"r"(gmem_ptr), "r"(tdesc_ptr),
                  "r"(as_vector_t(toff)));
    } else if constexpr (sizeof(dtype) == 8) {
      INLINE_PISA("async_tensor_prefetch.2d.64b.L2c.L3uc.global [%0], [%1], %2;" ::"r"(gmem_ptr), "r"(tdesc_ptr),
                  "r"(as_vector_t(toff)));
    } else {
      static_assert(false, "Unsupported data size");
    }
  } else if constexpr (dim == 4) {
    if constexpr (sizeof(dtype) == 1) {
      INLINE_PISA("async_tensor_prefetch.4d.8b.L2c.L3uc.global [%0], [%1], %2;" ::"r"(gmem_ptr), "r"(tdesc_ptr),
                  "r"(as_vector_t(toff)));
    } else if constexpr (sizeof(dtype) == 2) {
      INLINE_PISA("async_tensor_prefetch.4d.16b.L2c.L3uc.global [%0], [%1], %2;" ::"r"(gmem_ptr), "r"(tdesc_ptr),
                  "r"(as_vector_t(toff)));
    } else if constexpr (sizeof(dtype) == 4) {
      INLINE_PISA("async_tensor_prefetch.4d.32b.L2c.L3uc.global [%0], [%1], %2;" ::"r"(gmem_ptr), "r"(tdesc_ptr),
                  "r"(as_vector_t(toff)));
    } else if constexpr (sizeof(dtype) == 8) {
      INLINE_PISA("async_tensor_prefetch.4d.64b.L2c.L3uc.global [%0], [%1], %2;" ::"r"(gmem_ptr), "r"(tdesc_ptr),
                  "r"(as_vector_t(toff)));
    } else {
      static_assert(false, "Unsupported data size");
    }
  } else {
    static_assert(false, "Unsupported dim");
  }
}

template <size_t dim, typename dtype, typename tdesc_t, typename abar_t, typename mat_desc_t>
inline void async_tensor_store(tdesc_t tdesc, mat_desc_t mat_desc, dtype *gmem_ptr,
                               const sycl::marray<int32_t, dim> &coord, abar_t abar) {
  if constexpr (dim == 2) {
    if constexpr (sizeof_bits<dtype>() == 4) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.2d.4b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 6) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.2d.6b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 8) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.2d.8b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 16) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.2d.16b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 32) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.2d.32b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 64) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.2d.64b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else {
      static_assert(false, "Unsupported data size");
    }
  } else if constexpr (dim == 3) {
    if constexpr (sizeof_bits<dtype>() == 4) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.3d.4b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 6) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.3d.6b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 8) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.3d.8b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 16) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.3d.16b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 32) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.3d.32b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 64) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.3d.64b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else {
      static_assert(false, "Unsupported data size");
    }
  } else if constexpr (dim == 4) {
    if constexpr (sizeof_bits<dtype>() == 4) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.4d.4b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 8) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.4d.8b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 16) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.4d.16b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 32) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.4d.32b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (sizeof_bits<dtype>() == 64) {
      INLINE_PISA("async_tensor_copy.global.shared_workgroup.4d.64b.L2wb.L3uc.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
                      gmem_ptr),
                  "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else {
      static_assert(false, "Unsupported data size");
    }
  } else {
    static_assert(false, "Unsupported dim");
  }
}

template <size_t dim, typename dtype, typename tdesc_t, typename abar_t, typename mat_desc_t>
inline void async_tensor_atomic_store(tdesc_t tdesc, mat_desc_t mat_desc, dtype *gmem_ptr,
                                      const sycl::marray<int32_t, dim> &coord, abar_t abar) {
  if constexpr (dim == 2) {
    if constexpr (std::is_same_v<dtype, fp16>) {
      INLINE_PISA(
          "async_tensor_fred.global.shared_workgroup.2d.add.hf.L2wb.L3wb.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
              gmem_ptr),
          "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (std::is_same_v<dtype, bf16>) {
      INLINE_PISA(
          "async_tensor_fred.global.shared_workgroup.2d.add.bf.L2wb.L3wb.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
              gmem_ptr),
          "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (std::is_same_v<dtype, float>) {
      INLINE_PISA(
          "async_tensor_fred.global.shared_workgroup.2d.add.f.L2wb.L3wb.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
              gmem_ptr),
          "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (std::is_same_v<dtype, double>) {
      INLINE_PISA(
          "async_tensor_fred.global.shared_workgroup.2d.add.df.L2wb.L3wb.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
              gmem_ptr),
          "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else {
      static_assert(false, "Unsupported data size");
    }
  } else if constexpr (dim == 4) {
    if constexpr (std::is_same_v<dtype, fp16>) {
      INLINE_PISA(
          "async_tensor_fred.global.shared_workgroup.4d.add.hf.L2wb.L3wb.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
              gmem_ptr),
          "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (std::is_same_v<dtype, bf16>) {
      INLINE_PISA(
          "async_tensor_fred.global.shared_workgroup.4d.add.bf.L2wb.L3wb.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
              gmem_ptr),
          "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (std::is_same_v<dtype, float>) {
      INLINE_PISA(
          "async_tensor_fred.global.shared_workgroup.4d.add.f.L2wb.L3wb.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
              gmem_ptr),
          "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else if constexpr (std::is_same_v<dtype, double>) {
      INLINE_PISA(
          "async_tensor_fred.global.shared_workgroup.4d.add.df.L2wb.L3wb.abarrier [%0], %1, [%2], [%3], %4;" ::"r"(
              gmem_ptr),
          "r"(mat_desc), "r"(abar), "r"(tdesc), "r"(as_vector_t(coord)));
    } else {
      static_assert(false, "Unsupported data size");
    }
  } else {
    static_assert(false, "Unsupported dim");
  }
}

inline void cbar_arrive() {
#ifndef VC_WA
  INLINE_PISA("cbarrier.arrive.release;" ::);
#endif
}

inline void cbar_wait() {
#ifndef VC_WA
  INLINE_PISA("cbarrier.wait.acquire;" ::);
#endif
}

// Note: this already scaled with cluster_size
template <uint32_t dim = 0>
uint32_t inline get_cluster_id() {
  uint32_t ret;
  if constexpr (dim == 0) {
    INLINE_PISA("mov.32b %0, %%clusterid.x;" : "=r"(ret) :);
  } else if constexpr (dim == 1) {
    INLINE_PISA("mov.32b %0, %%clusterid.y;" : "=r"(ret) :);
  } else if constexpr (dim == 2) {
    INLINE_PISA("mov.32b %0, %%clusterid.z;" : "=r"(ret) :);
  } else {
    static_assert(sizeof(dim) < 0, "wrong dim");
  }
  return ret;
}

template <uint32_t dim = 0>
uint32_t inline get_cluster_wgid() {
  uint32_t ret;
  if constexpr (dim == 0) {
    INLINE_PISA("mov.32b %0, %%clusterwgid.x;" : "=r"(ret) :);
  } else if constexpr (dim == 1) {
    INLINE_PISA("mov.32b %0, %%clusterwgid.y;" : "=r"(ret) :);
  } else if constexpr (dim == 2) {
    INLINE_PISA("mov.32b %0, %%clusterwgid.z;" : "=r"(ret) :);
  } else {
    static_assert(sizeof(dim) < 0, "wrong dim");
  }
  return ret;
}

// async linear copy, the `size` is in unit of bytes
template <typename dtype>
inline void async_linear_prefetch(dtype *gmem_ptr, uint32_t size) {
  INLINE_PISA("async_linear_prefetch.L2c.L3uc.global [%0], %1;" ::"r"(gmem_ptr), "r"(size));
}

template <typename slm_dtype, typename dtype, typename abar_ptr_t = uint64_t *>
inline void async_linear_load(slm_dtype *slm_ptr, dtype *gmem_ptr, uint32_t size, abar_ptr_t abar_ptr) {
  INLINE_PISA("async_linear_copy.shared_workgroup.global.L2c.L3uc.abarrier [%0], [%1], [%2], %3;" ::"r"(slm_ptr),
              "r"(gmem_ptr), "r"(abar_ptr), "r"(size));
}

template <typename slm_dtype, typename dtype, typename abar_ptr_t = uint64_t *>
inline void async_linear_load(slm_dtype *slm_ptr, dtype *gmem_ptr, uint32_t size, abar_ptr_t abar_ptr,
                              uint32_t wg_mask) {
  INLINE_PISA("async_linear_copy.shared_cluster.global.L2c.L3uc.abarrier [%0], [%1], [%2], %3, %4;" ::"r"(slm_ptr),
              "r"(gmem_ptr), "r"(abar_ptr), "r"(size), "r"(wg_mask));
}

template <typename slm_dtype, typename dtype, typename abar_ptr_t = uint64_t *>
inline void async_linear_store(dtype *gmem_ptr, slm_dtype *slm_ptr, uint32_t size, abar_ptr_t abar_ptr) {
  INLINE_PISA("async_linear_copy.global.shared_workgroup.L2wb.L3uc.abarrier [%0], [%1], [%2], %3;" ::"r"(gmem_ptr),
              "r"(slm_ptr), "r"(abar_ptr), "r"(size));
}

#define ROW_LOAD_LINEAR_A64(row_size, basic_dtype, data_bits) \
  "async_row_copy.shared_workgroup.global.linear." #row_size ".a64." #data_bits "." #basic_dtype \
  ".zero.L2c.L3uc.abarrier [%0], [%1], [%2], %3;"

template <uint32_t row_size, typename dtype, typename slm_dtype, typename abar_ptr_t = uint64_t *>
inline void row_copy_linear_a64_load(slm_dtype *slm_ptr, uint64_t offset, uint32_t size, abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_LINEAR_A64(32, uint, 32b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_LINEAR_A64(512, uint, 32b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_LINEAR_A64(1024, uint, 32b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_LINEAR_A64(32, uint, 16b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_LINEAR_A64(512, uint, 16b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_LINEAR_A64(1024, uint, 16b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_LINEAR_A64(2048, uint, 16b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 6) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_LINEAR_A64(32, uint, 6b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 96) {
      INLINE_PISA(ROW_LOAD_LINEAR_A64(96, uint, 6b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_LINEAR_A64(512, uint, 6b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_LINEAR_A64(1024, uint, 6b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_LINEAR_A64(2048, uint, 6b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_LOAD_LINEAR_NCS_A64(row_size, data_bits) \
  "async_row_copy.shared_workgroup.global.ncs.linear." #row_size ".a64." #data_bits \
  ".L2c.L3uc.abarrier [%0], [%1], [%2];"

template <uint32_t row_size, typename dtype, typename slm_dtype, typename abar_ptr_t = uint64_t *>
inline void row_copy_linear_a64_load(slm_dtype *slm_ptr, uint64_t offset, abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_LINEAR_NCS_A64(32, 32b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_LINEAR_NCS_A64(512, 32b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_LINEAR_NCS_A64(1024, 32b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_LINEAR_NCS_A64(32, 16b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_LINEAR_NCS_A64(512, 16b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_LINEAR_NCS_A64(1024, 16b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_LINEAR_NCS_A64(2048, 16b)::"r"(slm_ptr), "r"(offset), "r"(abar_ptr));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_LOAD_LINEAR_A32S(row_size, basic_dtype, data_bits) \
  "async_row_copy.shared_workgroup.global.linear." #row_size ".a32s." #data_bits "." #basic_dtype \
  ".zero.L2c.L3uc.abarrier [%0], [%1], [%2], %3, %4;"

template <uint32_t row_size, typename dtype, typename slm_dtype, typename abar_ptr_t = uint64_t *>
inline void row_copy_linear_a32s_load(slm_dtype *slm_ptr, dtype *gmem_ptr, int32_t offset, uint32_t size,
                                      abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(32, uint, 32b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(512, uint, 32b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(1024, uint, 32b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(32, uint, 16b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(512, uint, 16b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(1024, uint, 16b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(2048, uint, 16b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 6) {
    if constexpr (row_size == 24) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(24, uint, 6b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(32, uint, 6b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 96) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(96, uint, 6b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(512, uint, 6b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(1024, uint, 6b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(2048, uint, 6b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 64) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(32, uint, 64b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(256, uint, 64b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(512, uint, 64b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(1024, uint, 64b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_LINEAR_A32S(2048, uint, 64b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_LOAD_LINEAR_NCS_A32S(row_size, data_bits) \
  "async_row_copy.shared_workgroup.global.ncs.linear." #row_size ".a32s." #data_bits \
  ".L2c.L3uc.abarrier [%0], [%1], [%2], %3;"

template <uint32_t row_size, typename dtype, typename slm_dtype, typename abar_ptr_t = uint64_t *>
inline void row_copy_linear_a32s_load(slm_dtype *slm_ptr, dtype *gmem_ptr, int32_t offset, abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_LINEAR_NCS_A32S(32, 32b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_LINEAR_NCS_A32S(512, 32b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_LINEAR_NCS_A32S(1024, 32b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_LINEAR_NCS_A32S(32, 16b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_LINEAR_NCS_A32S(512, 16b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_LINEAR_NCS_A32S(1024, 16b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_LINEAR_NCS_A32S(2048, 16b)::"r"(slm_ptr), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_STORE_LINEAR_A64(row_size, data_bits) \
  "async_row_copy.global.shared_workgroup.linear." #row_size ".a64." #data_bits \
  ".L2wb.L3uc.abarrier [%0], [%1], [%2], %3;"

template <uint32_t row_size, typename dtype, typename slm_dtype, typename abar_ptr_t = uint64_t *>
inline void row_copy_linear_a64_store(slm_dtype *slm_ptr, uint64_t offset, uint32_t size, abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_LINEAR_A64(32, 32b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_LINEAR_A64(512, 32b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_LINEAR_A64(1024, 32b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_LINEAR_A64(32, 16b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_LINEAR_A64(512, 16b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_LINEAR_A64(1024, 16b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 6) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_LINEAR_A64(32, 6b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 96) {
      INLINE_PISA(ROW_STORE_LINEAR_A64(96, 6b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_LINEAR_A64(512, 6b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_LINEAR_A64(1024, 6b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_STORE_LINEAR_NCS_A64(row_size, data_bits) \
  "async_row_copy.global.shared_workgroup.ncs.linear." #row_size ".a64." #data_bits \
  ".L2wb.L3uc.abarrier [%0], [%1], [%2];"

template <uint32_t row_size, typename dtype, typename slm_dtype, typename abar_ptr_t = uint64_t *>
inline void row_copy_linear_a64_store(slm_dtype *slm_ptr, uint64_t offset, abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_LINEAR_NCS_A64(32, 32b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_LINEAR_NCS_A64(512, 32b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_LINEAR_NCS_A64(1024, 32b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_LINEAR_NCS_A64(32, 16b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_LINEAR_NCS_A64(512, 16b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_LINEAR_NCS_A64(1024, 16b)::"r"(offset), "r"(slm_ptr), "r"(abar_ptr));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_STORE_LINEAR_A32S(row_size, data_bits) \
  "async_row_copy.global.shared_workgroup.linear." #row_size ".a32s." #data_bits \
  ".L2wb.L3uc.abarrier [%0], [%1], [%2], %3, %4;"

template <uint32_t row_size, typename dtype, typename slm_dtype, typename abar_ptr_t = uint64_t *>
inline void row_copy_linear_a32s_store(slm_dtype *slm_ptr, dtype *gmem_ptr, int32_t offset, uint32_t size,
                                       abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_LINEAR_A32S(32, 32b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_LINEAR_A32S(512, 32b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_LINEAR_A32S(1024, 32b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_LINEAR_A32S(32, 16b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_LINEAR_A32S(512, 16b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_LINEAR_A32S(1024, 16b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 6) {
    if constexpr (row_size == 24) {
      INLINE_PISA(ROW_STORE_LINEAR_A32S(24, 6b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_LINEAR_A32S(32, 6b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 96) {
      INLINE_PISA(ROW_STORE_LINEAR_A32S(96, 6b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_LINEAR_A32S(512, 6b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_LINEAR_A32S(1024, 6b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_STORE_LINEAR_NCS_A32S(row_size, data_bits) \
  "async_row_copy.global.shared_workgroup.ncs.linear." #row_size ".a32s." #data_bits \
  ".L2wb.L3uc.abarrier [%0], [%1], [%2], %3;"

template <uint32_t row_size, typename dtype, typename slm_dtype, typename abar_ptr_t = uint64_t *>
inline void row_copy_linear_a32s_store(slm_dtype *slm_ptr, dtype *gmem_ptr, int32_t offset, abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_LINEAR_NCS_A32S(32, 32b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_LINEAR_NCS_A32S(512, 32b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_LINEAR_NCS_A32S(1024, 32b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_LINEAR_NCS_A32S(32, 16b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_LINEAR_NCS_A32S(512, 16b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_LINEAR_NCS_A32S(1024, 16b)::"r"(gmem_ptr), "r"(slm_ptr), "r"(abar_ptr), "r"(offset));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_LOAD_TILED_A64(row_size, basic_dtype, data_bits) \
  "async_row_copy.shared_workgroup.global.tiled." #row_size ".a64." #data_bits "." #basic_dtype \
  ".zero.L2c.L3uc.abarrier %0, [%1], [%2], %3;"

template <uint32_t row_size, typename dtype, typename matrix_desc_t, typename abar_ptr_t = uint64_t *>
inline void row_copy_tiled_a64_load(matrix_desc_t mat_desc, uint64_t offset, uint32_t size, abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 8) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A64(32, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A64(64, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A64(128, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A64(256, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A64(512, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A64(1024, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A64(1984, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A64(2048, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A64(32, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A64(64, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A64(128, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A64(256, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A64(512, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A64(1024, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A64(1984, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A64(2048, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 64) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A64(32, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A64(64, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A64(128, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A64(256, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A64(512, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A64(1024, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A64(1984, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A64(2048, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A64(32, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A64(64, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A64(128, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A64(256, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A64(512, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A64(1024, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A64(1984, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A64(2048, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 6) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A64(32, uint, 6b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A64(64, uint, 6b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 96) {
      INLINE_PISA(ROW_LOAD_TILED_A64(96, uint, 6b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A64(128, uint, 6b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A64(256, uint, 6b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A64(512, uint, 6b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A64(1024, uint, 6b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A64(1984, uint, 6b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A64(2048, uint, 6b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size));
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_LOAD_TILED_NCS_A64(row_size, data_bits) \
  "async_row_copy.shared_workgroup.global.ncs.tiled." #row_size ".a64." #data_bits \
  ".L2c.L3uc.abarrier %0, [%1], " \
  "[%2];"

template <uint32_t row_size, typename dtype, typename matrix_desc_t, typename abar_ptr_t = uint64_t *>
inline void row_copy_tiled_a64_load(matrix_desc_t mat_desc, uint64_t offset, abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 8) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(32, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(64, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(128, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(256, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(512, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(1024, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(1984, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(2048, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(32, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(64, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(128, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(256, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(512, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(1024, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(1984, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(2048, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 64) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(32, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(64, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(128, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(256, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(512, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(1024, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(1984, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(2048, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(32, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(64, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(128, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(256, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(512, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(1024, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(1984, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64(2048, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_LOAD_TILED_A32S(row_size, basic_dtype, data_bits) \
  "async_row_copy.shared_workgroup.global.tiled." #row_size ".a32s." #data_bits "." #basic_dtype \
  ".zero.L2c.L3uc.abarrier %0, [%1], [%2], %3, %4;"

template <uint32_t row_size, typename dtype, typename matrix_desc_t, typename abar_ptr_t = uint64_t *>
inline void row_copy_tiled_a32s_load(matrix_desc_t mat_desc, dtype *gmem_ptr, int32_t offset, uint32_t size,
                                     abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 8) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(32, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(64, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(128, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(256, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(512, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(1024, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(1984, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(2048, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(32, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(64, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(128, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(256, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(512, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(1024, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(1984, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(2048, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 64) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(32, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(64, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(128, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(256, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(512, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(1024, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(1984, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(2048, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(32, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(64, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(128, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(256, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(512, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(1024, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(1984, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(2048, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 6) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(32, uint, 6b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(64, uint, 6b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 96) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(96, uint, 6b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(128, uint, 6b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(256, uint, 6b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(512, uint, 6b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(1024, uint, 6b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(1984, uint, 6b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A32S(2048, uint, 6b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_LOAD_TILED_NCS_A32S(row_size, data_bits) \
  "async_row_copy.shared_workgroup.global.ncs.tiled." #row_size ".a32s." #data_bits \
  ".L2c.L3uc.abarrier %0, [%1], [%2], %3;"

template <uint32_t row_size, typename dtype, typename matrix_desc_t, typename abar_ptr_t = uint64_t *>
inline void row_copy_tiled_a32s_load(matrix_desc_t mat_desc, dtype *gmem_ptr, int32_t offset, abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 8) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(32, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(64, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(128, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(256, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(512, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(1024, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(1984, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(2048, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(32, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(64, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(128, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(256, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(512, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(1024, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(1984, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(2048, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 64) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(32, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(64, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(128, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(256, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(512, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(1024, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(1984, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(2048, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(32, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(64, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(128, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(256, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(512, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(1024, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(1984, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S(2048, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_LOAD_TILED_A64_MULTICAST(row_size, basic_dtype, data_bits) \
  "async_row_copy.shared_cluster.global.tiled." #row_size ".a64." #data_bits "." #basic_dtype \
  ".zero.L2c.L3uc.abarrier %0, [%1], [%2], %3, %4;"

template <uint32_t row_size, typename dtype, typename matrix_desc_t, typename abar_ptr_t = uint64_t *>
inline void row_copy_tiled_a64_load(matrix_desc_t mat_desc, uint64_t offset, uint32_t size, abar_ptr_t abar_ptr,
                                    uint32_t wg_mask) {
  if constexpr (sizeof_bits<dtype>() == 8) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(32, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(64, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(128, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(256, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(512, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(1024, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(1984, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(2048, uint, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(32, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(64, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(128, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(256, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(512, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(1024, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(1984, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(2048, uint, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 64) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(32, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(64, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(128, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(256, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(512, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(1024, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(1984, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(2048, uint, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(32, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(64, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(128, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(256, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(512, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(1024, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(1984, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A64_MULTICAST(2048, uint, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(size),
                  "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_LOAD_TILED_NCS_A64_MULTICAST(row_size, data_bits) \
  "async_row_copy.shared_cluster.global.ncs.tiled." #row_size ".a64." #data_bits \
  ".L2c.L3uc.abarrier %0, [%1], [%2], %3;"

template <uint32_t row_size, typename dtype, typename matrix_desc_t, typename abar_ptr_t = uint64_t *>
inline void row_copy_tiled_a64_load(matrix_desc_t mat_desc, uint64_t offset, abar_ptr_t abar_ptr, uint32_t wg_mask) {
  if constexpr (sizeof_bits<dtype>() == 8) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(32, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(64, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(128, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(256, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(512, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(1024, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(1984, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(2048, 8b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(32, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(64, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(128, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(256, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(512, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(1024, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(1984, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(2048, 32b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 64) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(32, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(64, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(128, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(256, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(512, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(1024, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(1984, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(2048, 64b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(32, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(64, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(128, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(256, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(512, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(1024, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(1984, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A64_MULTICAST(2048, 16b)::"r"(mat_desc), "r"(offset), "r"(abar_ptr), "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_LOAD_TILED_A32S_MULTICAST(row_size, basic_dtype, data_bits) \
  "async_row_copy.shared_cluster.global.tiled." #row_size ".a32s." #data_bits "." #basic_dtype \
  ".zero.L2c.L3uc.abarrier %0, [%1], [%2], %3, %4, %5;"

template <uint32_t row_size, typename dtype, typename matrix_desc_t, typename abar_ptr_t = uint64_t *>
inline void row_copy_tiled_a32s_load(matrix_desc_t mat_desc, dtype *gmem_ptr, int32_t offset, uint32_t size,
                                     abar_ptr_t abar_ptr, uint32_t wg_mask) {
  if constexpr (sizeof_bits<dtype>() == 8) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(32, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(64, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(128, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(256, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(512, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(1024, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(1984, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(2048, uint, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(32, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(64, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(128, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(256, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(512, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(1024, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(1984, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(2048, uint, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 64) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(32, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(64, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(128, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(256, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(512, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(1024, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(1984, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(2048, uint, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(32, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(64, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(128, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(256, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(512, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(1024, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(1984, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_A32S_MULTICAST(2048, uint, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(size), "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_LOAD_TILED_NCS_A32S_MULTICAST(row_size, data_bits) \
  "async_row_copy.shared_cluster.global.ncs.tiled." #row_size ".a32s." #data_bits \
  ".L2c.L3uc.abarrier %0, [%1], [%2], %3, %4;"

template <uint32_t row_size, typename dtype, typename matrix_desc_t, typename abar_ptr_t = uint64_t *>
inline void row_copy_tiled_a32s_load(matrix_desc_t mat_desc, dtype *gmem_ptr, int32_t offset, abar_ptr_t abar_ptr,
                                     uint32_t wg_mask) {
  if constexpr (sizeof_bits<dtype>() == 8) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(32, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(64, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(128, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(256, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(512, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(1024, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(1984, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(2048, 8b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(32, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(64, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(128, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(256, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(512, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(1024, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(1984, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(2048, 32b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 64) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(32, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(64, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(128, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(256, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(512, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(1024, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(1984, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(2048, 64b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(32, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(64, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(128, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(256, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(512, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr), "r"(offset),
                  "r"(wg_mask));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(1024, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(wg_mask));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(1984, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(wg_mask));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_LOAD_TILED_NCS_A32S_MULTICAST(2048, 16b)::"r"(mat_desc), "r"(gmem_ptr), "r"(abar_ptr),
                  "r"(offset), "r"(wg_mask));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_STORE_TILED_A64(row_size, data_bits) \
  "async_row_copy.global.shared_workgroup.tiled." #row_size ".a64." #data_bits \
  ".L2wb.L3uc.abarrier [%0], %1, [%2], " \
  "%3;"

template <uint32_t row_size, typename dtype, typename matrix_desc_t, typename abar_ptr_t = uint64_t *>
inline void row_copy_tiled_a64_store(matrix_desc_t mat_desc, uint64_t offset, uint32_t size, abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 8) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_A64(32, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_A64(64, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_A64(128, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_A64(256, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_A64(512, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_A64(1024, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_A64(1984, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_A64(2048, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_A64(32, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_A64(64, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_A64(128, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_A64(256, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_A64(512, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_A64(1024, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_A64(1984, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_A64(2048, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 64) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_A64(32, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_A64(64, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_A64(128, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_A64(256, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_A64(512, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_A64(1024, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_A64(1984, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_A64(2048, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_A64(32, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_A64(64, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_A64(128, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_A64(256, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_A64(512, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_A64(1024, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_A64(1984, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_A64(2048, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 6) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_A64(32, 6b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_A64(64, 6b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 96) {
      INLINE_PISA(ROW_STORE_TILED_A64(96, 6b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_A64(128, 6b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_A64(256, 6b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_A64(512, 6b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_A64(1024, 6b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_A64(1984, 6b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_A64(2048, 6b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_STORE_TILED_NCS_A64(row_size, data_bits) \
  "async_row_copy.global.shared_workgroup.ncs.tiled." #row_size ".a64." #data_bits \
  ".L2wb.L3uc.abarrier [%0], %1, " \
  "[%2];"

template <uint32_t row_size, typename dtype, typename matrix_desc_t, typename abar_ptr_t = uint64_t *>
inline void row_copy_tiled_a64_store(matrix_desc_t mat_desc, uint64_t offset, abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 8) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(32, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(64, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(128, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(256, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(512, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(1024, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(1984, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(2048, 8b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(32, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(64, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(128, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(256, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(512, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(1024, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(1984, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(2048, 32b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 64) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(32, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(64, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(128, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(256, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(512, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(1024, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(1984, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(2048, 64b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(32, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(64, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(128, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(256, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(512, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(1024, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(1984, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A64(2048, 16b)::"r"(offset), "r"(mat_desc), "r"(abar_ptr));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_STORE_TILED_A32S(row_size, data_bits) \
  "async_row_copy.global.shared_workgroup.tiled." #row_size ".a32s." #data_bits \
  ".L2wb.L3uc.abarrier [%0], %1, [%2], " \
  "%3, %4;"

template <uint32_t row_size, typename dtype, typename matrix_desc_t, typename abar_ptr_t = uint64_t *>
inline void row_copy_tiled_a32s_store(matrix_desc_t mat_desc, dtype *gmem_ptr, int32_t offset, uint32_t size,
                                      abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 8) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_A32S(32, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_A32S(64, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_A32S(128, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_A32S(256, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_A32S(512, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_A32S(1024, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_A32S(1984, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_A32S(2048, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_A32S(32, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_A32S(64, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_A32S(128, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_A32S(256, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_A32S(512, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_A32S(1024, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_A32S(1984, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_A32S(2048, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 64) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_A32S(32, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_A32S(64, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_A32S(128, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_A32S(256, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_A32S(512, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_A32S(1024, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_A32S(1984, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_A32S(2048, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_A32S(32, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_A32S(64, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_A32S(128, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_A32S(256, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_A32S(512, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_A32S(1024, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_A32S(1984, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_A32S(2048, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 6) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_A32S(32, 6b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_A32S(64, 6b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 96) {
      INLINE_PISA(ROW_STORE_TILED_A32S(96, 6b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_A32S(128, 6b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_A32S(256, 6b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_A32S(512, 6b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_A32S(1024, 6b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_A32S(1984, 6b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_A32S(2048, 6b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset), "r"(size));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

#define ROW_STORE_TILED_NCS_A32S(row_size, data_bits) \
  "async_row_copy.global.shared_workgroup.ncs.tiled." #row_size ".a32s." #data_bits \
  ".L2wb.L3uc.abarrier [%0], %1, [%2], " \
  "%3;"

template <uint32_t row_size, typename dtype, typename matrix_desc_t, typename abar_ptr_t = uint64_t *>
inline void row_copy_tiled_a32s_store(matrix_desc_t mat_desc, dtype *gmem_ptr, int32_t offset, abar_ptr_t abar_ptr) {
  if constexpr (sizeof_bits<dtype>() == 8) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(32, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(64, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(128, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(256, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(512, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(1024, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(1984, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(2048, 8b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 32) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(32, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(64, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(128, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(256, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(512, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(1024, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(1984, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(2048, 32b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 64) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(32, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(64, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(128, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(256, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(512, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(1024, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(1984, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(2048, 64b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else if constexpr (sizeof_bits<dtype>() == 16) {
    if constexpr (row_size == 32) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(32, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 64) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(64, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 128) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(128, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 256) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(256, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 512) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(512, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1024) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(1024, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 1984) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(1984, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else if constexpr (row_size == 2048) {
      INLINE_PISA(ROW_STORE_TILED_NCS_A32S(2048, 16b)::"r"(gmem_ptr), "r"(mat_desc), "r"(abar_ptr), "r"(offset));
    } else {
      static_assert(false, "Unsupported row size");
    }
  } else {
    static_assert(false, "Unsupported data type");
  }
}

template <typename dtype_dst, typename dtype_src>
inline void cvt(dtype_dst &dst, const dtype_src &src) {
  if constexpr (std::is_same_v<dtype_dst, bf16> && std::is_same_v<dtype_src, float>) {
    INLINE_PISA("ftrunc.bf.f %0, %1;" : "=r"(dst) : "r"(src));
  } else if constexpr (std::is_same_v<dtype_dst, float> && std::is_same_v<dtype_src, bf16>) {
    INLINE_PISA("fext.f.bf %0, %1;" : "=r"(dst) : "r"(src));
  }
  // else if constexpr (std::is_same_v<dtype_dst, int8_t> && std::is_same_v<dtype_src, float>) {
  //   INLINE_PISA("f2i.s8.f %0, %1;" : "=r"(dst) : "r"(src));
  // }
  else {
    dst = src;
  }
}

template <typename dtype_dst, typename dtype_packed, typename dtype_src>
inline void cvt_pack(dtype_packed &dst_packed, const dtype_src &src0, const dtype_src &src1) {
  if constexpr (std::is_same_v<dtype_dst, bf16> && std::is_same_v<dtype_src, float>) {
    INLINE_PISA("ftrunc2.bfx2 %0, %1, %2;" : "=r"(dst_packed) : "r"(src0), "r"(src1));
  } else if constexpr (std::is_same_v<dtype_dst, fp16> && std::is_same_v<dtype_src, float>) {
    INLINE_PISA("ftrunc2.hfx2 %0, %1, %2;" : "=r"(dst_packed) : "r"(src0), "r"(src1));
  } else {
    static_assert(false, "Unsupported dtype");
  }
}

template <uint32_t n_elem, typename dtype_dst, typename dtype_src>
inline void copy_cvt(dtype_dst *dst_ptr, dtype_src *src_ptr) {
#pragma unroll
  for (int i = 0; i < n_elem; i++) {
    cvt(dst_ptr[i], src_ptr[i]);
  }
}

template <typename dtype_dst, uint32_t n_elem, typename dtype_packed, typename dtype_src>
inline void copy_cvt_pack(dtype_packed *dst_packed_ptr, dtype_src *src_ptr) {
  constexpr uint32_t packed_num = sizeof(dtype_packed) / sizeof(dtype_dst);
  static_assert(packed_num == 2);
#pragma unroll
  for (int i = 0; i < n_elem / packed_num; i++) {
    cvt_pack<dtype_dst>(dst_packed_ptr[i], src_ptr[packed_num * i], src_ptr[packed_num * i + 1]);
  }
}

template <uint32_t n_elem, typename dtype_dst, typename dtype_src>
inline void copy_cvt_pack(dtype_dst *dst_ptr, dtype_src *src_ptr) {
  using dtype_packed = uint32_t;
  constexpr uint32_t packed_num = sizeof(dtype_packed) / sizeof(dtype_dst);
  static_assert(packed_num == 2);
  static_assert(n_elem % packed_num == 0);
  sycl::marray<dtype_packed, n_elem / packed_num> dst_packed_array;
#pragma unroll
  for (int i = 0; i < n_elem / packed_num; i++) {
    cvt_pack<dtype_dst>(dst_packed_array[i], src_ptr[packed_num * i], src_ptr[packed_num * i + 1]);
  }
  sycl::marray<dtype_dst, n_elem> dst_array = sycl::bit_cast<sycl::marray<dtype_dst, n_elem>>(dst_packed_array);
#pragma unroll
  for (int i = 0; i < n_elem; i++) {
    dst_ptr[i] = dst_array[i];
  }
}

template <uint32_t n_elem_to_pack, typename dtype_dst, typename dtype_src>
inline void pack_data(dtype_dst *dst_ptr, const dtype_src *src_ptr) {
  constexpr uint32_t packed_num = sizeof(dtype_dst) / sizeof(dtype_src);
  constexpr uint32_t n_packed_elem = n_elem_to_pack / packed_num;
  constexpr uint32_t n_packed_elem_left = n_elem_to_pack % packed_num;
#pragma unroll
  for (int i = 0; i < n_packed_elem; i++) {
    sycl::marray<dtype_src, packed_num> src_tmp;
    sycl::marray<dtype_dst, 1> dst_tmp;
#pragma unroll
    for (int j = 0; j < packed_num; j++) {
      src_tmp[j] = src_ptr[i * packed_num + j];
    }
    dst_tmp = sycl::bit_cast<sycl::marray<dtype_dst, 1>>(src_tmp);
    dst_ptr[i] = dst_tmp[0];
  }

  if constexpr (n_packed_elem_left > 0) {
    sycl::marray<dtype_src, packed_num> src_tmp;
    sycl::marray<dtype_dst, 1> dst_tmp;
#pragma unroll
    for (int j = 0; j < n_packed_elem_left; j++) {
      src_tmp[j] = src_ptr[n_packed_elem * packed_num + j];
    }
    dst_tmp = sycl::bit_cast<sycl::marray<dtype_dst, 1>>(src_tmp);
    dst_ptr[n_packed_elem] = dst_tmp[0];
  }
}

template <uint32_t n_elem_to_unpack, typename dtype_dst, typename dtype_src>
inline void unpack_data(dtype_dst *dst_ptr, dtype_src *src_ptr) {
  constexpr uint32_t packed_num = sizeof(dtype_src) / sizeof(dtype_dst);
  constexpr uint32_t n_packed_elem = n_elem_to_unpack / packed_num;
  constexpr uint32_t n_packed_elem_left = n_elem_to_unpack % packed_num;
#pragma unroll
  for (int i = 0; i < n_packed_elem; i++) {
    sycl::marray<dtype_dst, packed_num> dst_tmp;
    sycl::marray<dtype_src, 1> src_tmp;
    src_tmp[0] = src_ptr[i];
    dst_tmp = sycl::bit_cast<sycl::marray<dtype_dst, packed_num>>(src_tmp);
#pragma unroll
    for (int j = 0; j < packed_num; j++) {
      dst_ptr[i * packed_num + j] = dst_tmp[j];
    }
  }

  if constexpr (n_packed_elem_left > 0) {
    sycl::marray<dtype_dst, packed_num> dst_tmp;
    sycl::marray<dtype_src, 1> src_tmp;
    src_tmp[0] = src_ptr[n_packed_elem];
    dst_tmp = sycl::bit_cast<sycl::marray<dtype_dst, packed_num>>(src_tmp);
#pragma unroll
    for (int j = 0; j < n_packed_elem_left; j++) {
      dst_ptr[n_packed_elem * packed_num + j] = dst_tmp[j];
    }
  }
}

template <uint32_t bytes>
struct v_ld_st {
  template <typename dtype, uint32_t vs, typename slm_dtype>
  static inline void load(sycl::marray<dtype, vs> &vdata, slm_dtype *slm_ptr) {
    static_assert(sizeof(dtype) * vs == bytes);
#pragma unroll
    for (int i = 0; i < vs; i++) {
      vdata[i] = ((dtype *)slm_ptr)[i];
    }
  }

  template <typename dtype, uint32_t vs, typename slm_dtype>
  static inline void store(const sycl::marray<dtype, vs> &vdata, slm_dtype *slm_ptr) {
    static_assert(sizeof(dtype) * vs == bytes);
#pragma unroll
    for (int i = 0; i < vs; i++) {
      ((dtype *)slm_ptr)[i] = vdata[i];
    }
  }
};

template <>
struct v_ld_st<4> {
  using vtype = vector_t<uint32_t, 1>;

  template <typename dtype, uint32_t vs, typename slm_dtype>
  static inline void load(sycl::marray<dtype, vs> &vdata, slm_dtype *slm_ptr) {
    vtype temp;
    INLINE_PISA("ld.shared.32b %0, [%1];" : "=r"(temp) : "r"(slm_ptr));
    vdata = sycl::bit_cast<sycl::marray<dtype, vs>>(temp);
  }

  template <typename dtype, uint32_t vs, typename slm_dtype>
  static inline void store(const sycl::marray<dtype, vs> &vdata, slm_dtype *slm_ptr) {
    INLINE_PISA("st.shared.32b [%0], %1;" ::"r"(slm_ptr), "r"(sycl::bit_cast<vtype>(vdata)));
  }
};

template <>
struct v_ld_st<8> {
  using vtype = vector_t<uint32_t, 2>;

  template <typename dtype, uint32_t vs, typename slm_dtype>
  static inline void load(sycl::marray<dtype, vs> &vdata, slm_dtype *slm_ptr) {
    vtype temp;
    INLINE_PISA("ld.shared.v2.32b %0, [%1];" : "=r"(temp) : "r"(slm_ptr));
    vdata = sycl::bit_cast<sycl::marray<dtype, vs>>(temp);
  }

  template <typename dtype, uint32_t vs, typename slm_dtype>
  static inline void store(const sycl::marray<dtype, vs> &vdata, slm_dtype *slm_ptr) {
    INLINE_PISA("st.shared.v2.32b [%0], %1;" ::"r"(slm_ptr), "r"(sycl::bit_cast<vtype>(vdata)));
  }
};

template <>
struct v_ld_st<16> {
  using vtype = vector_t<uint32_t, 4>;

  template <typename dtype, uint32_t vs, typename slm_dtype>
  static inline void load(sycl::marray<dtype, vs> &vdata, slm_dtype *slm_ptr) {
    vtype temp;
    INLINE_PISA("ld.shared.v4.32b %0, [%1];" : "=r"(temp) : "r"(slm_ptr));
    vdata = sycl::bit_cast<sycl::marray<dtype, vs>>(temp);
  }

  template <typename dtype, uint32_t vs, typename slm_dtype>
  static inline void store(const sycl::marray<dtype, vs> &vdata, slm_dtype *slm_ptr) {
    INLINE_PISA("st.shared.v4.32b [%0], %1;" ::"r"(slm_ptr), "r"(sycl::bit_cast<vtype>(vdata)));
  }
};

template <>
struct v_ld_st<32> {
  using vtype = vector_t<uint32_t, 8>;

  template <typename dtype, uint32_t vs, typename slm_dtype>
  static inline void load(sycl::marray<dtype, vs> &vdata, slm_dtype *slm_ptr) {
    vtype temp;
    INLINE_PISA("ld.shared.v8.32b %0, [%1];" : "=r"(temp) : "r"(slm_ptr));
    vdata = sycl::bit_cast<sycl::marray<dtype, vs>>(temp);
  }

  template <typename dtype, uint32_t vs, typename slm_dtype>
  static inline void store(const sycl::marray<dtype, vs> &vdata, slm_dtype *slm_ptr) {
    INLINE_PISA("st.shared.v8.32b [%0], %1;" ::"r"(slm_ptr), "r"(sycl::bit_cast<vtype>(vdata)));
  }
};

template <uint32_t vs, typename slm_dtype, typename reg_dtype>
inline void slm_vstore(slm_dtype *slm_ptr, const reg_dtype *reg_ptr) {
  using v_engine = v_ld_st<vs * sizeof(reg_dtype)>;
  using reg_utype = typename uint_type<reg_dtype>::type;
  sycl::marray<reg_utype, vs> src_tmp;
#pragma unroll
  for (int i = 0; i < vs; i++) {
    src_tmp[i] = ((reg_utype *)reg_ptr)[i];
  }
  v_engine::template store<reg_utype, vs>(src_tmp, slm_ptr);
}

template <uint32_t vs, typename slm_dtype, typename reg_dtype>
inline void slm_vload(reg_dtype *reg_ptr, const slm_dtype *slm_ptr) {
  using v_engine = v_ld_st<vs * sizeof(reg_dtype)>;
  using reg_utype = typename uint_type<reg_dtype>::type;
  sycl::marray<reg_utype, vs> dst_tmp;
  v_engine::template load<reg_utype, vs>(dst_tmp, slm_ptr);
#pragma unroll
  for (int i = 0; i < vs; i++) {
    ((reg_utype *)reg_ptr)[i] = dst_tmp[i];
  }
}

template <slm_matrix_type slm_mat_type, typename dtype, uint32_t vs, uint32_t row_size>
uint32_t cm_get_slm_offset(uint32_t idx_x, uint32_t idx_y) {
  if constexpr (slm_mat_type == slm_matrix_type::type1) {
    constexpr uint32_t cm_bytes_x = 32;
    constexpr uint32_t cm_size_y = 32;
    constexpr uint32_t cm_byte_size = 1024;
    constexpr uint32_t cm_size_x = cm_bytes_x / sizeof(dtype);
    static_assert((cm_size_x % vs) == 0);
    constexpr uint32_t num_cm_per_row = row_size / cm_size_x;

    constexpr uint32_t bank_bytes = 64;
    constexpr uint32_t swizzle_bytes = bank_bytes / 2;
    constexpr uint32_t bank_num = 4;
    constexpr uint32_t cm_rows_per_bank = bank_bytes / cm_bytes_x;
    constexpr uint32_t cm_rows_per_mma = cm_size_y / bank_num;

    uint32_t cm_idx_y = idx_y / cm_size_y;
    uint32_t cm_idx_x = idx_x / cm_size_x;
    auto slm_cm_base = cm_idx_y * num_cm_per_row * cm_byte_size + cm_idx_x * cm_byte_size;

    uint32_t idx_y_within_cm = idx_y % cm_size_y;
    uint32_t idx_x_within_cm = idx_x % cm_size_x;
    auto within_cm_offset = (idx_y_within_cm / cm_rows_per_mma) * bank_bytes +
        (idx_y_within_cm % cm_rows_per_mma) / cm_rows_per_bank * bank_num * bank_bytes +
        (idx_y_within_cm % cm_rows_per_bank) * cm_bytes_x + idx_x_within_cm * sizeof(dtype);

    auto within_cm_offset_swizzle = within_cm_offset ^ ((cm_idx_x & 1u) << 5);

    auto slm_offset = slm_cm_base + within_cm_offset_swizzle;
    return slm_offset;
  } else if constexpr (slm_mat_type == slm_matrix_type::type2) {
    constexpr uint32_t cm_bytes_y = 32;
    constexpr uint32_t cm_size_x = 32;
    constexpr uint32_t cm_byte_size = 1024;
    constexpr uint32_t cm_size_y = cm_bytes_y / sizeof(dtype);
    constexpr uint32_t num_cm_per_row = row_size / cm_size_x;

    constexpr uint32_t bank_bytes = 64;
    constexpr uint32_t swizzle_bytes = bank_bytes / 2;
    constexpr uint32_t bank_num = 4;
    constexpr uint32_t cols_per_bank = cm_size_x / bank_num;
    constexpr uint32_t rows_per_sub_bank = bank_bytes / (cols_per_bank * sizeof(dtype));
    constexpr uint32_t within_cm_stride = bank_bytes * bank_num;

    uint32_t cm_idx_y = idx_y / cm_size_y;
    uint32_t cm_idx_x = idx_x / cm_size_x;
    auto slm_cm_base = cm_idx_y * num_cm_per_row * cm_byte_size + cm_idx_x * cm_byte_size;

    uint32_t idx_y_within_cm = idx_y % cm_size_y;
    uint32_t idx_x_within_cm = idx_x % cm_size_x;
    uint32_t bank_id = idx_x_within_cm / cols_per_bank;
    uint32_t sub_bank_id = idx_y_within_cm / rows_per_sub_bank;

    uint32_t idx_x_within_sub_bank = idx_x_within_cm % cols_per_bank;
    uint32_t idx_y_within_sub_bank = idx_y_within_cm % rows_per_sub_bank;

    auto bank_start_offset = sub_bank_id * bank_bytes * bank_num + bank_id * bank_bytes;
    auto within_bank_swizzle_offset = (idx_y_within_sub_bank * cols_per_bank * sizeof(dtype) +
                                       idx_x_within_sub_bank * sizeof(dtype) + (cm_idx_x & 1u) * swizzle_bytes) %
        bank_bytes;

    auto slm_offset = slm_cm_base + bank_start_offset + within_bank_swizzle_offset;
    return slm_offset;
  } else if constexpr (slm_mat_type == slm_matrix_type::type3) {
    constexpr uint32_t cm_bytes_y = 8;
    constexpr uint32_t cm_size_x = 32;
    constexpr uint32_t cm_byte_size = 256;
    constexpr uint32_t cm_size_y = cm_bytes_y / sizeof(dtype);
    constexpr uint32_t num_cm_per_row = row_size / cm_size_x;

    constexpr uint32_t bank_bytes = 64;
    constexpr uint32_t swizzle_bytes = bank_bytes / 2;
    constexpr uint32_t bank_num = 4;
    constexpr uint32_t cols_per_bank = cm_size_x / bank_num;
    constexpr uint32_t rows_per_sub_bank = bank_bytes / (cols_per_bank * sizeof(dtype));
    constexpr uint32_t within_cm_stride = bank_bytes * bank_num;

    uint32_t cm_idx_y = idx_y / cm_size_y;
    uint32_t cm_idx_x = idx_x / cm_size_x;
    auto slm_cm_base = cm_idx_y * num_cm_per_row * cm_byte_size + cm_idx_x * cm_byte_size;

    uint32_t idx_y_within_cm = idx_y % cm_size_y;
    uint32_t idx_x_within_cm = idx_x % cm_size_x;
    uint32_t bank_id = idx_x_within_cm / cols_per_bank;
    uint32_t sub_bank_id = idx_y_within_cm / rows_per_sub_bank;

    uint32_t idx_x_within_sub_bank = idx_x_within_cm % cols_per_bank;
    uint32_t idx_y_within_sub_bank = idx_y_within_cm % rows_per_sub_bank;

    auto bank_start_offset = sub_bank_id * bank_bytes * bank_num + bank_id * bank_bytes;
    auto within_bank_swizzle_offset = (idx_y_within_sub_bank * cols_per_bank * sizeof(dtype) +
                                       idx_x_within_sub_bank * sizeof(dtype) + (cm_idx_x & 1u) * swizzle_bytes) %
        bank_bytes;

    auto slm_offset = slm_cm_base + bank_start_offset + within_bank_swizzle_offset;
    return slm_offset;
  } else {
    static_assert(false, "Unsupported slm matrix type");
  }
}

//load vs elements based on the 2d coord. (idx_x % (32/sizeof(dtype))) + vs should within one CM.
template <typename dtype, uint32_t vs, uint32_t row_size, typename slm_dtype,
          slm_matrix_type slm_mat_type = slm_matrix_type::type1>
inline void cm_vload(dtype *data_ptr, slm_dtype *slm_ptr, uint32_t idx_x, uint32_t idx_y, uint32_t smem_offset = 0) {
  uint32_t slm_offset = cm_get_slm_offset<slm_mat_type, dtype, vs, row_size>(idx_x, idx_y);
  // copy_cvt<vs>(data_ptr, (dtype*)slm_offset);
  slm_vload<vs>(data_ptr, slm_ptr + slm_offset + smem_offset);
}

template <typename dtype, uint32_t vs, typename mat_desc_t = uint32_t>
inline void cm_vrow_load(dtype *data_ptr, const mat_desc_t &mat_desc, const sycl::marray<uint16_t, 2> &pos) {
  static_assert(((vs & (vs - 1)) == 0), "vs needs to be power of 2");
  constexpr uint32_t dbits = sizeof_bits<dtype>();
  if constexpr (dbits < 8) {
    static_assert(vs <= 32, "for sub-byte type, the maximum vs is 32");
  } else {
    static_assert(vs * dbits <= 256, "the maximum bits per load is 256");
  }
  constexpr uint32_t dbits_u32 = sizeof(uint32_t) * BITS_PER_BYTE;
  constexpr uint32_t vs_u32 = (dbits * vs + dbits_u32 - 1) / dbits_u32;
  using vtype = vector_t<uint32_t, vs_u32>;
  vtype temp;
  if constexpr (dbits == 64) {
    if constexpr (vs == 1) {
      INLINE_PISA("ld_matrix.vl1.vrow.64b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("ld_matrix.vl2.vrow.64b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("ld_matrix.vl4.vrow.64b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else {
      static_assert(false, "Unsupported vector size for 64-bit data");
    }
  } else if constexpr (dbits == 32) {
    if constexpr (vs == 1) {
      INLINE_PISA("ld_matrix.vl1.vrow.32b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("ld_matrix.vl2.vrow.32b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("ld_matrix.vl4.vrow.32b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("ld_matrix.vl8.vrow.32b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else {
      static_assert(false, "Unsupported vector size for 32-bit data");
    }
  } else if constexpr (dbits == 16) {
    if constexpr (vs == 1) {
      INLINE_PISA("ld_matrix.vl1.vrow.16b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("ld_matrix.vl2.vrow.16b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("ld_matrix.vl4.vrow.16b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("ld_matrix.vl8.vrow.16b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 16) {
      INLINE_PISA("ld_matrix.vl16.vrow.16b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else {
      static_assert(false, "Unsupported vector size for 16-bit data");
    }
  } else if constexpr (dbits == 8) {
    if constexpr (vs == 1) {
      INLINE_PISA("ld_matrix.vl1.vrow.8b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("ld_matrix.vl2.vrow.8b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("ld_matrix.vl4.vrow.8b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("ld_matrix.vl8.vrow.8b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 16) {
      INLINE_PISA("ld_matrix.vl16.vrow.8b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 32) {
      INLINE_PISA("ld_matrix.vl32.vrow.8b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else {
      static_assert(false, "Unsupported vector size for 8-bit data");
    }
  } else if constexpr (dbits == 4) {
    if constexpr (vs == 1) {
      INLINE_PISA("ld_matrix.vl1.vrow.4b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("ld_matrix.vl2.vrow.4b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("ld_matrix.vl4.vrow.4b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("ld_matrix.vl8.vrow.4b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 16) {
      INLINE_PISA("ld_matrix.vl16.vrow.4b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 32) {
      INLINE_PISA("ld_matrix.vl32.vrow.4b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else {
      static_assert(false, "Unsupported vector size for 4-bit data");
    }

  } else {
    static_assert(false, "Unsupported data size");
  }

  constexpr uint32_t vs_u8 = (dbits * vs + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
  // for sub-byte type, will cvt to uint8_t first.
  using dtype_reg = std::conditional_t<(dbits < 8), uint8_t, dtype>;
  // to make it u8/element aligned
  constexpr uint32_t vs_dst = (dbits < 8) ? vs_u8 : vs;
  // to make it u32 aligned
  constexpr uint32_t vs_reg = vs_u32 * dbits_u32 / sizeof_bits<dtype_reg>();
  sycl::marray<dtype_reg, vs_reg> dst = sycl::bit_cast<sycl::marray<dtype_reg, vs_reg>>(temp);
  dtype_reg *data_reg_ptr = reinterpret_cast<dtype_reg *>(data_ptr);
#pragma unroll
  for (uint32_t i = 0; i < vs_dst; i++) {
    data_reg_ptr[i] = dst[i];
  }
}

template <typename dtype, uint32_t vs, typename mat_desc_t = uint32_t>
inline void cm_vrow_store(const mat_desc_t &mat_desc, dtype *data_ptr, const sycl::marray<uint16_t, 2> &pos) {
  static_assert(((vs & (vs - 1)) == 0), "vs needs to be power of 2");
  constexpr uint32_t dbits = sizeof_bits<dtype>();
  if constexpr (dbits < 8) {
    static_assert(vs <= 32, "for sub-byte type, the maximum vs is 32");
  } else {
    static_assert(vs * dbits <= 256, "the maximum bits per load is 256");
  }
  constexpr uint32_t dbits_u32 = sizeof(uint32_t) * BITS_PER_BYTE;
  constexpr uint32_t vs_u32 = (dbits * vs + dbits_u32 - 1) / dbits_u32;
  constexpr uint32_t vs_u8 = (dbits * vs + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
  using vtype = vector_t<uint32_t, vs_u32>;
  // for sub-byte type, will cvt to uint8_t first.
  using dtype_reg = std::conditional_t<(dbits < 8), uint8_t, dtype>;
  // to make it u8/element aligned
  constexpr uint32_t vs_src = (dbits < 8) ? vs_u8 : vs;
  // to make it u32 aligned
  constexpr uint32_t vs_reg = vs_u32 * dbits_u32 / sizeof_bits<dtype_reg>();
  dtype_reg *data_reg_ptr = reinterpret_cast<dtype_reg *>(data_ptr);
  sycl::marray<dtype_reg, vs_reg> src;
#pragma unroll
  for (uint32_t i = 0; i < vs_src; i++) {
    src[i] = data_reg_ptr[i];
  }
  if constexpr (dbits == 64) {
    if constexpr (vs == 1) {
      INLINE_PISA("st_matrix.vl1.vrow.64b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("st_matrix.vl2.vrow.64b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("st_matrix.vl4.vrow.64b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(false, "Unsupported vector size for 64-bit data");
    }
  } else if constexpr (dbits == 32) {
    if constexpr (vs == 1) {
      INLINE_PISA("st_matrix.vl1.vrow.32b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("st_matrix.vl2.vrow.32b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("st_matrix.vl4.vrow.32b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("st_matrix.vl8.vrow.32b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(false, "Unsupported vector size for 32-bit data");
    }
  } else if constexpr (dbits == 16) {
    if constexpr (vs == 1) {
      INLINE_PISA("st_matrix.vl1.vrow.16b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("st_matrix.vl2.vrow.16b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("st_matrix.vl4.vrow.16b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("st_matrix.vl8.vrow.16b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 16) {
      INLINE_PISA("st_matrix.vl16.vrow.16b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(false, "Unsupported vector size for 16-bit data");
    }
  } else if constexpr (dbits == 8) {
    if constexpr (vs == 1) {
      INLINE_PISA("st_matrix.vl1.vrow.8b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("st_matrix.vl2.vrow.8b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("st_matrix.vl4.vrow.8b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("st_matrix.vl8.vrow.8b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 16) {
      INLINE_PISA("st_matrix.vl16.vrow.8b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 32) {
      INLINE_PISA("st_matrix.vl32.vrow.8b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(false, "Unsupported vector size for 8-bit data");
    }
  } else if constexpr (dbits == 4) {
    if constexpr (vs == 1) {
      INLINE_PISA("st_matrix.vl1.vrow.4b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("st_matrix.vl2.vrow.4b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("st_matrix.vl4.vrow.4b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("st_matrix.vl8.vrow.4b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 16) {
      INLINE_PISA("st_matrix.vl16.vrow.4b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 32) {
      INLINE_PISA("st_matrix.vl32.vrow.4b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(false, "Unsupported vector size for 8-bit data");
    }
  } else {
    static_assert(false, "Unsupported data size");
  }
}

template <typename dtype, uint32_t vs, typename mat_desc_t = uint32_t>
inline void cm_vrow_load_unordered(dtype *data_ptr, const mat_desc_t &mat_desc, const sycl::marray<uint16_t, 2> &pos) {
  static_assert(((vs & (vs - 1)) == 0), "vs needs to be power of 2");
  constexpr uint32_t dbits = sizeof_bits<dtype>();
  if constexpr (dbits < 8) {
    static_assert(vs <= 32, "for sub-byte type, the maximum vs is 32");
  } else {
    static_assert(vs * dbits <= 256, "the maximum bits per load is 256");
  }
  constexpr uint32_t dbits_u32 = sizeof(uint32_t) * BITS_PER_BYTE;
  constexpr uint32_t vs_u32 = (dbits * vs + dbits_u32 - 1) / dbits_u32;
  using vtype = vector_t<uint32_t, vs_u32>;
  vtype temp;
  if constexpr (dbits == 64) {
    if constexpr (vs == 1) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl1.cooprow.64b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl2.cooprow.64b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl4.cooprow.64b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else {
      static_assert(false, "Unsupported vector size for 64-bit data");
    }
  } else if constexpr (dbits == 32) {
    if constexpr (vs == 1) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl1.cooprow.32b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl2.cooprow.32b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl4.cooprow.32b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl8.cooprow.32b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else {
      static_assert(false, "Unsupported vector size for 32-bit data");
    }
  } else if constexpr (dbits == 16) {
    if constexpr (vs == 1) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl1.cooprow.16b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl2.cooprow.16b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl4.cooprow.16b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl8.cooprow.16b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 16) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl16.cooprow.16b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else {
      static_assert(false, "Unsupported vector size for 16-bit data");
    }
  } else if constexpr (dbits == 8) {
    if constexpr (vs == 1) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl1.cooprow.8b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl2.cooprow.8b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl4.cooprow.8b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl8.cooprow.8b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 16) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl16.cooprow.8b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 32) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl32.cooprow.8b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else {
      static_assert(false, "Unsupported vector size for 8-bit data");
    }
  } else if constexpr (dbits == 4) {
    if constexpr (vs == 1) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl1.cooprow.4b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl2.cooprow.4b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl4.cooprow.4b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl8.cooprow.4b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 16) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl16.cooprow.4b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 32) {
      INLINE_PISA("ld_matrix.unordered.al1.as1.arow.vl32.cooprow.4b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else {
      static_assert(false, "Unsupported vector size for 4-bit data");
    }

  } else {
    static_assert(false, "Unsupported data size");
  }

  constexpr uint32_t vs_u8 = (dbits * vs + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
  // for sub-byte type, will cvt to uint8_t first.
  using dtype_reg = std::conditional_t<(dbits < 8), uint8_t, dtype>;
  // to make it u8/element aligned
  constexpr uint32_t vs_dst = (dbits < 8) ? vs_u8 : vs;
  // to make it u32 aligned
  constexpr uint32_t vs_reg = vs_u32 * dbits_u32 / sizeof_bits<dtype_reg>();
  sycl::marray<dtype_reg, vs_reg> dst = sycl::bit_cast<sycl::marray<dtype_reg, vs_reg>>(temp);
  dtype_reg *data_reg_ptr = reinterpret_cast<dtype_reg *>(data_ptr);
#pragma unroll
  for (uint32_t i = 0; i < vs_dst; i++) {
    data_reg_ptr[i] = dst[i];
  }
}

template <typename dtype, uint32_t vs, typename mat_desc_t = uint32_t>
inline void cm_vrow_store_unordered(const mat_desc_t &mat_desc, dtype *data_ptr, const sycl::marray<uint16_t, 2> &pos) {
  static_assert(((vs & (vs - 1)) == 0), "vs needs to be power of 2");
  constexpr uint32_t dbits = sizeof_bits<dtype>();
  if constexpr (dbits < 8) {
    static_assert(vs <= 32, "for sub-byte type, the maximum vs is 32");
  } else {
    static_assert(vs * dbits <= 256, "the maximum bits per load is 256");
  }
  constexpr uint32_t dbits_u32 = sizeof(uint32_t) * BITS_PER_BYTE;
  constexpr uint32_t vs_u32 = (dbits * vs + dbits_u32 - 1) / dbits_u32;
  constexpr uint32_t vs_u8 = (dbits * vs + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
  using vtype = vector_t<uint32_t, vs_u32>;
  // for sub-byte type, will cvt to uint8_t first.
  using dtype_reg = std::conditional_t<(dbits < 8), uint8_t, dtype>;
  // to make it u8/element aligned
  constexpr uint32_t vs_src = (dbits < 8) ? vs_u8 : vs;
  // to make it u32 aligned
  constexpr uint32_t vs_reg = vs_u32 * dbits_u32 / sizeof_bits<dtype_reg>();
  dtype_reg *data_reg_ptr = reinterpret_cast<dtype_reg *>(data_ptr);
  sycl::marray<dtype_reg, vs_reg> src;
#pragma unroll
  for (uint32_t i = 0; i < vs_src; i++) {
    src[i] = data_reg_ptr[i];
  }
  if constexpr (dbits == 64) {
    if constexpr (vs == 1) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl1.cooprow.64b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl2.cooprow.64b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl4.cooprow.64b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(false, "Unsupported vector size for 64-bit data");
    }
  } else if constexpr (dbits == 32) {
    if constexpr (vs == 1) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl1.cooprow.32b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl2.cooprow.32b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl4.cooprow.32b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl8.cooprow.32b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(false, "Unsupported vector size for 32-bit data");
    }
  } else if constexpr (dbits == 16) {
    if constexpr (vs == 1) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl1.cooprow.16b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl2.cooprow.16b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl4.cooprow.16b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl8.cooprow.16b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 16) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl16.cooprow.16b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(false, "Unsupported vector size for 16-bit data");
    }
  } else if constexpr (dbits == 8) {
    if constexpr (vs == 1) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl1.cooprow.8b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl2.cooprow.8b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl4.cooprow.8b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl8.cooprow.8b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 16) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl16.cooprow.8b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 32) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl32.cooprow.8b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(false, "Unsupported vector size for 8-bit data");
    }
  } else if constexpr (dbits == 4) {
    if constexpr (vs == 1) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl1.cooprow.4b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 2) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl2.cooprow.4b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl4.cooprow.4b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 8) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl8.cooprow.4b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 16) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl16.cooprow.4b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 32) {
      INLINE_PISA("st_matrix.unordered.al1.as1.arow.vl32.cooprow.4b %0, %1, %2;" ::"r"(mat_desc),
                  "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)), "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(false, "Unsupported vector size for 8-bit data");
    }
  } else {
    static_assert(false, "Unsupported data size");
  }
}

template <typename dtype, uint32_t vs, typename mat_desc_t = uint32_t>
inline void cm_vcol_load(dtype *data_ptr, const mat_desc_t &mat_desc, const sycl::marray<uint16_t, 2> &pos) {
  static_assert(((vs & (vs - 1)) == 0), "vs needs to be power of 2");
  constexpr uint32_t dbits = sizeof_bits<dtype>();
  if constexpr (dbits < 8) {
    static_assert(vs <= 16, "for sub-byte type, the maximum vs is 16");
  } else {
    static_assert(vs * dbits <= 256, "the maximum bits per load is 256");
  }
  constexpr uint32_t dbits_u32 = sizeof(uint32_t) * BITS_PER_BYTE;
  constexpr uint32_t vs_u32 = (dbits * vs + dbits_u32 - 1) / dbits_u32;
  constexpr uint32_t vs_dst = vs_u32 * dbits_u32 / dbits;
  using vtype = vector_t<uint32_t, vs_u32>;
  vtype temp;
  if constexpr (dbits == 64) {
    if constexpr (vs == 2) {
      INLINE_PISA("ld_matrix.vl2.vcol.64b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("ld_matrix.vl4.vcol.64b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else {
      static_assert(false, "Unsupported vector size for 64-bit data");
    }
  } else if constexpr (dbits == 32) {
    if constexpr (vs == 2) {
      INLINE_PISA("ld_matrix.vl2.vcol.32b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("ld_matrix.vl4.vcol.32b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else {
      static_assert(false, "Unsupported vector size for 32-bit data");
    }
  } else if constexpr (dbits == 16) {
    if constexpr (vs == 2) {
      INLINE_PISA("ld_matrix.vl2.vcol.16b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("ld_matrix.vl4.vcol.16b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else {
      static_assert(false, "Unsupported vector size for 16-bit data");
    }
  } else if constexpr (dbits == 8) {
    if constexpr (vs == 2) {
      INLINE_PISA("ld_matrix.vl2.vcol.8b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("ld_matrix.vl4.vcol.8b %0, %1, %2;"
                  : "=r"(temp)
                  : "r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)));
    } else {
      static_assert(false, "Unsupported vector size for 8-bit data");
    }
  } else {
    static_assert(false, "Unsupported data size");
  }
  sycl::marray<dtype, vs_dst> dst = sycl::bit_cast<sycl::marray<dtype, vs_dst>>(temp);
#pragma unroll
  for (uint32_t i = 0; i < vs; i++) {
    data_ptr[i] = dst[i];
  }
}

template <typename dtype, uint32_t vs, typename mat_desc_t = uint32_t>
inline void cm_vcol_store(const mat_desc_t &mat_desc, dtype *data_ptr, const sycl::marray<uint16_t, 2> &pos) {
  static_assert(((vs & (vs - 1)) == 0), "vs needs to be power of 2");
  constexpr uint32_t dbits = sizeof_bits<dtype>();
  if constexpr (dbits < 8) {
    static_assert(vs <= 32, "for sub-byte type, the maximum vs is 32");
  } else {
    static_assert(vs * dbits <= 256, "the maximum bits per load is 256");
  }
  constexpr uint32_t dbits_u32 = sizeof(uint32_t) * BITS_PER_BYTE;
  constexpr uint32_t vs_u32 = (dbits * vs + dbits_u32 - 1) / dbits_u32;
  constexpr uint32_t vs_src = vs_u32 * dbits_u32 / dbits;
  using vtype = vector_t<uint32_t, vs_u32>;
  sycl::marray<dtype, vs_src> src;
#pragma unroll
  for (uint32_t i = 0; i < vs; i++) {
    src[i] = data_ptr[i];
  }
  if constexpr (dbits == 64) {
    if constexpr (vs == 2) {
      INLINE_PISA("st_matrix.vl2.vcol.64b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("st_matrix.vl4.vcol.64b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(false, "Unsupported vector size for 64-bit data");
    }
  } else if constexpr (dbits == 32) {
    if constexpr (vs == 2) {
      INLINE_PISA("st_matrix.vl2.vcol.32b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("st_matrix.vl4.vcol.32b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(false, "Unsupported vector size for 32-bit data");
    }
  } else if constexpr (dbits == 16) {
    if constexpr (vs == 2) {
      INLINE_PISA("st_matrix.vl2.vcol.16b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("st_matrix.vl4.vcol.16b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(false, "Unsupported vector size for 16-bit data");
    }
  } else if constexpr (dbits == 8) {
    if constexpr (vs == 2) {
      INLINE_PISA("st_matrix.vl2.vcol.8b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (vs == 4) {
      INLINE_PISA("st_matrix.vl4.vcol.8b %0, %1, %2;" ::"r"(mat_desc), "r"(sycl::bit_cast<vector_t<uint16_t, 2>>(pos)),
                  "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(false, "Unsupported vector size for 8-bit data");
    }
  } else {
    static_assert(false, "Unsupported data size");
  }
}

//store vs elements based on the 2d coord. (idx_x % (32/sizeof(dtype))) + vs should within one CM.
template <typename dtype, uint32_t vs, uint32_t row_size, typename slm_dtype,
          slm_matrix_type slm_mat_type = slm_matrix_type::type1>
inline void cm_vstore(slm_dtype *slm_ptr, dtype *data_ptr, uint32_t idx_x, uint32_t idx_y, uint32_t smem_offset = 0) {
  uint32_t slm_offset = cm_get_slm_offset<slm_mat_type, dtype, vs, row_size>(idx_x, idx_y);
  // copy_cvt<vs>((dtype*)slm_offset, data_ptr);
  slm_vstore<vs>(slm_ptr + slm_offset + smem_offset, data_ptr);
}

#define unary_packed_op(func, dtype_math, ret, src0, packed_num) \
  if constexpr (std::is_same_v<dtype_math, fp16>) { \
    if constexpr (packed_num == 1) { \
      INLINE_PISA(#func ".hf %0, %1;" : "=r"(ret) : "r"(src0)); \
    } else if constexpr (packed_num == 2) { \
      INLINE_PISA(#func ".hfx2 %0, %1;" : "=r"(ret) : "r"(src0)); \
    } \
  } else if constexpr (std::is_same_v<dtype_math, bf16>) { \
    if constexpr (packed_num == 1) { \
      INLINE_PISA(#func ".bf %0, %1;" : "=r"(ret) : "r"(src0)); \
    } else if constexpr (packed_num == 2) { \
      INLINE_PISA(#func ".bfx2 %0, %1;" : "=r"(ret) : "r"(src0)); \
    } \
  } else if constexpr (std::is_same_v<dtype_math, float>) { \
    INLINE_PISA(#func ".f %0, %1;" : "=r"(ret) : "r"(src0)); \
  } else { \
    static_assert(sizeof(dtype_math) == 0, "unsupported case"); \
  }

#define binary_packed_op(func, dtype_math, ret, src0, src1, packed_num) \
  if constexpr (std::is_same_v<dtype_math, fp16>) { \
    if constexpr (packed_num == 1) { \
      INLINE_PISA(#func ".hf %0, %1, %2;" : "=r"(ret) : "r"(src0), "r"(src1)); \
    } else if constexpr (packed_num == 2) { \
      INLINE_PISA(#func ".hfx2 %0, %1, %2;" : "=r"(ret) : "r"(src0), "r"(src1)); \
    } \
  } else if constexpr (std::is_same_v<dtype_math, bf16>) { \
    if constexpr (packed_num == 1) { \
      INLINE_PISA(#func ".bf %0, %1, %2;" : "=r"(ret) : "r"(src0), "r"(src1)); \
    } else if constexpr (packed_num == 2) { \
      INLINE_PISA(#func ".bfx2 %0, %1, %2;" : "=r"(ret) : "r"(src0), "r"(src1)); \
    } \
  } else if constexpr (std::is_same_v<dtype_math, float>) { \
    INLINE_PISA(#func ".f %0, %1, %2;" : "=r"(ret) : "r"(src0), "r"(src1)); \
  } else { \
    static_assert(sizeof(dtype_math) == 0, "unsupported case"); \
  }

#define ternary_packed_op(func, dtype_math, ret, src0, src1, src2, packed_num) \
  if constexpr (std::is_same_v<dtype_math, fp16>) { \
    if constexpr (packed_num == 1) { \
      INLINE_PISA(#func ".hf %0, %1, %2, %3;" : "=r"(ret) : "r"(src0), "r"(src1), "r"(src2)); \
    } else if constexpr (packed_num == 2) { \
      INLINE_PISA(#func ".hfx2 %0, %1, %2, %3;" : "=r"(ret) : "r"(src0), "r"(src1), "r"(src2)); \
    } \
  } else if constexpr (std::is_same_v<dtype_math, bf16>) { \
    if constexpr (packed_num == 1) { \
      INLINE_PISA(#func ".bf %0, %1, %2, %3;" : "=r"(ret) : "r"(src0), "r"(src1), "r"(src2)); \
    } else if constexpr (packed_num == 2) { \
      INLINE_PISA(#func ".bfx2 %0, %1, %2, %3;" : "=r"(ret) : "r"(src0), "r"(src1), "r"(src2)); \
    } \
  } else if constexpr (std::is_same_v<dtype_math, float>) { \
    INLINE_PISA(#func ".f %0, %1, %2, %3;" : "=r"(ret) : "r"(src0), "r"(src1), "r"(src2)); \
  } else { \
    static_assert(sizeof(dtype_math) == 0, "unsupported case"); \
  }

template <typename dtype_math, typename dtype_reg>
inline dtype_reg packed_fmax(const dtype_reg &src0, const dtype_reg &src1) {
  constexpr uint32_t packed_num = sizeof(dtype_reg) / sizeof(dtype_math);
  dtype_reg ret;
  binary_packed_op(fmax, dtype_math, ret, src0, src1, packed_num);
  return ret;
}

template <typename dtype_math, typename dtype_reg>
inline dtype_reg packed_fcmp(const dtype_reg &src0, const dtype_reg &src1) {
  dtype_reg ret;
  INLINE_PISA("fcmp.gt.bfx2 %0, %1, %2;" : "=r"(ret) : "r"(src0), "r"(src1));
  return ret;
}

template <typename dtype_math, typename dtype_reg>
inline dtype_reg packed_fadd(const dtype_reg &src0, const dtype_reg &src1) {
  constexpr uint32_t packed_num = sizeof(dtype_reg) / sizeof(dtype_math);
  dtype_reg ret;
  binary_packed_op(fadd, dtype_math, ret, src0, src1, packed_num);
  return ret;
}

template <typename dtype_math, typename dtype_reg>
inline dtype_reg packed_fsub(const dtype_reg &src0, const dtype_reg &src1) {
  constexpr uint32_t packed_num = sizeof(dtype_reg) / sizeof(dtype_math);
  dtype_reg ret;
  binary_packed_op(fsub, dtype_math, ret, src0, src1, packed_num);
  return ret;
}

template <typename dtype_math, typename dtype_reg>
inline dtype_reg packed_fmul(const dtype_reg &src0, const dtype_reg &src1) {
  constexpr uint32_t packed_num = sizeof(dtype_reg) / sizeof(dtype_math);
  dtype_reg ret;
  binary_packed_op(fmul, dtype_math, ret, src0, src1, packed_num);
  return ret;
}

template <typename dtype_math, typename dtype_reg>
inline dtype_reg packed_fmad(const dtype_reg &src0, const dtype_reg &src1, const dtype_reg &src2) {
  constexpr uint32_t packed_num = sizeof(dtype_reg) / sizeof(dtype_math);
  dtype_reg ret;
  ternary_packed_op(fmad, dtype_math, ret, src0, src1, src2, packed_num) return ret;
}

template <typename dtype_math, typename dtype_reg>
inline dtype_reg packed_fexp2(const dtype_reg &src0) {
  constexpr uint32_t packed_num = sizeof(dtype_reg) / sizeof(dtype_math);
  dtype_reg ret;
  unary_packed_op(fexp2, dtype_math, ret, src0, packed_num) return ret;
}

template <typename dtype_math, typename dtype_reg>
inline dtype_reg packed_fabs(const dtype_reg &src0) {
  constexpr uint32_t packed_num = sizeof(dtype_reg) / sizeof(dtype_math);
  dtype_reg ret;
  unary_packed_op(fabs, dtype_math, ret, src0, packed_num) return ret;
}

template <typename dtype_math, uint32_t mask = 0xffffffff, typename dtype_reg>
inline void packed_fred_max(dtype_reg &src0) {
  constexpr uint32_t packed_num = sizeof(dtype_reg) / sizeof(dtype_math);
  if constexpr (std::is_same_v<dtype_math, fp16>) {
    if constexpr (packed_num == 1) {
      INLINE_PISA("fred.max.hf %0, %0, %1;" : "+r"(src0) : "i"(mask));
    } else if constexpr (packed_num == 2) {
      INLINE_PISA("fred.max.hfx2 %0, %0, %1;" : "+r"(src0) : "i"(mask));
    }
  } else if constexpr (std::is_same_v<dtype_math, bf16>) {
    if constexpr (packed_num == 1) {
      INLINE_PISA("fred.max.bf %0, %0, %1;" : "+r"(src0) : "i"(mask));
    } else if constexpr (packed_num == 2) {
      INLINE_PISA("fred.max.bfx2 %0, %0, %1;" : "+r"(src0) : "i"(mask));
    }
  } else if constexpr (std::is_same_v<dtype_math, float>) {
    INLINE_PISA("fred.max.f %0, %0, %1;" : "+r"(src0) : "i"(mask));
  } else {
    static_assert(sizeof(dtype_math) == 0, "unsupported case");
  }
}

template <typename dtype_dst, typename dtype_src>
inline void u16_replicate(dtype_dst &dst, dtype_src &src) {
  static_assert(sizeof(dtype_src) == 2);
  static_assert(sizeof(dtype_dst) == 4);
  constexpr uint32_t width = 16;
  constexpr uint32_t offset = 16;
  INLINE_PISA("zext.32b.16b %0, %1;" : "=r"(dst) : "r"(src));
  INLINE_PISA("bfi.32b %0, %1, %2, %3, %4;" : "=r"(dst) : "r"(dst), "r"(dst), "r"(width), "r"(offset));
}

template <typename dtype_dst, typename dtype_src>
inline void u16_extract(dtype_dst &dst, dtype_src &src) {
  static_assert(sizeof(dtype_src) == 4);
  static_assert(sizeof(dtype_dst) == 2);
  constexpr uint32_t width = 16;
  constexpr uint32_t offset = 0;
  vector_t<uint16_t, 2> dst_tmp;
  INLINE_PISA("mov.32b %0.xy, %1;" : "=r"(dst_tmp) : "r"(src));
  INLINE_PISA("mov.16b %0, %1.x;" : "=r"(dst) : "r"(dst_tmp));
}

template <typename dtype = uint32_t>
inline dtype rol(const dtype &src, uint32_t shift) {
  dtype ret;
  if constexpr (sizeof(dtype) == 4) {
    INLINE_PISA("shf.l.32b %0, %1, %1, %2;" : "=r"(ret) : "r"(src), "r"(shift));
  } else {
    static_assert(sizeof(dtype) == 0, "unsupported dtype");
  }
  return ret;
}

template <typename dtype_math, uint32_t N_math, typename dtype_reg>
inline void gtp_tmov(dtype_reg *dst_ptr, const dtype_reg *src_ptr) {
  constexpr uint32_t N_reg = N_math * sizeof(dtype_math) / sizeof(dtype_reg);
  constexpr uint32_t N_u32 = N_math * sizeof(dtype_math) / sizeof(uint32_t);
  sycl::marray<dtype_reg, N_reg> src;
  using vtype = vector_t<uint32_t, N_u32>;
#pragma unroll
  for (int i = 0; i < N_reg; i++) {
    src[i] = src_ptr[i];
  }
  vtype vdst;
  if constexpr (std::is_same_v<dtype_math, fp16>) {
    if constexpr (N_math == 32) {
      INLINE_PISA("tmov.f16.m32n32.xch %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (N_math == 16) {
      INLINE_PISA("tmov.f16.m32n16.xch %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(sizeof(dtype_math) == 0, "unsupported N");
    }
  } else if constexpr (std::is_same_v<dtype_math, bf16>) {
    if constexpr (N_math == 32) {
      INLINE_PISA("tmov.bf16.m32n32.xch %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype>(src)));
    } else if constexpr (N_math == 16) {
      INLINE_PISA("tmov.bf16.m32n16.xch %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype>(src)));
    } else {
      static_assert(sizeof(dtype_math) == 0, "unsupported N");
    }
  } else {
    static_assert(sizeof(dtype_math) == 0, "unsupported dtype");
  }
  sycl::marray<dtype_reg, N_reg> dst = sycl::bit_cast<sycl::marray<dtype_reg, N_reg>>(vdst);
#pragma unroll
  for (int i = 0; i < N_reg; i++) {
    dst_ptr[i] = dst[i];
  }
}

template <typename dtype_math, uint32_t N_math, typename dtype_reg>
inline dtype_math gtp_tred_max(const dtype_reg *src_ptr) {
  dtype_math ret;
  constexpr uint32_t N_reg = N_math * sizeof(dtype_math) / sizeof(dtype_reg);
  constexpr uint32_t N_u32 = N_math * sizeof(dtype_math) / sizeof(uint32_t);
  sycl::marray<dtype_reg, N_reg> src0;
  using vtype = vector_t<uint32_t, N_u32>;
#pragma unroll
  for (int i = 0; i < N_reg; i++) {
    src0[i] = src_ptr[i];
  }
  uint32_t tmp;
  if constexpr (std::is_same_v<dtype_math, fp16>) {
    if constexpr (N_math == 32) {
      INLINE_PISA("tred.f16.f16.m32n32.rednd.max %0, %1;" : "=r"(tmp) : "r"(sycl::bit_cast<vtype>(src0)));
    } else if constexpr (N_math == 16) {
      INLINE_PISA("tred.f16.f16.m32n16.rednd.max %0, %1;" : "=r"(tmp) : "r"(sycl::bit_cast<vtype>(src0)));
    } else {
      static_assert(sizeof(dtype_math) == 0, "unsupported N");
    }
    INLINE_PISA("trunc.16b.32b %0, %1;" : "=r"(ret) : "r"(tmp));
  } else if constexpr (std::is_same_v<dtype_math, bf16>) {
    if constexpr (N_math == 32) {
      INLINE_PISA("tred.bf16.bf16.m32n32.rednd.max %0, %1;" : "=r"(tmp) : "r"(sycl::bit_cast<vtype>(src0)));
    } else if constexpr (N_math == 16) {
      INLINE_PISA("tred.bf16.bf16.m32n16.rednd.max %0, %1;" : "=r"(tmp) : "r"(sycl::bit_cast<vtype>(src0)));
    } else {
      static_assert(sizeof(dtype_math) == 0, "unsupported N");
    }
    INLINE_PISA("trunc.16b.32b %0, %1;" : "=r"(ret) : "r"(tmp));
  } else {
    static_assert(sizeof(dtype_math) == 0, "unsupported dtype");
  }
  return ret;
}

template <typename dtype_math, uint32_t N_math, typename dtype_reg>
inline dtype_math gtp_tred_max(const dtype_reg *src_ptr, dtype_math acc) {
  dtype_math ret;
  constexpr uint32_t N_reg = N_math * sizeof(dtype_math) / sizeof(dtype_reg);
  constexpr uint32_t N_u32 = N_math * sizeof(dtype_math) / sizeof(uint32_t);
  sycl::marray<dtype_reg, N_reg> src0;
  using vtype = vector_t<uint32_t, N_u32>;
#pragma unroll
  for (int i = 0; i < N_reg; i++) {
    src0[i] = src_ptr[i];
  }
  uint32_t tmp;
  uint32_t acc_tmp;
  if constexpr (std::is_same_v<dtype_math, fp16>) {
    INLINE_PISA("zext.32b.16b %0, %1;" : "=r"(acc_tmp) : "r"(acc));
    if constexpr (N_math == 32) {
      INLINE_PISA("tred.f16.f16.m32n32.rednd.max.acc %0, %1, %2;"
                  : "=r"(tmp)
                  : "r"(sycl::bit_cast<vtype>(src0)), "r"(acc_tmp));
    } else if constexpr (N_math == 16) {
      INLINE_PISA("tred.f16.f16.m32n16.rednd.max.acc %0, %1, %2;"
                  : "=r"(tmp)
                  : "r"(sycl::bit_cast<vtype>(src0)), "r"(acc_tmp));
    } else {
      static_assert(sizeof(dtype_math) == 0, "unsupported N");
    }
    INLINE_PISA("trunc.16b.32b %0, %1;" : "=r"(ret) : "r"(tmp));
  } else if constexpr (std::is_same_v<dtype_math, bf16>) {
    INLINE_PISA("zext.32b.16b %0, %1;" : "=r"(acc_tmp) : "r"(acc));
    if constexpr (N_math == 32) {
      INLINE_PISA("tred.bf16.bf16.m32n32.rednd.max.acc %0, %1, %2;"
                  : "=r"(tmp)
                  : "r"(sycl::bit_cast<vtype>(src0)), "r"(acc_tmp));
    } else if constexpr (N_math == 16) {
      INLINE_PISA("tred.bf16.bf16.m32n16.rednd.max.acc %0, %1, %2;"
                  : "=r"(tmp)
                  : "r"(sycl::bit_cast<vtype>(src0)), "r"(acc_tmp));
    } else {
      static_assert(sizeof(dtype_math) == 0, "unsupported N");
    }
    INLINE_PISA("trunc.16b.32b %0, %1;" : "=r"(ret) : "r"(tmp));
  } else {
    static_assert(sizeof(dtype_math) == 0, "unsupported dtype");
  }
  return ret;
}

template <typename dtype_math, uint32_t N_math, typename dtype_reg>
inline float gtp_texp_red_sum(dtype_reg *dst_ptr, const dtype_reg *src_ptr) {
  float ret;
  constexpr uint32_t N_reg = N_math * sizeof(dtype_math) / sizeof(dtype_reg);
  constexpr uint32_t N_u32 = N_math * sizeof(dtype_math) / sizeof(uint32_t);
  sycl::marray<dtype_reg, N_reg> src0;
  using vtype = vector_t<uint32_t, N_u32>;
#pragma unroll
  for (int i = 0; i < N_reg; i++) {
    src0[i] = src_ptr[i];
  }
  vtype vdst;
  if constexpr (std::is_same_v<dtype_math, fp16>) {
    if constexpr (N_math == 32) {
      INLINE_PISA("texp.f16.m32n32.rednd %0, %1, %2;" : "=r"(vdst), "=r"(ret) : "r"(sycl::bit_cast<vtype>(src0)));
    } else if constexpr (N_math == 16) {
      INLINE_PISA("texp.f16.m32n16.rednd %0, %1, %2;" : "=r"(vdst), "=r"(ret) : "r"(sycl::bit_cast<vtype>(src0)));
    } else {
      static_assert(sizeof(dtype_math) == 0, "unsupported N");
    }
  } else if constexpr (std::is_same_v<dtype_math, bf16>) {
    if constexpr (N_math == 32) {
      INLINE_PISA("texp.bf16.m32n32.rednd %0, %1, %2;" : "=r"(vdst), "=r"(ret) : "r"(sycl::bit_cast<vtype>(src0)));
    } else if constexpr (N_math == 16) {
      INLINE_PISA("texp.bf16.m32n16.rednd %0, %1, %2;" : "=r"(vdst), "=r"(ret) : "r"(sycl::bit_cast<vtype>(src0)));
    } else {
      static_assert(sizeof(dtype_math) == 0, "unsupported N");
    }
  } else {
    static_assert(sizeof(dtype_math) == 0, "unsupported dtype");
  }
  sycl::marray<dtype_reg, N_reg> dst = sycl::bit_cast<sycl::marray<dtype_reg, N_reg>>(vdst);
#pragma unroll
  for (int i = 0; i < N_reg; i++) {
    dst_ptr[i] = dst[i];
  }
  return ret;
}

template <typename dtype_math, uint32_t N_math, typename dtype_reg>
inline float gtp_texp_red_sum(dtype_reg *dst_ptr, const dtype_reg *src_ptr, float sum_src) {
  float ret;
  constexpr uint32_t N_reg = N_math * sizeof(dtype_math) / sizeof(dtype_reg);
  constexpr uint32_t N_u32 = N_math * sizeof(dtype_math) / sizeof(uint32_t);
  sycl::marray<dtype_reg, N_reg> src0;
  using vtype = vector_t<uint32_t, N_u32>;
#pragma unroll
  for (int i = 0; i < N_reg; i++) {
    src0[i] = src_ptr[i];
  }
  vtype vdst;
  if constexpr (std::is_same_v<dtype_math, fp16>) {
    if constexpr (N_math == 32) {
      INLINE_PISA("texp.f16.m32n32.rednd.acc %0, %1, %2, %3;"
                  : "=r"(vdst), "=r"(ret)
                  : "r"(sycl::bit_cast<vtype>(src0)), "r"(sum_src));
    } else if constexpr (N_math == 16) {
      INLINE_PISA("texp.f16.m32n16.rednd.acc %0, %1, %2, %3;"
                  : "=r"(vdst), "=r"(ret)
                  : "r"(sycl::bit_cast<vtype>(src0)), "r"(sum_src));
    } else {
      static_assert(sizeof(dtype_math) == 0, "unsupported N");
    }
  } else if constexpr (std::is_same_v<dtype_math, bf16>) {
    if constexpr (N_math == 32) {
      INLINE_PISA("texp.bf16.m32n32.rednd.acc %0, %1, %2, %3;"
                  : "=r"(vdst), "=r"(ret)
                  : "r"(sycl::bit_cast<vtype>(src0)), "r"(sum_src));
    } else if constexpr (N_math == 16) {
      INLINE_PISA("texp.bf16.m32n16.rednd.acc %0, %1, %2, %3;"
                  : "=r"(vdst), "=r"(ret)
                  : "r"(sycl::bit_cast<vtype>(src0)), "r"(sum_src));
    } else {
      static_assert(sizeof(dtype_math) == 0, "unsupported N");
    }
  } else {
    static_assert(sizeof(dtype_math) == 0, "unsupported dtype");
  }
  sycl::marray<dtype_reg, N_reg> dst = sycl::bit_cast<sycl::marray<dtype_reg, N_reg>>(vdst);
#pragma unroll
  for (int i = 0; i < N_reg; i++) {
    dst_ptr[i] = dst[i];
  }
  return ret;
}

template <typename dtype_dst, typename dtype_src, uint32_t N, typename dtype_reg>
inline void gtp_tcvd(dtype_reg *dst_ptr, const dtype_reg *src_ptr) {
  constexpr uint32_t N_dst =
      (N * ::sizeof_bits<dtype_dst>() / BITS_PER_BYTE + sizeof(dtype_reg) - 1) / sizeof(dtype_reg);
  constexpr uint32_t N_src = (N * sizeof(dtype_src) + sizeof(dtype_reg) - 1) / sizeof(dtype_reg);
  constexpr uint32_t N_u32_dst =
      (N * ::sizeof_bits<dtype_dst>() / BITS_PER_BYTE + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  constexpr uint32_t N_u32_src = (N * sizeof(dtype_src) + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  sycl::marray<dtype_reg, N_src> src;
  using vtype_src = vector_t<uint32_t, N_u32_src>;
  using vtype_dst = vector_t<uint32_t, N_u32_dst>;
#pragma unroll
  for (int i = 0; i < N_src; i++) {
    src[i] = src_ptr[i];
  }
  vtype_dst vdst;
  if constexpr (std::is_same_v<dtype_src, float>) {
    if constexpr (std::is_same_v<dtype_dst, bf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e5m2.f32.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e5m2.f32.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 8) {
        INLINE_PISA("tcvd.e5m2.f32.m32n8 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 1) {
        INLINE_PISA("tcvd.e5m2.f32.m32n1 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst, hf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e4m3.f32.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e4m3.f32.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst, fp4_e2m1>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e2m1.f32.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e2m1.f32.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else {
      static_assert(sizeof(dtype_dst) == 0, "unsupported dtype_dst");
    }
  } else if constexpr (std::is_same_v<dtype_src, fp16>) {
    if constexpr (std::is_same_v<dtype_dst, bf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e5m2.f16.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e5m2.f16.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 8) {
        INLINE_PISA("tcvd.e5m2.f16.m32n8 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 1) {
        INLINE_PISA("tcvd.e5m2.f16.m32n1 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst, hf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e4m3.f16.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e4m3.f16.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst, fp4_e2m1>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e2m1.f16.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e2m1.f16.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else {
      static_assert(sizeof(dtype_dst) == 0, "unsupported dtype_dst");
    }
  } else if constexpr (std::is_same_v<dtype_src, bf16>) {
    if constexpr (std::is_same_v<dtype_dst, bf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e5m2.bf16.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e5m2.bf16.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 8) {
        INLINE_PISA("tcvd.e5m2.bf16.m32n8 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst, hf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e4m3.bf16.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e4m3.bf16.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 8) {
        INLINE_PISA("tcvd.e4m3.bf16.m32n8 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst, fp4_e2m1>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e2m1.bf16.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e2m1.bf16.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else {
      static_assert(sizeof(dtype_dst) == 0, "unsupported dtype_dst");
    }
  } else {
    static_assert(sizeof(dtype_src) == 0, "unsupported dtype_src");
  }
  sycl::marray<dtype_reg, N_dst> dst = sycl::bit_cast<sycl::marray<dtype_reg, N_dst>>(vdst);
#pragma unroll
  for (int i = 0; i < N_dst; i++) {
    dst_ptr[i] = dst[i];
  }
}

template <typename dtype_dst, typename dtype_src, uint32_t N>
inline void gtp_tcvd(dtype_dst *dst_ptr, const dtype_src *src_ptr) {
  constexpr uint32_t dbits_u32 = sizeof(uint32_t) * BITS_PER_BYTE;
  constexpr uint32_t dbits_src = ::sizeof_bits<dtype_src>();
  constexpr uint32_t dbits_dst = ::sizeof_bits<dtype_dst>();

  constexpr uint32_t N_u32_src = (N * dbits_src + dbits_u32 - 1) / dbits_u32;
  constexpr uint32_t N_u32_dst = (N * dbits_dst + dbits_u32 - 1) / dbits_u32;
  // could bigger than N
  constexpr uint32_t N_src = N_u32_src * dbits_u32 / dbits_src;

  sycl::marray<dtype_src, N_src> src;

  using vtype_src = vector_t<uint32_t, N_u32_src>;
  using vtype_dst = vector_t<uint32_t, N_u32_dst>;
#pragma unroll
  for (int i = 0; i < N; i++) {
    src[i] = src_ptr[i];
  }
  vtype_dst vdst;
  if constexpr (std::is_same_v<dtype_src, fp16>) {
    if constexpr (std::is_same_v<dtype_dst, bf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e5m2.f16.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e5m2.f16.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 8) {
        INLINE_PISA("tcvd.e5m2.f16.m32n8 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 1) {
        INLINE_PISA("tcvd.e5m2.f16.m32n1 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst, hf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e4m3.f16.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e4m3.f16.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst, fp4_e2m1>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e2m1.f16.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e2m1.f16.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else {
      static_assert(sizeof(dtype_dst) == 0, "unsupported dtype_dst");
    }
  } else if constexpr (std::is_same_v<dtype_src, bf16>) {
    if constexpr (std::is_same_v<dtype_dst, bf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e5m2.bf16.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e5m2.bf16.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 8) {
        INLINE_PISA("tcvd.e5m2.bf16.m32n8 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst, hf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e4m3.bf16.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e4m3.bf16.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst, fp4_e2m1>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e2m1.bf16.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e2m1.bf16.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst,
                                        mxint8>) { // todo: compiler has not supported tcvdmx without mxeb/...
      e8m0 scale(1.0f);
      uint32_t v = scale.data;
      if constexpr (N == 32) {
        INLINE_PISA("tcvdmx.s8.bf16.m32n32.mxeb %0, %1, %2;"
                    : "=r"(vdst)
                    : "r"(sycl::bit_cast<vtype_src>(src)), "r"(v));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvdmx.s8.bf16.m32n16.mxeb %0, %1, %2;"
                    : "=r"(vdst)
                    : "r"(sycl::bit_cast<vtype_src>(src)), "r"(v));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst, int8_t>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.s8.bf16.m32n32 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.s8.bf16.m32n16 %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else {
      static_assert(sizeof(dtype_dst) == 0, "unsupported dtype_dst");
    }
  } else {
    static_assert(sizeof(dtype_src) == 0, "unsupported dtype_src");
  }

  constexpr uint32_t N_u8_dst = (N * dbits_dst + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
  // for sub-byte type, will cvt to uint8_t first.
  using dtype_reg = std::conditional_t<(dbits_dst < 8), uint8_t, dtype_dst>;
  // to make it u8/element aligned
  constexpr uint32_t N_dst = (dbits_dst < 8) ? N_u8_dst : N;
  // to make it u32 aligned
  constexpr uint32_t N_reg = N_u32_dst * dbits_u32 / sizeof_bits<dtype_reg>();
  sycl::marray<dtype_reg, N_reg> dst = sycl::bit_cast<sycl::marray<dtype_reg, N_reg>>(vdst);
  dtype_reg *dst_reg_ptr = reinterpret_cast<dtype_reg *>(dst_ptr);
#pragma unroll
  for (uint32_t i = 0; i < N_dst; i++) {
    dst_reg_ptr[i] = dst[i];
  }
}

template <typename dtype_dst, typename dtype_src, uint32_t N, typename dtype_reg>
inline void gtp_tcvdmx(dtype_reg *dst_ptr, const dtype_reg *src_ptr, const uint8_t &meta) {
  constexpr uint32_t N_dst = N * ::sizeof_bits<dtype_dst>() / BITS_PER_BYTE / sizeof(dtype_reg);
  constexpr uint32_t N_src = N * sizeof(dtype_src) / sizeof(dtype_reg);
  constexpr uint32_t N_u32_dst = N * ::sizeof_bits<dtype_dst>() / BITS_PER_BYTE / sizeof(uint32_t);
  constexpr uint32_t N_u32_src = N * sizeof(dtype_src) / sizeof(uint32_t);
  sycl::marray<dtype_reg, N_src> src;
  using vtype_src = vector_t<uint32_t, N_u32_src>;
  using vtype_dst = vector_t<uint32_t, N_u32_dst>;
#pragma unroll
  for (int i = 0; i < N_src; i++) {
    src[i] = src_ptr[i];
  }
  vtype_dst vdst;
  uint32_t meta_u32 = meta;
  if constexpr (std::is_same_v<dtype_src, fp16>) {
    if constexpr (std::is_same_v<dtype_dst, bf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvdmx.e5m2.f16.m32n32.mxnd %0, %1, %2;"
                    : "=r"(vdst)
                    : "r"(sycl::bit_cast<vtype_src>(src)), "r"(meta_u32));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvdmx.e5m2.f16.m32n16.mxnd %0, %1, %2;"
                    : "=r"(vdst)
                    : "r"(sycl::bit_cast<vtype_src>(src)), "r"(meta_u32));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst, fp4_e2m1>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvdmx.e2m1.f16.m32n32.mxnd %0, %1, %2;"
                    : "=r"(vdst)
                    : "r"(sycl::bit_cast<vtype_src>(src)), "r"(meta_u32));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvdmx.e2m1.f16.m32n16.mxnd %0, %1, %2;"
                    : "=r"(vdst)
                    : "r"(sycl::bit_cast<vtype_src>(src)), "r"(meta_u32));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else {
      static_assert(sizeof(dtype_dst) == 0, "unsupported dtype_dst");
    }
  } else if constexpr (std::is_same_v<dtype_src, bf16>) {
    if constexpr (std::is_same_v<dtype_dst, bf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvdmx.e5m2.bf16.m32n32.mxnd %0, %1, %2;"
                    : "=r"(vdst)
                    : "r"(sycl::bit_cast<vtype_src>(src)), "r"(meta_u32));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvdmx.e5m2.bf16.m32n16.mxnd %0, %1, %2;"
                    : "=r"(vdst)
                    : "r"(sycl::bit_cast<vtype_src>(src)), "r"(meta_u32));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst, fp4_e2m1>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvdmx.e2m1.bf16.m32n32.mxnd %0, %1, %2;"
                    : "=r"(vdst)
                    : "r"(sycl::bit_cast<vtype_src>(src)), "r"(meta_u32));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvdmx.e2m1.bf16.m32n16.mxnd %0, %1, %2;"
                    : "=r"(vdst)
                    : "r"(sycl::bit_cast<vtype_src>(src)), "r"(meta_u32));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else {
      static_assert(sizeof(dtype_dst) == 0, "unsupported dtype_dst");
    }
  } else {
    static_assert(sizeof(dtype_src) == 0, "unsupported dtype_src");
  }
  sycl::marray<dtype_reg, N_dst> dst = sycl::bit_cast<sycl::marray<dtype_reg, N_dst>>(vdst);
#pragma unroll
  for (int i = 0; i < N_dst; i++) {
    dst_ptr[i] = dst[i];
  }
}

template <typename dtype_dst, typename dtype_src, uint32_t N, typename dtype_reg>
inline void gtp_tcvd_xch(dtype_reg *dst_ptr, const dtype_reg *src_ptr) {
  constexpr uint32_t N_dst = N * ::sizeof_bits<dtype_dst>() / BITS_PER_BYTE / sizeof(dtype_reg);
  constexpr uint32_t N_src = N * sizeof(dtype_src) / sizeof(dtype_reg);
  constexpr uint32_t N_u32_dst = N * ::sizeof_bits<dtype_dst>() / BITS_PER_BYTE / sizeof(uint32_t);
  constexpr uint32_t N_u32_src = N * sizeof(dtype_src) / sizeof(uint32_t);
  sycl::marray<dtype_reg, N_src> src;
  using vtype_src = vector_t<uint32_t, N_u32_src>;
  using vtype_dst = vector_t<uint32_t, N_u32_dst>;
#pragma unroll
  for (int i = 0; i < N_src; i++) {
    src[i] = src_ptr[i];
  }
  vtype_dst vdst;
  if constexpr (std::is_same_v<dtype_src, fp16>) {
    if constexpr (std::is_same_v<dtype_dst, bf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e5m2.f16.m32n32.xch %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e5m2.f16.m32n16.xch %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst, hf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e4m3.f16.m32n32.xch %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e4m3.f16.m32n16.xch %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else {
      static_assert(sizeof(dtype_dst) == 0, "unsupported dtype_dst");
    }
  } else if constexpr (std::is_same_v<dtype_src, bf16>) {
    if constexpr (std::is_same_v<dtype_dst, bf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e5m2.bf16.m32n32.xch %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e5m2.bf16.m32n16.xch %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else if constexpr (std::is_same_v<dtype_dst, hf8>) {
      if constexpr (N == 32) {
        INLINE_PISA("tcvd.e4m3.bf16.m32n32.xch %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else if constexpr (N == 16) {
        INLINE_PISA("tcvd.e4m3.bf16.m32n16.xch %0, %1;" : "=r"(vdst) : "r"(sycl::bit_cast<vtype_src>(src)));
      } else {
        static_assert(sizeof(dtype_src) == 0, "unsupported N");
      }
    } else {
      static_assert(sizeof(dtype_dst) == 0, "unsupported dtype_dst");
    }
  } else {
    static_assert(sizeof(dtype_src) == 0, "unsupported dtype_src");
  }
  sycl::marray<dtype_reg, N_dst> dst = sycl::bit_cast<sycl::marray<dtype_reg, N_dst>>(vdst);
#pragma unroll
  for (int i = 0; i < N_dst; i++) {
    dst_ptr[i] = dst[i];
  }
}

uint32_t inline get_lane_id() {
  uint32_t ret;
  INLINE_PISA("mov.32b %0, %%laneid;" : "=r"(ret) :);
  return ret;
}

inline uint32_t get_sg_id() {
  uint32_t ret;
  INLINE_PISA("mov.32b %0, %%subgroupid;" : "=r"(ret) :);
  return ret;
}

template <uint32_t dim = 0>
uint32_t inline get_wgid() {
  uint32_t ret;
  if constexpr (dim == 0) {
    INLINE_PISA("mov.32b %0, %%groupid.x;" : "=r"(ret) :);
  } else if constexpr (dim == 1) {
    INLINE_PISA("mov.32b %0, %%groupid.y;" : "=r"(ret) :);
  } else if constexpr (dim == 2) {
    INLINE_PISA("mov.32b %0, %%groupid.z;" : "=r"(ret) :);
  } else {
    static_assert(sizeof(dim) < 0, "wrong dim");
  }
  return ret;
}

template <uint32_t dim = 0>
uint32_t inline get_wgcount() {
  uint32_t ret;
  if constexpr (dim == 0) {
    INLINE_PISA("mov.32b %0, %%groupcount.x;" : "=r"(ret) :);
  } else if constexpr (dim == 1) {
    INLINE_PISA("mov.32b %0, %%groupcount.y;" : "=r"(ret) :);
  } else if constexpr (dim == 2) {
    INLINE_PISA("mov.32b %0, %%groupcount.z;" : "=r"(ret) :);
  } else {
    static_assert(sizeof(dim) < 0, "wrong dim");
  }
  return ret;
}

template <typename dstType, typename srcType, typename metaType, uint32_t M, uint32_t N>
inline void tcvdmx_rednd_mxnd_srnd(dstType *dst, srcType *src, metaType *meta, uint32_t &seed) {

  static_assert(sizeof(srcType) == 2, "tcvdmx_rednd_mxnd_srnd only support srcType as bf16 or fp16");
  static_assert(sizeof(metaType) == 1, "tcvdmx_rednd_mxnd_srnd only support metaType as uint8_t");

  if constexpr (std::is_same_v<srcType, fp16> && std::is_same_v<metaType, e8m0> && std::is_same_v<dstType, bf8>) {
    if constexpr (M == 32 && N == 32) {
      constexpr int SrcPack = sizeof(uint32_t) / sizeof(srcType);
      constexpr int SrcSize = N / SrcPack;
      sycl::marray<uint32_t, SrcSize> sSrc;
      uint32_t *matC_ptr = reinterpret_cast<uint32_t *>(src);
#pragma unroll
      for (int i = 0; i < SrcSize; i++) {
        sSrc[i] = matC_ptr[i];
      }

      constexpr int DstPack = sizeof(uint32_t) * 8u / ::sizeof_bits<dstType>();
      constexpr int DstSize = N / DstPack;
      vector_t<uint32_t, DstSize> sdstVecT;

      uint32_t sMetaVecT;

      INLINE_PISA("tcvdmx.e5m2.f16.m32n32.rednd.mxnd.srnd.swseed %[dst], %[dm], %[src0], %[dsrc1] ;"
                  : [dst] "=r"(sdstVecT), [dm] "=r"(sMetaVecT)
                  : [src0] "r"(sycl::bit_cast<vector_t<uint32_t, SrcSize>>(sSrc)), [dsrc1] "+r"(seed));

      auto sDst = sycl::bit_cast<sycl::marray<uint8_t, DstSize * 4>>(sdstVecT);
      uint8_t *matD_ptr = reinterpret_cast<uint8_t *>(dst);
      constexpr uint32_t N_dst_size = (N * ::sizeof_bits<dstType>() + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
#pragma unroll
      for (int i = 0; i < N_dst_size; i++) {
        matD_ptr[i] = sDst[i];
      }

      uint8_t *meta_ptr = reinterpret_cast<uint8_t *>(meta);
      *meta_ptr = sMetaVecT & 0xff;
    } else {
      static_assert(false, "tcvdmx Unsupported M and N");
    }
  } else {
    static_assert(false, "tcvdmx_rednd_mxnd_srnd only support ");
  }
}

template <typename dstType, typename srcType, typename metaType, uint32_t M, uint32_t N, bool ndim = true>
inline void tcvdmx_rne(dstType *dst, srcType *src, metaType *meta) {

  static_assert(sizeof(srcType) == 2, "tcvdmx_rne only support srcType as bf16 or fp16");
  static_assert(sizeof(metaType) == 1, "tcvdmx_rne only support metaType as uint8_t");

  if constexpr (std::is_same_v<srcType, fp16> && std::is_same_v<metaType, e8m0> && std::is_same_v<dstType, bf8>) {

    constexpr int SrcPack = sizeof(uint32_t) / sizeof(srcType);
    constexpr int SrcSize = (N + SrcPack - 1) / SrcPack;
    sycl::marray<uint32_t, SrcSize> sSrc;
    uint32_t *matC_ptr = reinterpret_cast<uint32_t *>(src);
#pragma unroll
    for (int i = 0; i < SrcSize; i++) {
      sSrc[i] = matC_ptr[i];
    }

    constexpr int DstPack = sizeof(uint32_t) * 8u / ::sizeof_bits<dstType>();
    constexpr int DstSize = (N + DstPack - 1) / DstPack;
    vector_t<uint32_t, DstSize> sdstVecT;

    uint32_t sMetaVecT;

    if constexpr (M == 32 && N == 32) {
      if constexpr (ndim) {
        INLINE_PISA("tcvdmx.e5m2.f16.m32n32.rednd.mxnd.rne %[dst], %[dm], %[src0] ;"
                    : [dst] "=r"(sdstVecT), [dm] "=r"(sMetaVecT)
                    : [src0] "r"(sycl::bit_cast<vector_t<uint32_t, SrcSize>>(sSrc)));
      } else {
        INLINE_PISA("tcvdmx.e5m2.f16.m32n32.redmd.mxmd.rne %[dst], %[dm], %[src0] ;"
                    : [dst] "=r"(sdstVecT), [dm] "=r"(sMetaVecT)
                    : [src0] "r"(sycl::bit_cast<vector_t<uint32_t, SrcSize>>(sSrc)));
      }
    } else if constexpr (M == 32 && N == 16) {

      if constexpr (ndim) {
        INLINE_PISA("tcvdmx.e5m2.f16.m32n16.rednd.mxnd.rne %[dst], %[dm], %[src0] ;"
                    : [dst] "=r"(sdstVecT), [dm] "=r"(sMetaVecT)
                    : [src0] "r"(sycl::bit_cast<vector_t<uint32_t, SrcSize>>(sSrc)));
      } else {
        INLINE_PISA("tcvdmx.e5m2.f16.m32n16.redmd.mxmd.rne %[dst], %[dm], %[src0] ;"
                    : [dst] "=r"(sdstVecT), [dm] "=r"(sMetaVecT)
                    : [src0] "r"(sycl::bit_cast<vector_t<uint32_t, SrcSize>>(sSrc)));
      }
    } else if constexpr (M == 32 && N == 1) {

      if constexpr (ndim) {
        INLINE_PISA("tcvdmx.e5m2.f16.m32n1.rednd.mxnd.rne %[dst], %[dm], %[src0] ;"
                    : [dst] "=r"(sdstVecT), [dm] "=r"(sMetaVecT)
                    : [src0] "r"(sycl::bit_cast<vector_t<uint32_t, SrcSize>>(sSrc)));
      } else {
        INLINE_PISA("tcvdmx.e5m2.f16.m32n1.redmd.mxmd.rne %[dst], %[dm], %[src0] ;"
                    : [dst] "=r"(sdstVecT), [dm] "=r"(sMetaVecT)
                    : [src0] "r"(sycl::bit_cast<vector_t<uint32_t, SrcSize>>(sSrc)));
      }
    } else {
      static_assert(false, "tcvdmx Unsupported M and N");
    }

    auto sDst = sycl::bit_cast<sycl::marray<uint8_t, DstSize * 4>>(sdstVecT);
    uint8_t *matD_ptr = reinterpret_cast<uint8_t *>(dst);
    constexpr uint32_t N_dst_size = (N * ::sizeof_bits<dstType>() + BITS_PER_BYTE - 1) / BITS_PER_BYTE;
#pragma unroll
    for (int i = 0; i < N_dst_size; i++) {
      matD_ptr[i] = sDst[i];
    }

    uint8_t *meta_ptr = reinterpret_cast<uint8_t *>(meta);
    *meta_ptr = sMetaVecT & 0xff;

  } else {
    static_assert(false, "tcvdmx_rne only support ");
  }
}

struct matrix_desc_t {
  template <typename slm_dtype>
  inline matrix_desc_t(slm_dtype *slm_ptr, uint32_t matrix_stride, slm_matrix_type cm_type) {
    constexpr uint32_t base_width = 12;
    constexpr uint32_t base_offset = 0;
    constexpr uint32_t matrix_stride_width = 11;
    constexpr uint32_t matrix_stride_offset = 16;
    constexpr uint32_t cm_type_width = 2;
    constexpr uint32_t cm_type_offset = 28;
    uint32_t base_tmp = (uint64_t)slm_ptr >> 9;
    INLINE_PISA("bfi.32b %0, %1, %2, %3, %4;"
                : "=r"(data)
                : "r"(data), "r"(base_tmp), "r"(base_width), "r"(base_offset));
    INLINE_PISA("bfi.32b %0, %1, %2, %3, %4;"
                : "=r"(data)
                : "r"(data), "r"(matrix_stride >> 2), "r"(matrix_stride_width), "r"(matrix_stride_offset));
    INLINE_PISA("bfi.32b %0, %1, %2, %3, %4;"
                : "=r"(data)
                : "r"(data), "r"(uint32_t(cm_type)), "r"(cm_type_width), "r"(cm_type_offset));
  }

  inline matrix_desc_t(uint32_t data_) : data(data_) {}

  inline matrix_desc_t(const matrix_desc_t &desc) : data(desc.get()) {}

  inline const uint32_t get() const { return data; }

  inline matrix_desc_t operator+(uint32_t offset) const { return matrix_desc_t(data + (offset >> 9)); }

  inline matrix_desc_t &operator+=(uint32_t offset) {
    this->data += (offset >> 9);
    return *this;
  }

  private:
  uint32_t data;
};

template <typename dtype_dst, typename dtype_src, uint32_t row_elem_per_wi>
inline void fp_tile_cvt(dtype_dst *dst_ptr, dtype_src *src_ptr) {
  constexpr uint32_t dbits_src = sizeof_bits<dtype_src>();
  constexpr uint32_t dbits_dst = sizeof_bits<dtype_dst>();
  constexpr uint32_t is_require_imm = (dbits_src > 16) && (dbits_dst < 16);
  using dtypeImm = typename std::conditional<is_require_imm, bf16, dtype_src>::type;

  constexpr uint32_t tp_size = (row_elem_per_wi <= 32) ? row_elem_per_wi : 32;
  dtypeImm arrary_imm[row_elem_per_wi];
#pragma unroll
  for (int k = 0; k < row_elem_per_wi; k += tp_size) {
    if constexpr (is_require_imm) {
      copy_cvt_pack<tp_size>(arrary_imm + k, src_ptr + k);
    } else {
      copy_cvt<tp_size>(arrary_imm + k, src_ptr + k);
    }
  }
#pragma unroll
  for (int k = 0; k < row_elem_per_wi; k += tp_size) {
    if constexpr (dbits_dst < 16) {
      gtp_tcvd<dtype_dst, dtypeImm, tp_size>(dst_ptr + k, arrary_imm + k);
    } else {
      copy_cvt_pack<tp_size>(dst_ptr + k, arrary_imm + k);
    }
  }
}
