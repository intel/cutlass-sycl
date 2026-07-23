#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// for tf32, bf8 and hf8, we define them in the namespace base_type
// copied from xetla
// for mx_fp4, we also define it in the namespace base_type, it's self defined

namespace base_type {

using bf16 = sycl::ext::oneapi::bfloat16;
using fp16 = sycl::half;

/// @brief xetla tf32 data type.
/// The difference between tf32 and fp32 is:
///
/// fp32: 0_00000000_00000000000000000000000
///
/// tf32: 0_00000000_0000000000
/// @note
/// The member function in tf32 class is only used in host side.
/// For device side, we will automatically convert it to its native type.
/// @see native_type_t
///
struct tf32 {
  uint32_t data;

  //#ifdef __SYCL_DEVICE_ONLY__
  //    ///
  //#else
  operator float() const {
    uint32_t temp = data;
    return *reinterpret_cast<float *>(&temp);
  }

  tf32(float val) { data = (*reinterpret_cast<uint32_t *>(&val)) & 0xFFFFE000; }

  tf32 &operator=(float val) {
    this->data = (*reinterpret_cast<uint32_t *>(&val)) & 0xFFFFE000;
    return *this;
  }

  //#endif
};

/// @brief xetla bf8 data type.
/// The difference between bf8 and fp16 is:
///
/// fp16: 0_00000_0000000000
///
/// bf8:  0_00000_00
/// FP32 => BF8 conversion is handled in two steps
/// First convert to FP16, then to BF8/FP32
/// @note
/// The member function in bf8 class is only used in host side.
/// For device side, we will automatically convert it to its native type.
/// @see native_type_t
///
struct bf8 {
#ifdef _WIN32
  typedef unsigned short ushort;
#endif
  uint8_t data;

  operator float() const {
    uint16_t temp = data;
    temp = temp << 0x8;
    fp16 temp_fp16 = *reinterpret_cast<fp16 *>(&temp);
    float temp_fp32 = temp_fp16;
    return temp_fp32;
  }

  bf8() {}

  bf8(float val) {
    fp16 val_fp16 = val;
    ushort *p = (ushort *)&val_fp16;
    ushort sign = p[0] >> 15;
    ushort exp = (p[0] >> 10) & 0b11111;
    ushort mant = p[0] & 0x3FF;
    ushort q_mant = (mant >> 8) & 0b11;
    uint8_t ret_tmp;
    // RNE rounding to convert FP16 mantissa to BF8 mantissa
    q_mant += (mant & 0x80) && ((mant & 0x7F) || (mant & 0x100));
    ret_tmp = sign << 7 | exp << 2;
    ret_tmp += q_mant;
    data = ret_tmp;
  }
};

/// @brief xetla hf8 data type.
/// The difference between hf8 and fp16 is:
///
/// fp16: 0_00000_0000000000
///
/// hf8:  0_0000_000
/// FP32 => HF8 conversion is handled in two steps
/// First convert to FP16, then to HF8/FP32
/// @note
/// The member function in hf8 class is only used in host side.
/// For device side, we will automatically convert it to its native type.
/// @see native_type_t
///
struct hf8 {
  uint8_t data;
  static constexpr int16_t max_exp_unbiased = 8;
  static constexpr int16_t min_exp_unbiased = -6;
  static constexpr int16_t exp_bias = 7;
  static constexpr uint16_t exp_size = 4;
  static constexpr uint16_t mant_size = 3;
  static constexpr uint8_t nan = 0x7f;
  static constexpr uint8_t max_val = 0x7e;
  static constexpr bool is_saturation = false;

  hf8() {}

  operator float() const {
    // Extract individual fields from hf8
    uint16_t sign = data >> 7;
    uint16_t exp = (data >> 3) & 0b1111;
    uint16_t mant = data & 0x07;
    uint16_t dst_val;
    if ((exp == 0xf) && (mant == 0x7)) {
      dst_val = 0x7fff;
    } else if ((exp == 0) && (mant == 0)) {
      dst_val = 0;
    } else if ((exp == 0) && (mant != 0)) {
      uint16_t lz_count = (mant > 3) ? 0 : ((mant > 1) ? 1 : 2);
      uint16_t dst_exp = exp - exp_bias + 15 - lz_count;
      uint16_t dst_mant = (mant << (lz_count + 1)) & 0x7;
      dst_val = (dst_exp << 10) | (dst_mant << 7);
    } else {
      uint16_t dst_exp = exp - exp_bias + 15;
      dst_val = (dst_exp << 10) | (mant << 7);
    }

    uint16_t temp = (sign << 15) | dst_val;
    fp16 temp_fp16 = *reinterpret_cast<fp16 *>(&temp);
    float temp_fp32 = temp_fp16;
    return temp_fp32;
  }

  hf8(float val) {
    // Convert to fp16
    fp16 val_fp16 = val;
    uint16_t *p = (uint16_t *)&val_fp16;
    uint16_t src = p[0];
    // Convert to hf8
    static constexpr uint16_t src_exp_size = 5;
    static constexpr uint16_t src_mant_size = 10;
    static constexpr uint16_t src_exp_bias = (1 << (src_exp_size - 1)) - 1;
    static constexpr uint16_t src_exp_mask = (1 << src_exp_size) - 1;
    static constexpr uint16_t src_mant_mask = (1 << src_mant_size) - 1;

    uint16_t src_sign = src >> (src_exp_size + src_mant_size);
    uint16_t src_exp = (src >> src_mant_size) & src_exp_mask;
    int16_t src_exp_unbiased = src_exp - src_exp_bias;
    uint16_t src_mant = src & src_mant_mask;

    bool is_src_inf_nan = src_exp == 0x1f;
    bool is_overflow = (src_exp_unbiased > max_exp_unbiased)
        // max normal mantissa is 0b110, RNE round
        || ((src_exp_unbiased == max_exp_unbiased) && (src_mant > 0x0340));
    bool is_zero = (src_exp_unbiased < (min_exp_unbiased - mant_size));
    bool is_denorm = (src_exp_unbiased < min_exp_unbiased) && (!is_zero);

    uint8_t dst_val;
    if (is_src_inf_nan) {
      dst_val = nan;
    } else if (is_overflow) {
      dst_val = is_saturation ? max_val : nan;
    } else if (is_zero) {
      dst_val = 0;
    } else if (is_denorm) {
      // src_denormal case already in is_zero branch
      uint16_t src_m = src_mant | 0x0400;
      int16_t shift_out_bit = min_exp_unbiased - src_exp_unbiased;
      bool sticky_flag = (src_m & ((1 << shift_out_bit) - 1)) != 0;
      src_m = src_m >> shift_out_bit;
      // RNE rounding
      uint16_t tail_size = src_mant_size - mant_size;
      // exclude the rounding bit
      sticky_flag = sticky_flag || ((src_m & ((1 << (tail_size - 1)) - 1)) != 0);
      bool lsb_bit = src_m & (1 << tail_size);
      bool rnd_bit = src_m & (1 << (tail_size - 1));
      bool carry = (lsb_bit && rnd_bit) || (rnd_bit && sticky_flag);

      dst_val = (src_m >> tail_size) + carry;
    } else {
      uint16_t tail_size = src_mant_size - mant_size;
      // exclude the rounding bit
      bool sticky_flag = (src_mant & ((1 << (tail_size - 1)) - 1)) != 0;
      bool lsb_bit = src_mant & (1 << tail_size);
      bool rnd_bit = src_mant & (1 << (tail_size - 1));
      bool carry = (lsb_bit && rnd_bit) || (rnd_bit && sticky_flag);
      uint16_t src_m = (src_mant >> tail_size) + carry;
      uint16_t src_e = src_exp_unbiased + exp_bias;
      // overflow will be handled in is_overflow
      dst_val = (src_e << mant_size) + src_m;
    }
    data = (src_sign << (exp_size + mant_size)) | dst_val;
  }
};

namespace impl {
// inside of namespace impl, not expose to user
struct fp4_e2m1 {
  uint8_t data;
  fp4_e2m1() = default;

  explicit fp4_e2m1(uint8_t val) { data = val; }

  // down cvt from float
  explicit fp4_e2m1(float val) {
    // initial implementation, now round to nearest, no other rounding mode
    // supported
    if (std::isnan(val) || std::isinf(val)) { data = 0x8; }

    static std::vector<float> LUT = {0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6};
    static std::vector<uint8_t> D_LUT = {0x0,  0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7,
                                         0x08, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf};

    // find the closet value in LUT
    auto it = std::lower_bound(LUT.begin(), LUT.end(), val);

    if (it == LUT.begin()) {
      data = D_LUT.front();
    } else if (it == LUT.end()) {
      data = D_LUT.back();
    } else {
      int idx = it - LUT.begin();
      if (std::abs(LUT[idx] - val) < std::abs(LUT[idx - 1] - val)) {
        data = D_LUT[idx];
      } else {
        data = D_LUT[idx - 1];
      }
    }
  }

  // only process least 4bits
  explicit operator bf8() const {
    std::vector<uint8_t> LUT = {0x00, 0x38, 0x3c, 0x3e, 0x40, 0x42, 0x44, 0x46,
                                0x80, 0xb8, 0xbc, 0xbe, 0xc0, 0xc2, 0xc4, 0xc6};
    uint32_t idx = data & 0xf;
    uint8_t looked_val = LUT[idx];
    bf8 looked_val_bf8 = (*reinterpret_cast<bf8 *>(&looked_val));
    return looked_val_bf8;
  }

  // only process least 4bits
  explicit operator fp16() const {
    std::vector<uint16_t> LUT = {0x0000, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400, 0x4600,
                                 0x8000, 0xb800, 0xbc00, 0xbe00, 0xc000, 0xc200, 0xc400, 0xc600};
    uint32_t idx = data & 0xf;
    uint16_t looked_val = LUT[idx];
    fp16 looked_val_fp16 = (*reinterpret_cast<fp16 *>(&looked_val));
    return looked_val_fp16;
  }

  explicit operator float() const {
    constexpr float FNAN = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> LUT = {0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6};
    uint32_t idx = data & 0xf;
    float looked_val = LUT[idx];
    return looked_val;
  }
};

struct fp6_e3m2 {
  uint8_t data;

  fp6_e3m2() = default;

  explicit fp6_e3m2(float val) {
    int sign = (val < 0) ? 1 : 0;
    val = std::fabs(val);
    int exponent = 0;
    while (val >= 2.0f) {
      val /= 2.0f;
      exponent++;
    }
    while (val < 1.0f && exponent > -3) {
      val *= 2.0f;
      exponent--;
    }
    int mantissa = static_cast<int>((val - 1.0f) * 4.0f);
    data = (sign << 5) | ((exponent + 3) << 2) | mantissa;
  }

  explicit operator float() const {
    int sign = (data & 0x20) ? -1 : 1;
    int exponent = ((data >> 2) & 0x07) - 3;
    float mantissa = 1.0f + (data & 0x03) / 4.0f;
    return sign * mantissa * std::pow(2.0f, exponent);
  }

  explicit operator double() const { return static_cast<float>(*this); }

  friend std::ostream &operator<<(std::ostream &os, const fp6_e3m2 &e) {
    os << static_cast<float>(e);
    return os;
  }
};

// FP6_E2M3 struct definition
struct fp6_e2m3 {
  uint8_t data;

  fp6_e2m3() = default;

  explicit fp6_e2m3(float val) {
    int sign = (val < 0) ? 1 : 0;
    val = std::fabs(val);
    int exponent = 0;
    while (val >= 2.0f) {
      val /= 2.0f;
      exponent++;
    }
    while (val < 1.0f && exponent > -1) {
      val *= 2.0f;
      exponent--;
    }
    int mantissa = static_cast<int>((val - 1.0f) * 8.0f);
    data = (sign << 5) | ((exponent + 1) << 3) | mantissa;
  }

  explicit operator float() const {
    int sign = (data & 0x20) ? -1 : 1;
    int exponent = ((data >> 3) & 0x03) - 1;
    float mantissa = 1.0f + (data & 0x07) / 8.0f;
    return sign * mantissa * std::pow(2.0f, exponent);
  }

  explicit operator double() const { return static_cast<float>(*this); }

  friend std::ostream &operator<<(std::ostream &os, const fp6_e2m3 &e) {
    os << static_cast<float>(e);
    return os;
  }
};

struct mxint8 {
  int8_t data;

  mxint8() = default;

  explicit mxint8(float val) { data = val * 64; }

  explicit operator float() const { return data / 64.0f; }

  explicit operator double() const { return data / 64.0; }

  friend std::ostream &operator<<(std::ostream &os, const mxint8 &e) {
    os << static_cast<float>(e.data / 64.0f);
    return os;
  }
};

static_assert(sizeof(mxint8) == 1);

class fp2_S1E1M0 {
  using storage_t = uint8_t;
  storage_t value;

  public:
  fp2_S1E1M0() = default;
  fp2_S1E1M0(const fp2_S1E1M0 &) = default;
  ~fp2_S1E1M0() = default;

  fp2_S1E1M0(storage_t val) : value(val) {}

  fp2_S1E1M0 &operator=(const storage_t &rhs) {
    value = rhs;
    return *this;
  }

  explicit operator bf8() const {
    std::vector<uint8_t> LUT = {0x00, 0x3c, 0x80, 0xbc};
    uint32_t idx = value & 0x3;
    uint8_t looked_val = LUT[idx];
    bf8 looked_val_bf8 = (*reinterpret_cast<bf8 *>(&looked_val));
    return looked_val_bf8;
  }

  explicit operator hf8() const {
    std::vector<uint8_t> LUT = {0x00, 0x38, 0x80, 0xb8};
    uint32_t idx = value & 0x3;
    uint8_t looked_val = LUT[idx];
    hf8 looked_val_hf8 = (*reinterpret_cast<hf8 *>(&looked_val));
    return looked_val_hf8;
  }

  explicit operator bf16() const {
    std::vector<uint16_t> LUT = {0x0000, 0x3f80, 0x8000, 0xbf80};
    uint32_t idx = value & 0x3;
    uint16_t looked_val = LUT[idx];
    bf16 looked_val_bf16 = (*reinterpret_cast<bf16 *>(&looked_val));
    return looked_val_bf16;
  }

  explicit operator fp16() const {
    std::vector<uint16_t> LUT = {0x0000, 0x3c00, 0x8000, 0xbc00};
    uint32_t idx = value & 0x3;
    uint16_t looked_val = LUT[idx];
    fp16 looked_val_fp16 = (*reinterpret_cast<fp16 *>(&looked_val));
    return looked_val_fp16;
  }

  storage_t raw() const { return value; }

  bool operator==(const fp2_S1E1M0 &rhs) { return value == rhs.raw(); }

  bool operator!=(const fp2_S1E1M0 &rhs) { return value != rhs.raw(); }

  operator uint8_t() const { return value; }
};

class int4_t {
  using storage_t = uint8_t;
  storage_t data;

  public:
  int4_t() = default;
  int4_t(const int4_t &) = default;
  ~int4_t() = default;

  int4_t(storage_t val) : data(val) {}

  int4_t &operator=(const storage_t &rhs) {
    data = rhs;
    return *this;
  }

  explicit operator int8_t() const {
    std::vector<int8_t> LUT = {0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1};
    uint32_t idx = data & 0xf;
    return LUT[idx];
  }

  storage_t value() const { return data & 0xf; }

  storage_t raw() const { return data; }

  bool operator==(const int4_t &rhs) { return data == rhs.raw(); }

  bool operator!=(const int4_t &rhs) { return data != rhs.raw(); }

  operator uint8_t() const { return data; }
};

class int2_t {
  using storage_t = int8_t;
  storage_t data;

  public:
  int2_t() = default;
  int2_t(const int2_t &) = default;
  ~int2_t() = default;

  int2_t(storage_t val) : data(val) {}

  int2_t &operator=(const storage_t &rhs) {
    data = rhs;
    return *this;
  }

  explicit operator int8_t() const {
    std::vector<int8_t> LUT = {0, 1, -0, -1};
    uint32_t idx = data & 0x3;
    return LUT[idx];
  }

  explicit operator int4_t() const {
    std::vector<int4_t> LUT = {0, 1, -0, -1};
    uint32_t idx = data & 0x3;
    return LUT[idx];
  }

  storage_t value() const { return data & 0x3; }

  storage_t raw() const { return data; }

  bool operator==(const int2_t &rhs) { return data == rhs.raw(); }

  bool operator!=(const int2_t &rhs) { return data != rhs.raw(); }

  operator uint8_t() const { return data; }
};

} // namespace impl

struct e8m0 {
  static constexpr int bias = 127;
  uint8_t data;

  e8m0() = default;

  explicit e8m0(uint8_t val) { this->data = val; }

  uint8_t value() const { return data; }

  explicit e8m0(float val) {
    if (std::isnan(val)) {
      data = 0xff;
      return;
    }
    // get the exp bits of the float
    // Mask to extract the exponent bits (bits 23 to 30)
    uint32_t expMask = 0x7F800000;
    uint32_t *valBits = reinterpret_cast<uint32_t *>(&val);
    data = ((*valBits & expMask) >> 23);
  }

  explicit operator float() const {
    if (data == 0xff) { return std::numeric_limits<float>::quiet_NaN(); }
    return sycl::pow(2.0f, float(data) - bias);
  }

  explicit operator double() const { return static_cast<double>(static_cast<float>(*this)); }

  // the function for << operator
  friend std::ostream &operator<<(std::ostream &os, const e8m0 &e) {
    os << static_cast<float>(e);
    return os;
  }
};

static_assert(sizeof(e8m0) == 1);

} // namespace base_type

using bf8 = base_type::bf8;
using hf8 = base_type::hf8;
using bf16 = base_type::bf16;
using fp16 = base_type::fp16;
using tf32 = base_type::tf32;
using e8m0 = base_type::e8m0;

#ifdef USE_IFPWRAPPER
#include <ifp_wrapper.hpp>
using fp4_e2m1 = ifpwrp::fp4_e2m1;
using mxint8 = ifpwrp::mxint8;
using fp6_e3m2 = ifpwrp::fp6_e3m2;
using fp6_e2m3 = ifpwrp::fp6_e2m3;
#else
using fp4_e2m1 = base_type::impl::fp4_e2m1;
using mxint8 = base_type::impl::mxint8;
using fp6_e3m2 = base_type::impl::fp6_e3m2;
using fp6_e2m3 = base_type::impl::fp6_e2m3;
#endif
using fp2_e1m0 = base_type::impl::fp2_S1E1M0;
using int2_t = base_type::impl::int2_t;
using int4_t = base_type::impl::int4_t;

template <typename T>
struct type_repr;

template <typename T>
constexpr auto type_repr_v = type_repr<T>::value;

#define DEFINE_TYPE_REPR(Type, Name) \
  template <> \
  struct type_repr<Type> { \
    static constexpr std::string_view value = Name; \
  }

DEFINE_TYPE_REPR(float, "FP32");
DEFINE_TYPE_REPR(double, "FP64");
DEFINE_TYPE_REPR(uint64_t, "U64");
DEFINE_TYPE_REPR(uint32_t, "U32");
DEFINE_TYPE_REPR(uint16_t, "U16");
DEFINE_TYPE_REPR(uint8_t, "U8");
DEFINE_TYPE_REPR(bf8, "BF8");
DEFINE_TYPE_REPR(hf8, "HF8");
DEFINE_TYPE_REPR(bf16, "BF16");
DEFINE_TYPE_REPR(fp16, "FP16");
DEFINE_TYPE_REPR(tf32, "TF32");
DEFINE_TYPE_REPR(e8m0, "E8M0");
DEFINE_TYPE_REPR(fp4_e2m1, "FP4E2M1");
DEFINE_TYPE_REPR(int32_t, "INT32");
DEFINE_TYPE_REPR(int8_t, "INT8");
DEFINE_TYPE_REPR(int4_t, "INT4");
DEFINE_TYPE_REPR(int2_t, "INT2");
DEFINE_TYPE_REPR(mxint8, "MXINT8");
DEFINE_TYPE_REPR(fp6_e3m2, "FP6E3M2");
DEFINE_TYPE_REPR(fp6_e2m3, "FP6E2M3");
