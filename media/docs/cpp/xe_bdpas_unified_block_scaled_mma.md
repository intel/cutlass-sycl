# FP8 Block-Scaled Grouped GEMM with Flexible Block Size and Scale Type (PR #324)

PR #324 adds support for FP8 block-scaled grouped GEMM with **arbitrary block sizes and
scale data types**, motivated by DeepSeek MoE inference on Intel Xe. This document explains
the motivation, design and implementation for contributors.

---

## Motivation

DeepSeek models use FP8 blockwise scaling for MoE (Mixture-of-Experts) GEMM with a block
size of 128 and `float` scale factors (see [sglang reference kernel](https://github.com/sgl-project/sglang/blob/f72a77038f7f176906f7c2a7458bcefaf2ec3b2c/sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu#L468)).

Before PR #324, the only block-scaled GEMM policy on Intel Xe was
`MainloopIntelXeXMX16BlockScaledGroup`, which hard-codes a block size of **32** and
requires **`ue8m0` (8-bit exponent)** scale factors — matching the MX microscaling
standard used by models like GPT-OSS. This policy dispatches to the hardware `bdpas`
instruction which enforces these constraints at the silicon level.

To support DeepSeek-style MoE we need:
- Block size **128** (not 32)
- Scale type **`float`** (not 8-bit `ue8m0`)
- `float_e4m3_t` operands
- Grouped GEMM (for MoE expert dispatch)

None of these are supported by the existing hardware `bdpas` path, so PR #324 introduces a
new software-emulated path alongside it.

---

## New: Software-Scaled DPAS Path

The new path issues a standard `dpas` instruction and applies scale factors in FP32
software afterwards. It is selected by passing a `cute::tuple<M,N,K>` as the `GroupSize`
in the dispatch policy — as opposed to `cute::Int<N>` which selects the existing hardware
path.

| | Existing hardware path | **New software path (PR #324)** |
|---|---|---|
| Dispatch policy `GroupSize` | `cute::Int<32>` | **`cute::tuple<1,128,128>`** (or any M,N,K) |
| Scale type | `ue8m0` (8-bit) | **`float`** (or half, bfloat16, …) |
| Block size | 32 (fixed by MX standard) | **Arbitrary** |
| ISA instruction | `bdpas` | `dpas` + FP32 multiply |
| Target use case | GPT-OSS / MX models | **DeepSeek MoE FP8** |
| CollectiveMma | `xe_blockscaled_mma.hpp` | **`xe_mma_blockscaled_fp8.hpp`** (new) |

### Usage

```cpp
// DeepSeek-style: block size 128, float scales, float_e4m3_t operands
using TiledMma = TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, float_e4m3_t>>, ...>::TiledMMA;

// cute::tuple GroupSize → new xe_mma_blockscaled_fp8.hpp mainloop
// <1, 128, 128>: per-row A scaling, 128-column and 128-K scale blocks for B
using GroupSizeMNK = cute::tuple<cute::_1, cute::Int<128>, cute::Int<128>>;
using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16BlockScaledGroupImpl<Stages, GroupSizeMNK>;

using CollectiveMainloop = CollectiveMma<GEMMDispatchPolicy, TileShape,
    cute::tuple<float_e4m3_t, float>, StridePairA,   // (FP8 data, float scale)
    cute::tuple<float_e4m3_t, float>, StridePairB,
    TiledMma, ...>;
```

---

## Implementation Details

### MMA Atom: Reusing `XE_BDPAS_TT` for Both Paths

Rather than introducing yet another atom type, the new path reuses `XE_BDPAS_TT` with a
dual-dispatch `mma_unpack` in `MMA_Traits<XE_BDPAS_TT>`. The mainloop signals which
execution path to use via the **arity of the zip tensor** passed to `cute::gemm`:

```
include/cutlass/gemm/collective/xe_mma_blockscaled_mxfp.hpp  →  make_zip_tensor(data, scale, m_offset, k_offset)  →  arity 4
include/cutlass/gemm/collective/xe_mma_blockscaled_fp8.hpp   →  make_zip_tensor(data, scale)                      →  arity 2
```

`mma_unpack` reads this at compile time:

```cpp
// include/cute/atom/mma_traits_xe.hpp
constexpr auto zip_arity = tuple_size<decltype(unzipped_A)>::value;
constexpr bool use_hardware_bdpas = (zip_arity == 4);

if constexpr (use_hardware_bdpas) {
    // emits bdpas instruction (hardware MX path, unchanged)
} else {
    // emits dpas, then: out[i] = dpas_result[i] * float(SFA[i]) * float(SFB[i]) + C[i]
}
```

### New Mainloop: `xe_mma_blockscaled_fp8.hpp`

The new `CollectiveMma` specialisation is selected when `GroupSize` is a `cute::tuple`:

```cpp
// Specialisation condition (dispatch_policy.hpp + xe_mma_blockscaled_fp8.hpp)
MainloopIntelXeXMX16BlockScaledImpl<Stages, cute::tuple<GroupSizeM, GroupSizeN, GroupSizeK>, Schedule>
```

Key properties:
- Supports `GroupSizeM == 1` (per-element-row A scaling) and arbitrary N/K group sizes.
- Scale tensors are loaded with **bounds-checking** to handle problem sizes that are not
  multiples of the tile:
  - B-scale coordinate is clamped: `cute::min(n_coord / GroupN, N_scale_extent - 1)`
  - A-scale falls back to `0.0f` for out-of-bounds rows (those accum lanes are unused by the epilogue)

---

## Component Map

| File | Role |
|---|---|
| `include/cute/atom/mma_traits_xe.hpp` | Added dual-path `mma_unpack` to `MMA_Traits<XE_BDPAS_TT>` |
| `include/cutlass/gemm/collective/xe_mma_blockscaled_fp8.hpp` | New mainloop for software-scaled path |
| `include/cutlass/gemm/dispatch_policy.hpp` | `MainloopIntelXeXMX16BlockScaledImpl<Stages, cute::tuple<M,N,K>>` wires to the new mainloop |
| `examples/51_xe35_block_scaled_grouped_gemm/51_xe35_block_scaled_grouped_gemm_fp8_e4m3.cpp` | End-to-end example for DeepSeek-style block-128 FP8 MoE GEMM |

---

## Testing

`51_xe35_block_scaled_grouped_gemm_fp8_e4m3` is the primary new test target. All 4 targets
in the `51`-series and all 5 targets in the `50`-series (existing hardware path, regression
check) compile and pass `--verify=1` on the CRI target.
