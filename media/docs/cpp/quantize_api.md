# Block-wise Quantization for SYCL-TLA

## Motivation

### Problem Statement

Current quantization workflows require:
1. Store high-precision GEMM output to global memory
2. Launch separate quantization kernel
3. Load quantized data for next operation

This incurs unnecessary global memory traffic and kernel launch overhead.

### Solution

Provide a `quantize_block_wise` API that:
- Operates directly on `SubgroupTensor` (data in GPU registers)
- Enables fusion with GEMM prologue/epilogue
- Supports MX floating-point formats (MXFP8, MXFP4) and native FP8 with flexible scale types (E8M0, BF16, FP32)
- Leverages hardware-specific instructions optimized for each architecture

### Target fusion patterns

| Pattern | Data flow | When |
|---------|-----------|------|
| **Post-op** (epilogue) | GEMM → Activation → **Quantize** → Store | FFN projections, attention output |
| **Pre-op** (prologue) | Load → Norm → **Quantize** → GEMM | Activation re-quantization before GEMM |

Both patterns keep the high-precision → low-precision conversion entirely in registers. The quantized tensor and its scale factors are written to global memory (post-op) or fed directly to the next DPAS instruction (pre-op) without an extra kernel launch.

---

## API and Implementation Considerations

### 1. API

```cpp
template <int BlockSize,
          class SrcEngine, class SrcLayout, class SrcTVLayout,
          class DstEngine, class DstLayout, class DstTVLayout,
          class ScaleEngine, class ScaleLayout, class ScaleTVLayout>
CUTE_HOST_DEVICE
void quantize_block_wise(
    SubgroupTensor<SrcEngine, SrcLayout, SrcTVLayout> const& src,
    SubgroupTensor<DstEngine, DstLayout, DstTVLayout>      & dst,
    SubgroupTensor<ScaleEngine, ScaleLayout, ScaleTVLayout> & scale
);
```

**File location:** `include/cute/quantization/quantize.hpp` (CuTe
level, reusable across epilogue, prologue, and standalone kernels).

**Design considerations:**

- **Why `dst` and `scale` are input parameters, not return values.**  The caller
  constructs `dst` and `scale` with the desired TV layout *before* calling
  `quantize_block_wise`. This lets the caller choose layouts that match the next
  operation — a store-friendly layout for an epilogue (`XE_2D_STORE`) or an
  MMA-friendly layout for a prologue (`XE_DPAS`). Returning them would force the
  API to pick a layout, adding an extra `reorder()` later.
- **Rounding modes** are hardcoded: **round-toward-zero** for scales,
  `cutlass::FloatRoundStyle::round_to_nearest` for elements.
- All intermediate computation uses **float32**. This matches the MX spec and
  has no performance penalty on Xe (native 32-bit FPU).
- The current implementation uses the software FPU path (`NumericConverter` +
  `reduce_over_group`). A future path can dispatch to `tcvdmx` (TensorPipe
  block-quantize instruction) on architectures that support it.

**Template parameters:**

| Parameter | Description |
|-----------|-------------|
| `BlockSize` | Elements per quantization block (compile-time; must be ≥ `sg_size` = 16) |

**Supported types**:

| Role | Types |
|------|-------|
| Source (`src`) | `bfloat16_t`, `half_t`, `float` |
| Destination (`dst`) | `float_e4m3_t`, `float_e5m2_t`, `float_e2m1_t` |
| Scale (`scale`) | `float_ue8m0_t`, `bfloat16_t`, `float` |

**Shape constraints:**

| Tensor | Logical shape |
|--------|--------------|
| `src` | $(M, N)$ |
| `dst` | $(M, N)$ — must match `src` |
| `scale` | $(M, N / \text{BlockSize})$ — N/BlockSize scales per row |


### 2. Algorithm

The core algorithm is:

1. Compute per-block `amax = max(abs(x_i))` for all elements in the block.
2. Compute `scale = round_toward_zero(target_max(DstType) / amax)`.
3. Quantize: `dst_i = round_to_nearest(x_i * scale)` (using `cutlass::FloatRoundStyle::round_to_nearest`, i.e., round-to-nearest ties-to-even).
4. Reorder elements from source TV layout to destination TV layout via `reorder()`.


#### Quantize along N → needs cross-lane reduction

Consider a 2×32 logical tensor with `sg_size = 16` and `BlockSize = 32`.  The TV layout distributes columns across threads:

```
Logical (M=2, N=32):

         n=0  n=1  n=2  ...  n=15 │ n=16  n=17  ...  n=31
       ┌─────────────────────────────┼──────────────────────────┐
  m=0  │ T0   T1   T2  ...  T15   │ T0    T1   ...   T15     │
  m=1  │ T0   T1   T2  ...  T15   │ T0    T1   ...   T15     │
       └─────────────────────────────┴──────────────────────────┘
                              one quantization block (32 cols)

  Thread 0 owns: (0,0), (1,0), (0,16), (1,16)  — 4 values
  Thread 1 owns: (0,1), (1,1), (0,17), (1,17)  — 4 values
  ...
```

A block of 32 elements along N spans **all 16 threads**.  No single thread sees every element in a block, so computing `amax` requires:
1. **Phase 1 (vertical):** each thread computes a local abs-max over its private values belonging to each (row, block) pair — no cross-lane communication.
2. **Phase 2 (horizontal):** `reduce_over_group` across the subgroup to produce the block-wide `amax` — one cross-lane reduction per (row, block).

This two-phase design minimises expensive shuffles to exactly one `reduce_over_group` call per (row, block) pair.

#### Reorder step

After quantization the element bit-width shrinks (e.g. 16-bit → 8-bit), so the number of elements packed per 32-bit register doubles or quadruples.  The implementation constructs a temporary `SubgroupTensor` with the *source* TV layout and calls `reorder()` to transform it into the caller-supplied *destination* TV layout.  `reorder()` handles cross-lane shuffles, packing, and layout conversion.

#### Preconditions

All of the above leads to the following preconditions. Some are validated via `static_assert` in `quantize.hpp`, while others (notably the thread-stride constraint) are currently usage requirements that are **not** checked at compile time:

- **`(BlockSize % sg_size) == 0`** — each quantization block must be multiple of subgroup width so the compile-time block-index derivation from `tv_layout(0, v)` is valid.
- **`BlockSize` evenly divides the quantized dimension** (N).
- **Thread stride maps only to dimension 1 (N)** — the row index (M) must be thread-independent. This is what makes `tv_layout(0, v)` yield correct (m, block_id) coordinates for all threads; this constraint is required for correctness but is not currently enforced by a `static_assert`.
- **Source and destination have the same logical shape.**
- **Scale shape matches the blocking** — $(M, N/\text{BlockSize})$.

## Performance Analysis (GEMM+Quantize vs. GEMM-only)

This section analyzes the theoretical performance ratio of GEMM+quantize
(post-GEMM quantization to FP8) versus standalone GEMM on Intel Xe GPUs.

### Overview

The GEMM+quantize kernel appends a post-GEMM quantization phase that converts
the F32 accumulator to FP8 output plus per-block scale factors. Unlike a true
epilogue (e.g., bias add or activation), the quantize phase is **not fused** into
the GEMM main loop — it runs sequentially after all k_tiles are done, operating
on the in-register accumulators. This trades:

- **Additional compute**: per-block abs-max reduction + scale + type conversion
- **Reduced memory traffic**: FP8 output (1 byte) + scales vs FP32 output (4 bytes)

### Instruction Counts

#### GEMM Main Loop

Per subgroup (SG) tile: 32×64 output, BF16 DPAS, WG tile = 256×256, k_tile = 32.

| Category                  | Instructions/k_tile | Pipeline  | How |
|---------------------------|--------------------:|-----------|-----|
| DPAS (8×16×16 atoms)      |                  32 | Systolic (XMX) | Each DPAS produces an 8×16 output. SG tile rows: 32/8=4, cols: 64/16=4 → 16 DPAS per K-atom. k_tile has 32/16=2 K-atoms → 16×2=32. |
| 2D block load A + B       |                6–8  | Memory    | `load_2d` loads up to 32 rows × 16 BF16 cols (16 is the max col width for BF16, set by the hardware 2D block load unit at 32 bytes = 16 × 2B). A tile is 32×32 BF16 → 32/16=2 loads; B tile is 32×64 BF16 → 64/16=4 loads; total 6. Up to 8 when VNNI transform at load time halves the effective max width, requiring sub-tile splits. |
| Reorder A + B             |              16–24  | ALU       | Each loaded block (6–8 from above) needs ~2–3 `mov` instructions to permute registers from the load layout into the packed layout DPAS expects (e.g., VNNI interleave for B). 6 blocks × 3 movs = 18 (low); 8 blocks × 3 = 24 (high). |
| Prefetch                  |                  6-8 | Memory    | Same as 2D block load |
| Barrier, loop overhead    |                  ~4 | ALU       | 4 separate instructions: (1) `barrier` to sync all threads in the WG before next k_tile, (2) `add` to increment k_offset by 32, (3) `cmp` to test k_offset < K, (4) `jmp` to branch back to loop top. |
| **Total per k_tile**      |       **~64–76**    |           | |

For K=1024 → 32 k_tiles → **~2048–2432 total instructions**, of which **1024 are DPAS**.
For K=2048 → 64 k_tiles → **~4096–4864 total instructions**, of which **2048 are DPAS**.

On real hardware, DPAS runs on the systolic array while ALU/memory overlap on
separate pipes. DPAS dominates (32 instructions/k_tile vs 20–28 ALU and 12–16
memory), so effective GEMM time per SG tile ≈ K cycles. Each subgroup computes
only its own 32×64 tile; the full M×N output is covered by launching
M/32 × N/64 subgroups across the GPU in parallel.

#### Post GEMM Quantization

**Setup**: The SG tile is 32×64 = 2048 output values in F32 accumulators.
With SIMD-16 subgroups, each thread holds 128 F32 values across 128 GRF slots.
Per-block quantization groups these into blocks of B values that share one scale
factor. The typical block size is B=32 (two SIMD-16 registers per block).

Total blocks per SG tile: 2048 / 32 = 64 blocks.

##### Theoretical per-block operations

| Step | Operation | Theoretical min | How |
|------|-----------|----------------:|-------|
| 1 | **Abs-max reduction** | 6 | `(abs)` folds into source modifier (free). Tree-reduce across 32 values: log₂(32)=5 `max` ops + 1 broadcast. |
| 2 | **Scale computation** | 3 | `max`(ε guard) + `inv`(reciprocal) + `mul`(max_fp8). Fixed cost per block. |
| 3 | **Scale values** | 2 | 2 `mul` instructions (one per SIMD-16 register, covering all 32 values). |
| 4 | **Type convert to FP8** | 2–4 | Best case: 2 `fcvt` F32→FP8 (one per register). Worst case: 2 `mov` F32→HF + 2 `fcvt` HF→FP8 (no direct F32→FP8 path). |
| 5 | **Store scale** | 1 | 1 `mov` to write the scale factor to output. |
| **Per block total** | | **14–16** | Theoretical floor assuming perfect register layout. |

##### Why HF/BF source is more expensive than F32 source

When the GEMM accumulator is in F32 but the quantize input type is HF/BF16
(e.g., after an explicit down-convert before quantization), two extra costs arise:

| Extra cost | Instructions | Reason |
|------------|------------:|--------|
| Type widening | +2 per block | 2 `mov` HF/BF→F32 (one per register) before abs-max (can't fold `(abs)` and type-convert in the same `max`) |
| Non-contiguous layout | +2–4 per block | Gather `mov`s if HF/BF values sit in alternating half-registers rather than contiguous F32 slots |
| **HF/BF overhead** | **+4–6** | |

This gives **~18–22 instructions/block** for HF/BF source vs **~14–16** for F32 source.

##### Theoretical SG tile totals

For 64 blocks per 32×64 SG tile:

| Source type | Per block (theoretical) | × 64 blocks | Reorder to store layout | **Total** |
|-------------|------------------------:|------------:|------------------------:|----------:|
| F32         | 14–16                   | 896–1024    | ~96 `mov`               | **~992–1120** |
| HF/BF       | 18–22                   | 1152–1408   | ~96 `mov`               | **~1248–1504** |

Note: The reorder cost (~96 `mov`) accounts for permuting the FP8 output from
the per-block processing order into the 2D store layout expected by `store_2d`.
This is a fixed cost per tile regardless of quantization algorithm.

### Performance Ratio

$$R = \frac{T_{GEMM}}{T_{GEMM} + T_{quant}}$$

DPAS runs on the systolic array; quantize runs on ALU sequentially after GEMM.

$$T_{GEMM} \approx K \text{ cycles}, \quad T_{quant} \approx 1376 \text{ cycles (HF/BF)} \text{ or } 1120 \text{ cycles (F32)}$$

| K | T_GEMM | R (HF/BF) | R (F32) |
|------:|-------:|-----------:|---------:|
| 1024 | 1024 | 42.7% | 47.8% |
| 2048 | 2048 | 59.8% | 64.7% |
| 4096 | 4096 | 74.8% | 78.5% |
| 8192 | 8192 | 85.6% | 88.0% |

### Measured Results

CRI, F32→E5M2, optimized quantize implementation.

| # | M | K | N | GEMM-only (TF/s) | GEMM+quant (TF/s) | Measured R |
|--:|------:|------:|------:|------:|------:|------:|
| 1 | 1024 | 1024 | 1024 | 180.9 | 142.2 | 78.6% |
| 2 | 2048 | 2048 | 2048 | 213.6 | 186.1 | 87.1% |
| 3 | 4096 | 4096 | 4096 | 225.7 | 208.9 | 92.6% |
| 4 | 512 | 1024 | 256 | 22.8 | 17.8 | 78.0% |
| 5 | 512 | 2048 | 256 | 26.3 | 21.3 | 80.9% |
| 6 | 512 | 4096 | 256 | 27.4 | 24.8 | 90.4% |
| 7 | 512 | 8192 | 256 | 28.4 | 27.1 | 95.6% |
| **Geomean** | | | | | | **85.9%** |

Across all measured shapes the measured ratio is **higher** than the
theoretical prediction, with the largest gap at small K and a narrowing gap
as K grows. This is because the theoretical model assumes $T_{GEMM} = K$
(perfect DPAS saturation). In practice, non-DPAS overhead (load latency stalls,
barrier waits, cache misses) makes $T_{GEMM} > K$. A larger $T_{GEMM}$ increases
R because the quantize cost becomes a smaller fraction of total time. This
overhead is proportionally larger at small K, explaining why the gap narrows
as K grows.
