# Xe35 Chunkwise Gated DeltaNet (GDN) Attention

A direct CuTe-on-SYCL port of the Gated DeltaNet (GDN) **chunkwise
attention forward pass** from the vllm-xpu-kernels project for Intel Xe GPUs. The kernel runs as five raw
device kernels submitted back-to-back on a single in-order SYCL queue — there
is **no** CUTLASS collective/kernel scaffolding.

> ⚠️ **Status — initial port, not yet Xe3-optimized.**
> This is the **first step** of bringing the upstream Xe2 GDN kernels into the
> SYCL-TLA environment: getting them to **build and run correctly** here. The
> code is a straight functional port — it does **not** yet exploit Xe3 (CRI)
> hardware features, except for setting GRF size at compilation (CRI: 512 vs. BMG 256).
> Running it on CRI as-is is expected to work but **not** to
> be performant; a follow-up pass is needed to adapt the kernels (tiling,
> register-file usage, DPAS shapes, etc.) to take advantage of Xe3 before any
> CRI performance numbers are meaningful.

> **At a glance**
> - **What:** GDN chunkwise bf16 activations + fp32 SSM state.
> - **Targets:** Intel **CRI** (Xe-3.5) and **BMG** (Xe20).
> - **Origin:** a CuTe/SYCL chunkwise GDN attention implementation for Intel Xe, with a caller-visible ABI modeled on common GDN attention kernels.

## Contents

- [Where the code lives](#where-the-code-lives)
- [The five-stage pipeline](#the-five-stage-pipeline)
  - [Data flow through the stages](#data-flow-through-the-stages)
- [Work hierarchy & GPU mapping](#work-hierarchy--gpu-mapping)
- [Public API](#public-api)
  - [`GDNArguments` reference](#gdnarguments-reference)
    - [Symbols & dtypes](#symbols--dtypes)
    - [Shape fields](#shape-fields)
    - [Input tensors](#input-tensors)
    - [Output tensors](#output-tensors)
    - [Workspace tensors](#workspace-tensors)
  - [Return status](#return-status)
  - [Mutability contract](#mutability-contract)
  - [`has_initial_state` contract](#has_initial_state-contract)
- [Sizing the workspaces](#sizing-the-workspaces)
- [Integrating the launcher](#integrating-the-launcher)
- [Constraints](#constraints)
- [File map](#file-map)

## Where the code lives

The kernel itself is **header-only** and lives under
[`applications/gdn_attention/`](../../../applications/gdn_attention). Three
consumers drive and validate it:

| Consumer | Path | Role |
|---|---|---|
| Example | [`examples/14_xe35_gdn_attention/`](../../../examples/14_xe35_gdn_attention) | Driver + runner; verifies against the shared host reference |
| Benchmark | [`benchmarks/applications/03_gdn/`](../../../benchmarks/applications/03_gdn) | Google Benchmark harness + configuration sweep |
| Unit test | [`test/unit/gdn_attention/`](../../../test/unit/gdn_attention) | GoogleTest coverage (`cutlass_test_unit_gdn_attention_chunkwise`) |

See the full [file map](#file-map) below for every header and its purpose.


## The five-stage pipeline

Each call to
[`cutlass::gdn::chunk_gated_delta_rule_launch<T, StateT>`](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule.hpp)
submits five device kernels on one in-order SYCL queue — no host wait between
stages.

| # | Stage | What it computes |
|---|---|---|
| 1 | `chunk_prepare` | L2-normalize Q (scaled by `1/sqrt(D)`) and K in place; cumulative-sum the per-token log-scale gate into `a[t] = cumsum(softplus(a + dt_bias) * -exp(A_log))`. |
| 2 | `chunk_compute_A` | Build the lower-triangular transition matrix `L[m,n] = (K_m·K_n) * exp(a[m] - a[n]) * b[m]` per chunk. |
| 3 | `chunk_inverse` | Invert `L` in place  |
| 4 | `chunk_compute_wu` | `U = L^-1 * V * diag(b)` and `W = L^-1 * K * diag(exp(a) * b)`. |
| 5 | `chunk_fwd_o` | `O = Q * S^T * exp(g) + O2 * U`; update SSM state `S_out = exp(g_last) * S_prev + U^T * K_scaled`. |

**Workspaces.** Three workspace tensors are supplied by the caller
(`A_workspace`, `w_workspace`, `u_workspace`); element counts come from
[`cutlass::gdn::get_workspace_sizes()`](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule.hpp).
They are sized by `total_virtual_seqlen` (the chunk-padded extent), **not**
`total_seqlen` (real tokens), because that is the stride the kernels use when
indexing them.

### Data flow through the stages

Each stage reads the tensors produced upstream and writes the next. `q`/`k`/`a`
are mutated **in place** (see [mutability contract](#mutability-contract)); the
`A`/`w`/`u` workspaces are scratch buffers reused only within a launch.

```
  STAGE              READS                          WRITES
  ─────              ─────                          ──────
  1 prepare          q, k, a, A_log, dt_bias        q, k, a            (in place)
        │
        ▼
  2 compute_A        k, a, b                        A  (L, lower-tri)  [workspace]
        │
        ▼
  3 inverse          A                              A  (L^-1)          [workspace, in place]
        │
        ▼
  4 compute_wu       A, q, k, v, b, a, A_log,       w (W), u (U)       [workspace]
        │            dt_bias, has_initial_state
        ▼
  5 fwd_o            A, w, u, q, b, a, ssm_state    core_attn_out  [total_seqlen, n_vh, hv]
                                                    ssm_state      (recurrent state, in place)

  Legend:  in place = mutates a caller tensor   ·   workspace = A/w/u scratch (per launch)
```

## Work hierarchy & GPU mapping

GDN decomposes the problem along three nested axes, and the kernels map them
onto the Xe execution hierarchy. Understanding this mapping explains the launch
geometry of each stage and the `xe_core_count` floor.

```
  PROBLEM DECOMPOSITION                         Xe EXECUTION HIERARCHY
  ─────────────────────                         ──────────────────────

  batch (variable-length sequences)
   └─ v_head           (num_v_heads)            sycl::nd_range<3>
       └─ chunk        (seq padded to kChunkSize)│
           └─ token    (kChunkSize = 64)         ├─ work-group  (Xe-core)
               └─ head_dim element               │   └─ sub-group (16 lanes, 1 DPAS row)
                                                 │       └─ work-item (SIMD lane)
  Tile math per chunk: kChunkSize×kChunkSize (64x64) transition matrix,
  inverted as a 4×4 grid of 16×16 DPAS blocks.
```

**How each stage is launched.** All five stages share one in-order queue.
`xe_core_count` is the device Xe-core count (floored — see [Constraints](#constraints));
`MaxThreadsPerXeCore == 512`, `sub_group_size == 16`.

```
  Stage               grid (global, dim-1 axis)        work-group   work split
  ──────────────────  ───────────────────────────────  ───────────  ─────────────────────────
  1 chunk_prepare     xe_core_count                     512 threads  1 sub-group ↦ (v_head,chunk);
                                                                      persistent loop over chunks
  2 chunk_compute_A   xe_core_count·512 / wg_size       MMA wg_size  1 work-group ↦ one chunk
  3 chunk_inverse     max(xe_core_count·512/16,         16 (1 sub-   1 work-group ↦ (chunk,v_head);
                          ⌈tvs/64⌉·num_v_heads)         group)       4×4 block forward-substitution
  4 chunk_compute_wu  xe_core_count·512 / wg_size       MMA wg_size  1 work-group ↦ (v_head,chunk)
  5 chunk_fwd_o       grid = (batch, num_v_heads)       MMA wg_size  1 work-group ↦ (batch,v_head);
                                                                      sequential scan over chunks
```

Stages 1–4 launch a *persistent* grid (sized to fill the device) and use an
internal `while` loop to stride over all `(v_head, chunk)` pairs, so a short
grid still covers every unit of work. Stage 5 instead launches exactly one
work-group per `(batch, v_head)` because the SSM-state recurrence must walk that
head's chunks **in order**.

> **Why the `xe_core_count` floor matters here:** stage 1 derives each sub-group's
> `(v_head, chunk)` assignment from `total_sg_range / num_v_heads`. If a
> degenerate `xe_core_count` (e.g. `1` on the CRI simulator) makes the total
> sub-group count smaller than `num_v_heads`, that division underflows to zero
> and per-head work is silently dropped. The floor guarantees at least one
> sub-group per v-head.

## Public API

To **launch** the kernel, include the launch header — it carries the inline
launcher definition and pulls in the device kernels. The lightweight public
header `xe35_chunk_gated_delta_rule.hpp` only *declares* the entry point and is
for host-only consumers that just need `GDNArguments` / `get_workspace_sizes`.

```cpp
#include "gdn_attention/xe35_chunk_gated_delta_rule_launch.hpp"

cutlass::gdn::GDNArguments args = ...;          // shapes + raw device pointers
auto sizes = cutlass::gdn::get_workspace_sizes(args);
// caller allocates A_workspace / w_workspace / u_workspace
// of sizes.* elements of T

sycl::queue queue{sycl::gpu_selector_v,
                  sycl::property::queue::in_order()};

cutlass::Status s = cutlass::gdn::chunk_gated_delta_rule_launch<
    cutlass::bfloat16_t, float>(queue, args);
queue.wait_and_throw();
```

### `GDNArguments` reference

[`GDNArguments`](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule.hpp)
is one plain struct holding every input the kernel needs — all the shapes and
device pointers in one place. There is no builder, no handle, and no hidden
state: you fill in the shape scalars and raw device pointers, then pass it by
`const&` to the launcher. Default-initialize it
(`GDNArguments a{};`) so any field you forget is a null/zero you can catch,
rather than garbage.

The fields fall into four groups — [shape](#shape-fields),
[inputs](#input-tensors), [outputs](#output-tensors), and
[workspaces](#workspace-tensors). **All tensors are row-major.** The symbols and
dtypes used in the tables below are defined first, so you can read every layout
column without jumping ahead.

#### Symbols & dtypes

| Symbol | Definition |
|---|---|
| `T` | Activation dtype — `cutlass::bfloat16_t`. A `void*` field tagged `→ T` points at `T` elements. |
| `StateT` | SSM-state dtype — `float`. |
| `FP32` / `int32` / `bool` | Element types of the side arrays (gates, offsets, the `has_initial_state` flags). |
| `tvs` | `total_virtual_seqlen` — real tokens after padding **each** sequence to a multiple of 64. |
| `kChunkSize` | Fixed chunk length, **64** (compile-time constant). The 64 is the baseline Xe2 (BMG) value inherited from the upstream port, not retuned for Xe3. |
| `cache_batch` | Number of `ssm_state` slots the caller manages (leading extent of `ssm_state`). |

#### Shape fields

These scalars describe the problem. They must be consistent with the pointer
layouts below — the kernel trusts them and indexes accordingly.

| Field | Type | Meaning |
|---|---|---|
| `batch_size` | `int` | Number of sequences. Equals `query_start_loc.size() − 1`. |
| `total_seqlen` | `int` | Total **real** (unpadded) tokens across all sequences. Stride of `core_attn_out`. |
| `total_virtual_seqlen` | `int` | Total tokens after padding **each** sequence up to a multiple of `kChunkSize` (64). Stride of `q`/`k`/`v` and the workspaces. See [`tvs`](#symbols--dtypes). |
| `num_k_heads` | `int` | Number of key/query heads. Must divide `num_v_heads`. |
| `num_v_heads` | `int` | Number of value heads (the GQA-expanded count). |
| `head_k_dim` | `int` | Per-head Q/K dimension. Positive multiple of 64 (validated: 128). |
| `head_v_dim` | `int` | Per-head V/output dimension. Positive multiple of 64 (validated: 128). |
| `ssm_state_stride_0` | `int` | Element stride between batch slots in `ssm_state`. Typically `num_v_heads · head_v_dim · head_k_dim`, but may be larger for a padded/aligned slot layout (matches upstream `ssm_state.stride(0)`) |

> **Why two seqlen fields?** `total_seqlen` is what the model actually produced;
> `total_virtual_seqlen` is the chunk-padded extent the kernel iterates over.
> Outputs are indexed by the former, the chunkwise machinery (and every
> workspace) by the latter. Mixing them up is the most common sizing bug — hence
> [`get_workspace_sizes`](#sizing-the-workspaces) takes the padded one for you.

#### Input tensors

| Field | Type | Layout | In/Out |
|---|---|---|---|
| `q` | `void*` → `T` | `[tvs, num_k_heads, head_k_dim]` | **in/out** — L2-normalized & scaled in place by stage 1 |
| `k` | `void*` → `T` | `[tvs, num_k_heads, head_k_dim]` | **in/out** — L2-normalized in place by stage 1 |
| `v` | `const void*` → `T` | `[tvs, num_v_heads, head_v_dim]` | in |
| `b` | `const float*` | `[num_v_heads, tvs]` | in — per-token gate `b` |
| `a` | `float*` | `[num_v_heads, tvs]` | **in/out** — per-token log-scale gate; cumsum'd in place by stage 1 |
| `A_log` | `const float*` | `[num_v_heads]` | in — per-head log decay |
| `dt_bias` | `const void*` → `T` | `[num_v_heads]` | in — per-head timestep bias |
| `query_start_loc` | `const int*` | `[batch_size + 1]` | in — packed sequence boundaries (prefix-sum offsets) |
| `cache_indices` | `const int*` | `[batch_size]` | in — `ssm_state` slot for each batch |
| `has_initial_state` | `const bool*` | `[batch_size]` or `nullptr` | in — see [contract](#has_initial_state-contract) |

The three **in/out** pointers (`q`, `k`, `a`) are the [mutability
contract](#mutability-contract): snapshot them on the host first if you need the
originals afterward.

#### Output tensors

| Field | Type | Layout | Notes |
|---|---|---|---|
| `core_attn_out` | `void*` → `T` | `[total_seqlen, num_v_heads, head_v_dim]` | The attention output. Sized by **real** tokens. |
| `ssm_state` | `void*` → `StateT` | `[cache_batch, num_v_heads, head_v_dim, head_k_dim]` | Recurrent state, **read then written in place** per batch. |

`ssm_state`'s leading extent `cache_batch` is the caller's slot pool, **not** a
`GDNArguments` field: batch `batch_id` reads/writes slot
`cache_indices[batch_id]`, and `ssm_state_stride_0` is the element stride between
slots. This decouples the on-device state cache from the per-launch batch order
(the vLLM paged-cache pattern).

#### Workspace tensors

Scratch buffers, allocated by the caller, reused only within a single launch.
Size them with [`get_workspace_sizes`](#sizing-the-workspaces) — do not hand-roll
the arithmetic.

| Field | Type | Layout | Produced by → consumed by |
|---|---|---|---|
| `A_workspace` | `void*` → `T` | `[num_v_heads, tvs, kChunkSize]` | stage 2 (`L`) → stage 3 (`L⁻¹`) → stages 4–5 |
| `w_workspace` | `void*` → `T` | `[num_v_heads, tvs, head_k_dim]` | stage 4 (`W`) → stage 5 |
| `u_workspace` | `void*` → `T` | `[num_v_heads, tvs, head_v_dim]` | stage 4 (`U`) → stage 5 |

### Return status

`chunk_gated_delta_rule_launch` is **asynchronous**: it validates the problem
shape on the host, submits the five stages, and returns immediately. The caller
must `queue.wait_and_throw()` (or otherwise synchronize) before reading
`core_attn_out` / `ssm_state`. The returned `cutlass::Status` reflects only the
host-side validation that happens *before* dispatch:

| Condition | Result |
|---|---|
| `batch_size <= 0` or `total_virtual_seqlen <= 0` | `Status::kSuccess` (valid no-op — nothing is submitted) |
| Any of `num_k_heads / num_v_heads / head_k_dim / head_v_dim <= 0` | `Status::kErrorInvalidProblem` (guards the GQA modulo) |
| `num_v_heads % num_k_heads != 0` | `Status::kErrorInvalidProblem` |
| otherwise | `Status::kSuccess` (five stages submitted) |

A `kSuccess` return therefore means **submitted**, not **finished** — runtime
faults (e.g. a null required pointer) surface from `queue.wait_and_throw()`, not
from the return value.

Device pointers are **not** null-checked — a null required pointer faults
inside the kernels by design, matching the raw-pointer ABI of the upstream
entry point. `has_initial_state` is the one documented nullable pointer.

### Mutability contract

Stages 1+ mutate `q`, `k`, `a`, and `ssm_state` **in place** on the device.
The caller MUST NOT rely on their pre-launch contents after the call. The
example runner snapshots the originals into host vectors before the launch
precisely because the chunkwise host reference needs them to replay the
pipeline.

### `has_initial_state` contract

`GDNArguments::has_initial_state` is a nullable `const bool*` of length
`batch_size`

- **non-null:** `has_initial_state[batch_id]` selects whether stages 4-5 load
  the prior SSM state from `ssm_state[cache_indices[batch_id], ...]` or start
  from zero.
- **null:** treated as "every batch has carry-over state". The caller is then
  responsible for pre-zeroing the corresponding `ssm_state` slots before the
  launch — mirrors the Python idiom
  `initial_state[~has_initial_state, ...] = 0` in
  `vllm/model_executor/layers/mamba/gdn`.

Both kernel-side load sites null-check the pointer with
`(has_initial_state == nullptr) || has_initial_state[batch_id]`.

## Sizing the workspaces

[`get_workspace_sizes(args)`](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule.hpp)
returns a `GDNWorkspaceSizes` with the **element counts** (of `T`) for the three
scratch buffers. It reads only the shape fields, so you can call it on a
shape-only `GDNArguments` before any device pointer exists:

```cpp
struct GDNWorkspaceSizes {
  size_t A_elems;   // num_v_heads · total_virtual_seqlen · kChunkSize
  size_t w_elems;   // num_v_heads · total_virtual_seqlen · head_k_dim
  size_t u_elems;   // num_v_heads · total_virtual_seqlen · head_v_dim
};
```

```cpp
cutlass::gdn::GDNArguments shape = /* shape scalars only */;
auto ws = cutlass::gdn::get_workspace_sizes(shape);

// allocate `ws.A_elems` / `ws.w_elems` / `ws.u_elems` elements **of T**
// (e.g. bytes = ws.A_elems * sizeof(T)), then store the pointers back:
shape.A_workspace = /* device alloc of ws.A_elems × sizeof(T) */;
shape.w_workspace = /* device alloc of ws.w_elems × sizeof(T) */;
shape.u_workspace = /* device alloc of ws.u_elems × sizeof(T) */;
```

The counts use `total_virtual_seqlen` (the chunk-padded extent), because that is
exactly the stride the kernels apply when indexing the workspaces (e.g.
`v_head_id · total_virtual_seqlen · kChunkSize`). The example runner follows this
pattern: it builds a `make_arguments_shape_only()` struct, sizes the workspaces
from it, then fills in pointers in `make_arguments()`.

## Integrating the launcher

End-to-end, a caller does five things. Steps 1–2 are pure host work (no device
code), so they can live in a translation unit that includes only the lightweight
[`xe35_chunk_gated_delta_rule.hpp`](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule.hpp);
the launch in step 4 needs the [launch
header](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule_launch.hpp).

```cpp
#include "gdn_attention/xe35_chunk_gated_delta_rule_launch.hpp"

using T      = cutlass::bfloat16_t;   // activations
using StateT = float;                 // SSM state

// 1. Describe the problem (shape scalars).
cutlass::gdn::GDNArguments args{};
args.batch_size           = batch;
args.total_seqlen         = total_seqlen;            // real tokens
args.total_virtual_seqlen = total_virtual_seqlen;    // chunk-padded tokens
args.num_k_heads          = num_k_heads;
args.num_v_heads          = num_v_heads;             // % num_k_heads == 0
args.head_k_dim           = 128;                     // multiple of 64
args.head_v_dim           = 128;                     // multiple of 64
args.ssm_state_stride_0   = num_v_heads * head_v_dim * head_k_dim;

// 2. Size + allocate the three workspaces (host-only math).
auto ws = cutlass::gdn::get_workspace_sizes(args);
//   allocate ws.A_elems / ws.w_elems / ws.u_elems elements of T on the device

// 3. Fill in every device pointer (inputs, outputs, workspaces).
args.q = d_q;  args.k = d_k;  args.v = d_v;
args.b = d_b;  args.a = d_a;  args.A_log = d_A_log;  args.dt_bias = d_dt_bias;
args.query_start_loc = d_qsl;  args.cache_indices = d_cache;
args.has_initial_state = d_has_init;   // or nullptr (see contract)
args.core_attn_out = d_out;  args.ssm_state = d_state;
args.A_workspace = d_A;  args.w_workspace = d_w;  args.u_workspace = d_u;

// 4. Launch on an IN-ORDER queue and synchronize.
sycl::queue queue{sycl::gpu_selector_v, sycl::property::queue::in_order()};
cutlass::Status st = cutlass::gdn::chunk_gated_delta_rule_launch<T, StateT>(queue, args);
if (st != cutlass::Status::kSuccess) { /* bad problem shape — see Return status */ }

// 5. Wait, then read core_attn_out / ssm_state.
queue.wait_and_throw();
```

**Checklist before you launch**

- [ ] Queue created `in_order()` — non-negotiable ([Constraints](#constraints)).
- [ ] `num_v_heads % num_k_heads == 0` and all head dims are positive multiples of 64.
- [ ] `q`/`k`/`v`/`b`/`a` sized by `total_virtual_seqlen`; `core_attn_out` by `total_seqlen`.
- [ ] Workspaces sized via `get_workspace_sizes` (in elements of `T`).
- [ ] Originals of `q`/`k`/`a` snapshotted if you still need them — they are clobbered.
- [ ] `ssm_state` slots for batches **without** carry-over pre-zeroed when `has_initial_state == nullptr`.

## Constraints

- **`kChunkSize == 64` is baked in.** The value 64 is the baseline Xe2 (BMG)
  choice inherited from the upstream port
  ([`vllm-xpu-kernels` `csrc/xpu/gdn_attn/xe_2`](https://github.com/vllm-project/vllm-xpu-kernels/tree/main/csrc/xpu/gdn_attn/xe_2));
  it has **not** been retuned for Xe3 (CRI). Mirrored in the kernel namespace as
  `cutlass::gdn::detail::chunk_size`. Sequences are padded per batch to a
  multiple of 64. Changing the constant requires touching every fixed-trip
  inner loop in the kernels.
- **`head_k_dim` / `head_v_dim` must be positive multiples of `kChunkSize`
  (64).** The runner and benchmark reject any other value; `128 / 128` is the
  only configuration currently validated end-to-end. The `compute_wu` and `fwd_o` stages
  walk the head dims in 64-wide column tiles, so a non-multiple would silently
  drop the trailing `head_dim % 64` columns rather than fault. Supporting other
  dims is a partial-tile change to those loops, not an algorithmic one.
- **Correctness is validated only for full-chunk sequences.** Per-batch
  `seq_len` **must be a multiple of `kChunkSize`** (currently 64: valid lengths
  are 64, 128, 256, 512, ...). The kernel will not fault on non-multiples
  (padding is applied automatically), but numerical correctness is **not
  guaranteed**: partial-tail sequences where `seq_len % kChunkSize != 0` (e.g.,
  200 = 3×64 + 8-row tail) exhibit differences between the kernel output and
  both the recurrent and chunkwise fp32 reference oracles on some seeds (~0.05%
  of SSM state elements exceed the 5% tolerance band, typically within 10-20%).
  Accordingly, the example driver, benchmark, and unit test all use only
  chunk-aligned `seq_len` values.
- **In-order queue required.** The launcher submits five stages without any
  host wait; an out-of-order queue would corrupt the pipeline.
- **No framework dependency.** Inputs are raw `void*` device pointers; this
  kernel does NOT plug into the CUTLASS collective/kernel `GemmUniversal`
  scaffolding. It is intentionally a direct CuTe-on-SYCL kernel.
- **`xe_core_count` floor.** Stage 1b (prepare-A reduction) needs one subgroup per
  v-head. On targets that report a degenerate `xe_core_count` (e.g. the CRI
  simulator with `xe_core_count = 1`), the launcher raises `xe_core_count` to the
  minimum that keeps
  `xe_core_count * (MaxThreadsPerXeCore / sub_group_size) >= num_v_heads`. Without this
  floor, the chunk-range arithmetic would underflow to zero and drop per-head
  work silently.


- the public `GDNArguments` wrapper (this repo's preferred ABI) and its
  validation gates,
- the `xe_core_count` floor described above.

## File map

The headers form two layers: a lightweight **API surface** that host-only code
can include freely, and the **device layer** that pulls in SYCL kernel code.
Callers that launch include the launcher; callers that only need shapes include
the public header.

```
  CONSUMERS                              GDN HEADERS (applications/gdn_attention/)
  ─────────                              ─────────────────────────────────────────

  examples/14_xe35_gdn_attention ─┐
  benchmarks/applications/03_gdn ──┼─include─▶ xe35_chunk_gated_delta_rule_launch.hpp
  test/unit/gdn_attention ────────┘            (inline launcher: validate + dispatch)
                                                  │ includes
                                                  ▼
                                             xe35_chunk_gated_delta_rule_kernels.hpp
                                             (5 device kernels + kernel_launcher)
                                                  │ uses
                                                  ▼
                                             xe35_chunk_gated_delta_rule_gemm.hpp
                                                  (CuTe GEMM helpers)

  host reference / shape-only ──include─▶ xe35_chunk_gated_delta_rule.hpp
                                          (public API: GDNArguments,
                                           get_workspace_sizes — NO device code)
```

| File | Purpose |
|---|---|
| [xe35_chunk_gated_delta_rule.hpp](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule.hpp) | Lightweight public API: `GDNArguments`, `get_workspace_sizes`, `chunk_gated_delta_rule_launch` declaration |
| [xe35_chunk_gated_delta_rule_launch.hpp](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule_launch.hpp) | Header-only launcher: argument validation + `chunk_gated_delta_rule_launch<T, StateT>` definition (inline template, instantiated at each call site) |
| [xe35_chunk_gated_delta_rule_kernels.hpp](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule_kernels.hpp) | Five device kernels + `detail::kernel_launcher` (upstream-aligned signature) |
| [xe35_chunk_gated_delta_rule_gemm.hpp](../../../applications/gdn_attention/xe35_chunk_gated_delta_rule_gemm.hpp) | CuTe GEMM helpers (`gemm_TTS`, `gemm_STS`, `gemm_TSS`, `gemm_TTS_k_multi`) |
| [xe35_gdn_attention_stage_references.hpp](../../../tools/util/include/cutlass/util/reference/host/xe35_gdn_attention_stage_references.hpp) | Per-stage host reference implementations (in `cutlass/util/reference/host/`) shared by the example and unit test |
| [xe35_gdn_attention_compare.hpp](../../../tools/util/include/cutlass/util/reference/host/xe35_gdn_attention_compare.hpp) | Host verification comparator (`compare_with_stats`/`print_compare_stats`, in `cutlass/util/reference/host/`) used by the example runner |
| [examples/14_xe35_gdn_attention/](../../../examples/14_xe35_gdn_attention) | Example driver, runner, and perf helpers (verifies via the shared host reference above) |
| [benchmarks/applications/03_gdn/](../../../benchmarks/applications/03_gdn) | Google Benchmark harness, configuration sweep |
| [test/unit/gdn_attention/](../../../test/unit/gdn_attention) | GoogleTest unit coverage (`cutlass_test_unit_gdn_attention_chunkwise`) |
