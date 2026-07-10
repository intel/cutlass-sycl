![ALT](https://raw.githubusercontent.com/intel/sycl-tla/main/media/images/gemm-hierarchy-with-epilogue-no-labels.png "Complete GEMM decomposition")

# SYCL\* Templates for Linear Algebra (SYCL\*TLA)

**This repository is forked from the NVIDIA CUTLASS repository and extends CUTLASS and CuTe API support to Intel GPUs through SYCL enablement.**
*This project was previously referred to as CUTLASS-SYCL, you may see references to CUTLASS-SYCL in the code and documentation.*
*For SYCL support instructions, refer to the [SYCL build documentation](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/build/building_with_sycl_support.md)*

*SYCL is a trademark of the Khronos Group Inc, Other names and brands may be claimed as the property of others.*
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/intel/sycl-tla/badge)](https://scorecard.dev/viewer/?uri=github.com/intel/sycl-tla)

SYCL\*TLA is a modular, header‑only C++ template framework for high‑performance 
GEMM, and fused epilogue kernels. It applies hierarchical tiling, composable policy 
abstractions, and efficient data‑movement primitives to build flexible, reusable 
building blocks for dense linear algebra. The SYCL implementation brings those 
optimizations to Intel GPUs with tuned kernels for modern execution units and memory 
hierarchies. It adds mixed‑precision and epilogue fusion pathways designed to 
simplify integrating advanced quantization and post‑processing into custom pipelines.

To support a wide variety of applications, SYCL\*TLA provides extensive
support for mixed-precision computations on Intel hardware, providing
specialized data-movement and multiply-accumulate abstractions for FP64, FP32,
FP16, BF16, 8b floating point types (E5M2 and E4M3 for FP8), narrow integer
types (4 and 8b signed and unsigned integers with support for zero-point
quantization), and mixed-precision operations with tensor-wise, channel-wise,
and group-wise quantization support. SYCL\*TLA demonstrates optimal matrix
multiply operations targeting Intel's programmable, high-throughput execution
units implemented in Intel Data Center GPU Max/Flex Series (Intel Xe
architecture, codename: Ponte-Vecchio) and Intel Arc B580 GPUs.

See the [Quick Start Guide](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/quickstart.md) to get started quickly.

See the [functionality docs](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/functionality.md) for a more comprehensive
list of kernel level features, data types, instructions, and minimum supported by CUTLASS on each GPU
architecture.

This project fast follows NVIDIA CUTLASS releases to ensure parity of APIs and features.

Base NVIDIA CUTLASS Versions for SYCL*TLA releases:
| SYCL*TLA | NVIDIA CUTLASS |
|-----------------|----------|
|0.1| 3.9|
|0.2 | 3.9.2 |
|0.3 | 3.9.2 |
|0.5 | 4.2.0 |
|0.6 | 4.2.0 |
|0.7 | 4.2.1 |
|0.8 | 4.2.1 |
|0.9 | 4.2.1 |
|0.9.1 | 4.2.1 |

# What's New in SYCL*TLA 0.9.1

## [SYCL*TLA 0.9.1](https://github.com/intel/sycl-tla/releases/tag/v0.9.1) (2026-06-11)
### Enhancements
- **Support Stream-K GEMM ops in Python API ([#800](https://github.com/intel/sycl-tla/pull/800))**
- **Support fast path for LinearCombination in xe_epilogue ([#802](https://github.com/intel/sycl-tla/pull/802))**
- **Support event-less launch when profiling is disabled in GemmUniversalAdapter ([#803](https://github.com/intel/sycl-tla/pull/803))**
- **Support subbyte reorder ([#793](https://github.com/intel/sycl-tla/pull/793))**
- **Add Handler-less and Event-less support in launch APIs ([#794](https://github.com/intel/sycl-tla/pull/794))**
- **Add SYCL subgroup lane index to canonical_lane_idx() ([#816](https://github.com/intel/sycl-tla/pull/816))**
- **Add memory-budget-based bounded buffer for EventManager ([#795](https://github.com/intel/sycl-tla/pull/795))**
- **Add more PyTorch GEMM configs ([#789](https://github.com/intel/sycl-tla/pull/789))**
- **Reverse Q scheduling order in FMHA tile scheduler ([#814](https://github.com/intel/sycl-tla/pull/814))**
- **Refine SLM r2s/s2r to reuse UniversalCopy without vectorization ([#776](https://github.com/intel/sycl-tla/pull/776))**
- **Refine barrier API ([#810](https://github.com/intel/sycl-tla/pull/810))**
- **Drop redundant __INTEL_LLVM_COMPILER checks ([#801](https://github.com/intel/sycl-tla/pull/801))**

### Bug Fixes
- **Fix shapes parameter issue in the MoE grouped GEMMs ([#820](https://github.com/intel/sycl-tla/pull/820))**
- **Fix average runtime and GFLOPS calculation in example 10 ([#818](https://github.com/intel/sycl-tla/pull/818))**
- **Fix rem mask of SDPA ([#813](https://github.com/intel/sycl-tla/pull/813))**
- **Fix int32 overflow in MoE GEMM for large expert counts ([#804](https://github.com/intel/sycl-tla/pull/804))**
- **Fix sub-byte pointer arithmetic and zero buffer allocation in grouped GEMM ([#790](https://github.com/intel/sycl-tla/pull/790))**
- **Fix wrong constexpr/lifetime evaluation ([#799](https://github.com/intel/sycl-tla/pull/799))**

### Documentation
- **Align build commands across README ([#788](https://github.com/intel/sycl-tla/pull/788))**
- **Update googlebenchmark to v1.9.5 ([#797](https://github.com/intel/sycl-tla/pull/797))**
- **Update PyTorch commit for cutlass-inductor workflow ([#808](https://github.com/intel/sycl-tla/pull/808))**
- **Update inductor workflow ([#806](https://github.com/intel/sycl-tla/pull/806))**
    
**See the [CHANGELOG](https://github.com/intel/sycl-tla/blob/main/CHANGELOG-SYCL.md) for details of all past releases and updates.**

# CuTe

SYCL\*TLA supports the newly introduced core library, CuTe, to describe and manipulate tensors of threads and data.
CuTe in SYCL\*TLA is a collection of C++ SYCL template abstractions for
defining and operating on hierarchically multidimensional layouts of threads and data.
CuTe provides `Layout` and `Tensor` objects that compactly package the type,
shape, memory space, and layout of data, while performing the complicated indexing for the user.
This lets programmers focus on the logical descriptions of their algorithms while
CuTe does the mechanical bookkeeping for them. With these tools, we can quickly design,
implement, and modify all dense linear algebra operations.

The core abstractions of CuTe are hierarchically multidimensional layouts
which can be composed with data arrays to represent tensors.
The representation of layouts is powerful enough to represent nearly
everything we need to implement efficient dense linear algebra.
Layouts can also be combined and manipulated via functional composition, on which we build a large set of common operations such as tiling and partitioning.

SYCL\*TLA and beyond adopts CuTe throughout the GEMM hierarchy in its templates.
This greatly simplifies the design and improves code composability and readability.
More documentation specific to CuTe can be found in its
[dedicated documentation directory](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/cute/00_quickstart.md).

# Compatibility

Minimum requirements:

- Architecture: Intel Data Center GPU Max Series (codename: Ponte-Vecchio)
- Compiler: Must support at least C++17
- DPC++ Compiler Version: oneAPI 2025.1 and onwards
- Intel Compute Runtime and Graphics Compiler: 
  - For Intel Data Center GPU Max Series: Runtime from [LTS driver installation guide](https://dgpu-docs.intel.com/driver/installation-lts2.html), IGC [v2.27.10](https://github.com/intel/intel-graphics-compiler/releases/tag/v2.27.10) or later

## Hardware Support

SYCL*TLA runs successfully on the following Intel GPUs.

|**GPU**|**Intel GPU Architecture**
|---|---|
|Intel Data Center GPU Max Series            |Xe-HPC|
|Intel Arc GPU B580 Graphics                       |Xe2|

## Validated Software Configurations

We are regularly testing following setup in CI.

|**Platform**|**Operating System** | **DPC++ Compiler** | **G++** | **Intel Compute Runtime** |**Intel Graphics Compiler** |
|-----------------|----------|-----------------|--------|---------------------|-----------------------|
|Xe-HPC| Ubuntu 24.04 |2025.3+ |G++13  | 25.48 | 2.24 |
|Xe2| Ubuntu 25.04 |2025.3+  |G++13  | 26.01 | 2.27 |





## Target Architecture

The target architecture information is passed on to SYCL*TLA via the cmake flag
`DPCPP_SYCL_TARGET`. 

```
cmake .. -DDPCPP_SYCL_TARGET="intel_gpu_pvc"
```
Or

```
cmake .. -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g21" 
```

Or

```
cmake .. -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g31" 
```

> Note: `-DDPCPP_SYCL_TARGET="bmg"` will compile for both `intel_gpu_bmg_g21`, `intel_gpu_bmg_g31` targets.

Please refer to the [functionality documentation](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/functionality.md)
for details on which kernels require which target architectures.

# Documentation

SYCL*TLA is described in the following documents.

- [Quick Start Guide](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/quickstart.md) - basics of building and running SYCL*TLA
- [Functionality](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/functionality.md) - summarizes functionality available in SYCL*TLA
- [Efficient GEMM in CUDA](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/efficient_gemm.md) - describes how GEMM kernels may be implemented efficiently in CUDA
- [CUTLASS 3.x Design](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/cutlass_3x_design.md) - describes the CUTLASS 3.x design, its benefits, and how CuTe enables us to write much more composable components
- [GEMM API 3.x](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/gemm_api_3x.md) - describes the CUTLASS 3.x GEMM model and C++ template concepts
- [Implicit GEMM Convolution](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/implicit_gemm_convolution.md) - describes 2-D and 3-D convolution in CUTLASS
- [Code Organization](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/code_organization.md) - describes the organization and contents of the CUTLASS project
- [Terminology](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/terminology.md) - describes terms used in the code
- [Programming Guidelines](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/programming_guidelines.md) - guidelines for writing efficient modern CUDA C++
- [Fundamental types](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/fundamental_types.md) - describes basic C++ classes used in CUTLASS to represent numeric quantities and arrays
- [Layouts](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/layout.md) - describes layouts of matrices and tensors in memory
- [Tile Iterators](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/tile_iterator_concept.md) - describes C++ concepts for iterating over tiles of matrices in memory
- [CUTLASS Utilities](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/utilities.md) - additional templates used to facilitate rapid development

# Resources


# Building SYCL*TLA

SYCL*TLA is a header-only template library and does not need to be built to be used by other
projects. Client applications should target SYCL*TLA's `include/` directory in their include
paths.

SYCL*TLA unit tests, examples, and utilities can be built with CMake.
The minimum version of CMake is given in the [Quickstart guide](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/quickstart.md).
Make sure you have Intel oneAPI DPC++ compiler installed and the environment is properly set up.

```bash
$ source /opt/intel/oneapi/setvars.sh
```

Create a build directory within the SYCL*TLA project, then run CMake. You need to specify
the target Intel GPU architecture using the `DPCPP_SYCL_TARGET` flag.
For Intel Data Center GPU Max Series (Ponte Vecchio), use `intel_gpu_pvc`.
For Intel Arc GPU B580 Graphics, use `intel_gpu_bmg_g21`.
For Intel Arc GPU Battlemage (G31), use `intel_gpu_bmg_g31`.

```bash
$ mkdir build && cd build

$ CC=icx CXX=icpx cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET="intel_gpu_pvc"     # compiles for Intel Data Center GPU Max Series
```

Or for Intel Arc GPU B580 Graphics:

```bash
$  CC=icx CXX=icpx cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g21" # compiles for Intel Arc GPU B580 Graphics
```

Or for Intel Arc GPU Battlemage (G31):

```bash
$  CC=icx CXX=icpx cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g31" # compiles for Intel Arc GPU Battlemage (G31)
```

To compile with G++ as host compiler, add the flag `-DDPCPP_HOST_COMPILER=g++-13` to the cmake command. Please note that the build system must be able to find `g++-13` in your PATH.

```bash
$  CC=icx CXX=icpx cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_HOST_COMPILER=g++-13 -DDPCPP_SYCL_TARGET="intel_gpu_bmg_g21" # compiles for Intel Arc GPU B580 Graphics with G++ as host compiler
```

From the `build/` directory, compile and run the SYCL*TLA unit tests by building the target `test_unit` with make.

The unit tests are organized as several binaries mirroring the top-level namespaces of SYCL*TLA,
and they may be executed in parallel via Ninja's `-j` command line argument.

```bash
$ ninja test_unit -j$(nproc)
...
...
...
[----------] Global test environment tear-down
[==========] XXX tests from YY test cases ran. (ZZZZ ms total)
[  PASSED  ] XXX tests.
```

All tests should pass on supported Intel GPU platforms, though the exact number of tests may vary over time.


# Project Structure

SYCL*TLA is arranged as a header-only library along with Utilities, Tools, Examples, and unit tests.

A detailed explanation of the source code organization may be found in the
[SYCL*TLA documentation](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/code_organization.md), but several main components are summarized below.

## SYCL*TLA

```
include/                     # client applications should target this directory in their build's include paths

  cutlass/                   # SYCL Templates for Linear Algebra Subroutines and Solvers - headers only

    arch/                    # direct exposure of Intel GPU architecture features (including instruction-level GEMMs)

    conv/                    # code specialized for convolution on Intel GPUs

    epilogue/                # code specialized for the epilogue of gemm/convolution using SYCL

    gemm/                    # code specialized for general matrix product computations with SYCL

    layout/                  # layout definitions for matrices, tensors, and other mathematical objects in memory

    platform/                # SYCL-capable Standard Library components for Intel GPUs

    reduction/               # bandwidth-limited reduction kernels optimized for Intel GPU architectures

    thread/                  # SYCL workgroup and subgroup code for Intel GPU execution units
    
    transform/               # code specialized for layout, type, and domain transformations using SYCL

    *                        # core vocabulary types, containers, and basic numeric operations

  cute/                      # CuTe Layout, layout algebra, MMA/Copy atoms, tiled MMA/Copy for SYCL

    algorithm/               # Definitions of core operations such as copy, gemm, and operations on cute::tuples

    arch/                    # Intel GPU architecture wrapper structs for copy and math instructions

    atom/                    # Meta-information for Intel GPU operators and SYCL kernels

      mma_atom.hpp           # cute::Mma_Atom and cute::TiledMma for Intel GPU architectures

      copy_atom.hpp          # cute::Copy_Atom and cute::TiledCopy optimized for SYCL

      *xe*.hpp               # Intel Xe architecture specific meta-information for copy and math operations

    *                        # Core library types such as Shape, Stride, Layout, Tensor, and associated operations

```

### SYCL*TLA Examples

[SYCL*TLA examples](https://github.com/intel/sycl-tla/tree/main/examples) apply SYCL*TLA templates to implement basic computations.

### Tools

```
tools/
  library/                   # SYCL*TLA Instance Library - contains instantiations of all supported SYCL*TLA templates
    include/
      cutlass/
        library/

  profiler/                  # Profiler                 - SYCL support not yet available
                             #                            (command-line utility for executing operations)
  
  util/                      # Utilities               - contains numerous helper classes for
    include/                 #                            managing tensors in Intel GPU device memory, reference
      cutlass/               #                            implementations for SYCL GEMM, random initialization
        util/                #                            of tensors, and I/O for Intel GPU environments.
```

### Test

The `test/unit/` directory consist of unit tests implemented with Google Test that demonstrate
basic usage of Core API components and complete tests of the CUTLASS GEMM computations.

Instructions for building and running the Unit tests are described in the [Quickstart guide](https://github.com/intel/sycl-tla/blob/main/media/docs/cpp/quickstart.md).

# About

SYCL*TLA is released by INTEL Corporation as Open Source software under the
[3-clause "New" BSD license](https://github.com/intel/sycl-tla/blob/main/LICENSE.txt).

# Contributors

The official list of SYCL*TLA developers and contributors is available here: [CONTRIBUTORS](https://github.com/intel/sycl-tla/blob/main/CONTRIBUTORS.md).

# Contributing

## Pull Request Templates

We provide concise PR templates to streamline documentation:

### Quick Start

**GitHub CLI:**
```bash
gh pr create --template .github/PULL_REQUEST_TEMPLATE/bug_fix.md
gh pr create --template .github/PULL_REQUEST_TEMPLATE/performance.md
gh pr create --template .github/PULL_REQUEST_TEMPLATE/feature.md
gh pr create --template .github/PULL_REQUEST_TEMPLATE/refactoring.md
```

**GitHub Web:** Add `?template=<name>.md` to PR URL (e.g., `?template=bug_fix.md`)

### Which Template?

- 🐛 **Bug fixes** → `bug_fix.md` - Root cause + verification
- ⚡ **Performance** → `performance.md` - Profiling data + benchmarks
- ✨ **Features** → `feature.md` - API design + examples
- 🔨 **Refactoring** → `refactoring.md` - Refactored/Redesigned code
- 📝 **Mixed/Other** → Default template

See [`.github/PULL_REQUEST_TEMPLATE`](https://github.com/intel/sycl-tla/tree/main/.github/PULL_REQUEST_TEMPLATE) for details.

# Copyright

Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2025 Intel Corporation. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
