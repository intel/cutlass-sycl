# SYCL*TLA (previously referred to as cutlass-sycl) Changelog

## [SYCL*TLA 0.9.2](https://github.com/intel/sycl-tla/releases/tag/v0.9.2) (2026-07-17)
### Enhancements
- **Support CRI platform ([#833](https://github.com/intel/sycl-tla/pull/833))**
- **Fix register spills in XeRowBroadcast/XeColBroadcast ([#815](https://github.com/intel/sycl-tla/pull/815))**
- **Fix identity fast path double-rounding and add EpilogueTile divisibility guard ([#832](https://github.com/intel/sycl-tla/pull/832))**

### Bug Fixes
- **Fix NumericConverter for SYCL device code (float→int32/int8/uint8) ([#817](https://github.com/intel/sycl-tla/pull/817))**
- **Fix inductor UTs workflow ([#823](https://github.com/intel/sycl-tla/pull/823))**
- **Fix BMG EVT workflow to run on any BMG runner ([#831](https://github.com/intel/sycl-tla/pull/831))**
- **Fix bare ushort in host-context templates on Windows ([#835](https://github.com/intel/sycl-tla/pull/835))**
  
### Documentation
- **Clean up NV references, make public repo Intel-first ([#821](https://github.com/intel/sycl-tla/pull/821))**

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

## [SYCL*TLA 0.9](https://github.com/intel/sycl-tla/releases/tag/v0.9) (2026-04-30)
### Enhancements
- **Support FP8 upconversion for CuTe SLM Copy ([#772](https://github.com/intel/sycl-tla/pull/772))**
- **Support AOT (Ahead-Of-Time) compilation instead of JIT ([#763](https://github.com/intel/sycl-tla/pull/763))**
- **Add vectorized test cases for CuTe SLM Copy ([#766](https://github.com/intel/sycl-tla/pull/766))**
- **Add some GEMM and Flash Attention benchmark cases ([#773](https://github.com/intel/sycl-tla/pull/773))**

### Bug Fixes
- **Fix example06 memory bandwidth computation bugs ([#778](https://github.com/intel/sycl-tla/pull/778))**
- **Fix Python GEMM Generation bugs ([#768](https://github.com/intel/sycl-tla/pull/768))**
- **Fix Python EVT test cases bugs ([#762](https://github.com/intel/sycl-tla/pull/762))**
- **Fix AOT multitarget support bugs ([#765](https://github.com/intel/sycl-tla/pull/765))**

## [SYCL*TLA 0.8](https://github.com/intel/sycl-tla/releases/tag/v0.8) (2026-03-25)
### Major Architecture Changes
- **Support BMG G31 Platform ([#755](https://github.com/intel/sycl-tla/pull/755))**
- **SLM Copy API functionalities and examples**
  - Support CuTe copy engines for 1D LDSM/STSM operations with vISA ([#753](https://github.com/intel/sycl-tla/pull/753))
  - Enable fusion example of 2 matmul operations through SLM Copoy API ([#747](https://github.com/intel/sycl-tla/pull/747))
  - Enable subgroup specialization example with SLM Copy API ([#735](https://github.com/intel/sycl-tla/pull/735))
- **Support default sub-byte reorder for low-precision data types ([#709](https://github.com/intel/sycl-tla/pull/709))**

### Enhancements
- **Flash Attention Performance Improvements (for BMG and BF16)**:  
  - Fix long context OOM issue ([#728](https://github.com/intel/sycl-tla/pull/728))
  - Overall performance improved from ~45% to ~78% of peak([#728](https://github.com/intel/sycl-tla/pull/728), [#743](https://github.com/intel/sycl-tla/pull/743),[#749](https://github.com/intel/sycl-tla/pull/749),[#750](https://github.com/intel/sycl-tla/pull/750))
  - Refine code and fix bugs ([#715](https://github.com/intel/sycl-tla/pull/715), [#716](https://github.com/intel/sycl-tla/pull/716),[#720](https://github.com/intel/sycl-tla/pull/720))
- **Epilogue Visitor Tree (EVT) Enhancements**:
  - Combine with SIGMOID function ([#686](https://github.com/intel/sycl-tla/pull/686))
  - Add Relu variation test cases ([#693](https://github.com/intel/sycl-tla/pull/693))
  - Refine code ([#703](https://github.com/intel/sycl-tla/pull/703), [#717](https://github.com/intel/sycl-tla/pull/717))
- **GEMM Enhancements**:
  - Support all GEMM tile shapes ([#738](https://github.com/intel/sycl-tla/pull/738))
  - Enhance examples ([#726](https://github.com/intel/sycl-tla/pull/726))

## [SYCL*TLA 0.7](https://github.com/intel/sycl-tla/releases/tag/v0.7) (2026-01-28)
### Major Architecture Improvements
- **Epilogue Visitor Tree (EVT) Support ([#647](https://github.com/intel/sycl-tla/pull/647))**: EVT support for Intel Xe architecture
  - Python EVT Compute support ([#650](https://github.com/intel/sycl-tla/pull/650))
  - XeAuxLoad for EVT support ([#674](https://github.com/intel/sycl-tla/pull/674))
  - XeAuxStore for EVT C++ support ([#691](https://github.com/intel/sycl-tla/pull/691))
  - Python EVT AuxStore support ([#698](https://github.com/intel/sycl-tla/pull/698))
  - XeRowReduction, XeColReduction and XeScalarReduction for EVT ([#680](https://github.com/intel/sycl-tla/pull/680))
  - XeRowBroadcast and XeColBroadcast for EVT ([#690](https://github.com/intel/sycl-tla/pull/690))
  - XeTopologicalVisitor for Python EVT ([#694](https://github.com/intel/sycl-tla/pull/694))
  - EVT Python test caching ([#689](https://github.com/intel/sycl-tla/pull/689))
  - Enable EVT mixed and layout tests ([#704](https://github.com/intel/sycl-tla/pull/704))
- **Rearchitected Xe Epilogue ([#621](https://github.com/intel/sycl-tla/pull/621))**: Complete redesign of epilogue architecture
  - Updated to use new MMA/Atom APIs ([#643](https://github.com/intel/sycl-tla/pull/643))
  - Refactored xe_array_epilogue.hpp to inherit from base class ([#688](https://github.com/intel/sycl-tla/pull/688))
  - Updated epilogue tests to use new MMA/Atom APIs ([#654](https://github.com/intel/sycl-tla/pull/654))
  - Use newer version of copy_atom in epilogue collective ([#573](https://github.com/intel/sycl-tla/pull/573))
- **Shared Local Memory (SLM) Support ([#673](https://github.com/intel/sycl-tla/pull/673))**: New SLM copy helper functions
  - Inline assembly for SLM load/store operations ([#677](https://github.com/intel/sycl-tla/pull/677))
 
### Enhancements
- **Flash Attention Performance Improvements ([#679](https://github.com/intel/sycl-tla/pull/679))**: Significant performance gains
  - 1.17x speedup for BF16 Flash Attention
  - 1.93x speedup for FP8 Flash Attention
- **Flash Attention Feature Enhancements**:
  - Support for cached KV and paged KV in new Flash Attention kernel ([#661](https://github.com/intel/sycl-tla/pull/661))
  - VarLen support for new Flash Attention API ([#616](https://github.com/intel/sycl-tla/pull/616))
  - CausalMask support with new Flash Attention API ([#604](https://github.com/intel/sycl-tla/pull/604))
  - FP8 Flash Attention support on BMG ([#613](https://github.com/intel/sycl-tla/pull/613))
  - Persistent SDPA kernel ([#608](https://github.com/intel/sycl-tla/pull/608))
- **Column Major Support ([#656](https://github.com/intel/sycl-tla/pull/656))**: Added support for Column Major C [Bias] in GEMM
- **MoE/Grouped GEMM Enhancements**:
  - MoE/grouped GEMM refinements ([#638](https://github.com/intel/sycl-tla/pull/638))
- **KCooperative Dispatch Policy ([#646](https://github.com/intel/sycl-tla/pull/646))**: New dispatch policy with unit tests and CMake changes ([#651](https://github.com/intel/sycl-tla/pull/651))
- **Build System Improvements**:
  - Support multiple targets in DPCPP_SYCL_TARGET ([#630](https://github.com/intel/sycl-tla/pull/630))
  - Lazy import DPCTL remove hard dependency of DPCTL ([#701](https://github.com/intel/sycl-tla/pull/701))
  - Suppress Build warnings ([#624](https://github.com/intel/sycl-tla/pull/624))
- **Reorder Operations**:
  - Support for broadcasting reorders ([#589](https://github.com/intel/sycl-tla/pull/589))
  - Reorder API changes ([#639](https://github.com/intel/sycl-tla/pull/639))
  - Reorder cleanup ([#614](https://github.com/intel/sycl-tla/pull/614))
- **Performance Optimizations**:
  - Improved performance for upconversion cases in xe_gemm ([#605](https://github.com/intel/sycl-tla/pull/605))
 
### Test Improvements
- **CuTe API Tests ([#658](https://github.com/intel/sycl-tla/pull/658))**: Comprehensive tests for new CuTe APIs
- **CUTLASS API Tests ([#642](https://github.com/intel/sycl-tla/pull/642))**: Extended CUTLASS API test coverage
- **MMA Unit Tests ([#557](https://github.com/intel/sycl-tla/pull/557))**: Complete rewrite of MMA unit tests
- **Prefetch, Transpose and VNNI Tests ([#632](https://github.com/intel/sycl-tla/pull/632))**: Unit tests for prefetch transpose and VNNI operations
- **CI Infrastructure**:
  - Updated CI workflows ([#672](https://github.com/intel/sycl-tla/pull/672))
  - Enabled new runners ([#663](https://github.com/intel/sycl-tla/pull/663))
  - CI driver issue fixes ([#676](https://github.com/intel/sycl-tla/pull/676))
- **Benchmark Improvements**:
  - Enabled GEMM benchmark workflow with new MMA atom ([#659](https://github.com/intel/sycl-tla/pull/659))
 
### Bug Fixes
- **Flash Attention Fixes**:
  - Fixed Flash Attention KV cache and prefill issues ([#617](https://github.com/intel/sycl-tla/pull/617))
- **CuTe Fixes**:
  - Fixed atom partitioning in some edge cases ([#628](https://github.com/intel/sycl-tla/pull/628))
- **Build and Compilation**:
  - Fixed CMake path issue ([#700](https://github.com/intel/sycl-tla/pull/700))
  - Fixed MMA unit test failure ([#687](https://github.com/intel/sycl-tla/pull/687))
- **Epilogue Fixes**:
  - Fixed void ElementC in epilogue ([#590](https://github.com/intel/sycl-tla/pull/590))
 
### Examples
- **MoE/Grouped GEMM Examples**:
  - Example of BF16/FP16 MoE Grouped GEMM with CuTe interface ([#600](https://github.com/intel/sycl-tla/pull/600))
  - Bug fix in CuTe interface MoE GEMM example ([#648](https://github.com/intel/sycl-tla/pull/648))
- **GEMM Examples**:
  - StreamK and mixed dtype examples with new atom API ([#665](https://github.com/intel/sycl-tla/pull/665))
  - Added dimension check to prevent out-of-bounds access in example 05_bmg_gemm_with_epilogue_splitk ([#529](https://github.com/intel/sycl-tla/pull/529))
 
### Known Issues
- **CuTe Column Major Support**: Column Major support for C matrix may introduce stability issues with older versions of driver. Please update to the latest driver version for optimal stability.
 
### Deprecation Notice
- Legacy APIs with old CuTe atoms are deprecated and will be removed in future releases. Users are encouraged to migrate to the new CuTe APIs for Xe architecture for better performance and support. Refer [Xe Rearchitecture](media/docs/cpp/xe_rearchitecture.md) for new APIs

## [SYCL*TLA 0.6](https://github.com/intel/sycl-tla/releases/tag/v0.6) (2025-11-03)
### Major Architecture Changes
- **Flash Attention Reimplementation ([#d02c58b](https://github.com/intel/sycl-tla/commit/d02c58b4))**: Complete rewrite of Flash Attention using new Xe atoms
  - Enhanced performance with optimized memory access patterns
  - Better integration with Intel Xe hardware capabilities
- **CUTLASS Library Generation ([#578](https://github.com/intel/sycl-tla/pull/578))**: Full support for CUTLASS library generation and operations
  - New Xe architecture support in library generation pipeline
  - Automated kernel instantiation and compilation support

### Enhancements
- **Python Operations Support ([#595](https://github.com/intel/sycl-tla/pull/595))**: Enhanced Python bindings with comprehensive test coverage
  - Improved Python API stability and usability
  - Enhanced test framework for Python operations
- **CuTe Subgroup Extensions**: New subgroup-scope operations for Intel Xe
  - Subgroup broadcast and reduction operations ([#9a6aa27](https://github.com/intel/sycl-tla/commit/9a6aa27c))
  - `make_subgroup_tensor` helpers for improved tensor manipulation ([#21fb89a](https://github.com/intel/sycl-tla/commit/21fb89a8))
- **Enhanced 2D Copy Operations**: Extended block 2D copy functionality
  - New `make_block_2d_copy_{C,D}` variants with subtiling support ([#48d82e8](https://github.com/intel/sycl-tla/commit/48d82e87))
  - Support for size-1 fragments in block 2D copies ([#2212f1b](https://github.com/intel/sycl-tla/commit/2212f1b9))
- **4-bit VNNI Reorders ([#593](https://github.com/intel/sycl-tla/pull/593))**: New 4-bit unit stride to VNNI reorder operations
- **Batch GEMM with new APIs ([#540](https://github.com/intel/sycl-tla/pull/540))**: Enhanced Batch GEMM with new streamlined APIs
- **Grouped GEMM with new APIs ([#574](https://github.com/intel/sycl-tla/pull/574))**: Enhanced grouped GEMM with new streamlined APIs

### Test Improvements
- **Python Test Coverage**: Comprehensive test suite improvements for Python operations
- **CI Infrastructure**: Enhanced continuous integration with PVC driver updates ([#575](https://github.com/intel/sycl-tla/pull/575))
- **Code Reorganization**: Renamed `python/cutlass` to `python/cutlass_cppgen` for clarity ([#587](https://github.com/intel/sycl-tla/pull/587))

### Bug Fixes
- **Epilogue Data Type Fixes**: 
  - Fixed trD compute type in Xe Epilogue ([#580](https://github.com/intel/sycl-tla/pull/580))
  - Resolved epilogue data type mismatches ([#563](https://github.com/intel/sycl-tla/pull/563))
- **CuTe Copy(new APIs) Improvements**: Multiple fixes for Xe copy operations ([#dec36a9](https://github.com/intel/sycl-tla/commit/dec36a9e))
- **Split Barrier Refactoring**: Improved split barrier functionality for better reliability ([#521dfcd](https://github.com/intel/sycl-tla/commit/521dfcd4))

### Notes and Known Issues
- Python Operations for FP8 and INT8 not generated for CUTLASS library in this release.
- Unit tests and benchmark tests are not yet migrated to newly re architected CuTe APIs.

## [SYCL*TLA 0.5](https://github.com/intel/cutlass-sycl/releases/tag/v0.5) (2025-09-26)
### Major Architecture Changes
- **Xe Rearchitecture ([#477](https://github.com/intel/cutlass-sycl/pull/477))**: Complete redesign of Xe CuTe atoms with new architecture
  - New MMA atoms for improved performance
  - Enhanced 2D copy atoms (loads, stores, prefetch with VNNI/transpose support)
  - New 2D copy helpers (low-level `make_block_2d_copy` and high-level `make_block_2d_copy_{A,B,C}`)
  - Generic and optimized reorder atoms for {int4, uint4, int8, uint8, e2m1, e4m3, e5m2} -> {half, bfloat16}
  - Requires IGC version [v2.18.5](https://github.com/intel/intel-graphics-compiler/releases/tag/v2.18.5) or later  

### New Features  
- **G++ Host Compiler Support ([#490](https://github.com/intel/cutlass-sycl/pull/490))**: Support for G++ 13 as host compiler
  - Migrated `syclcompat` to this repository as `cutlasscompat` for better compatibility
  - Fixed compilation issues when using G++ instead of clang++
  - Added new CI workflow for testing G++ host compiler builds
  - Enhanced build system to support `-DDPCPP_HOST_COMPILER=g++` option
- **Grouped GEMM for Mixed Dtype ([#457](https://github.com/intel/cutlass-sycl/pull/457))**: Extended grouped GEMM support to mixed precision operations
  - Added support for BF16 + S8 mixed dtype grouped GEMM
  - Added support for FP16 + U4 mixed dtype grouped GEMM
  - New examples: `10_bmg_grouped_gemm_bf16_f16_s8.cpp` and `10_bmg_grouped_gemm_f16_u4.cpp`

### Performance and Quality Improvements  
- **Flash Attention Accuracy Fix ([#489](https://github.com/intel/cutlass-sycl/pull/489))**: Resolved accuracy issues when seq_len % QK_BLK_N leaves a remainder
- **Improved Device-Side Random Uniform Filling ([#515](https://github.com/intel/cutlass-sycl/pull/515))**: Enhanced random number generation by reusing host implementation in SYCL
- **GPU Clock Timer Fix ([#511](https://github.com/intel/cutlass-sycl/pull/511))**: Resolved "Event is Already Being Recorded" error in loops
- **Compilation Warning Fixes ([#502](https://github.com/intel/cutlass-sycl/pull/502))**: Fixed warnings to enable -Werror compilation flag

### Code Quality and Refactoring
- **SYCLCompat Integration ([#514](https://github.com/intel/cutlass-sycl/pull/514))**: Imported `SYCLCompat` as `Compat` for better compatibility
- **CausalMask Refactoring ([#507](https://github.com/intel/cutlass-sycl/pull/507))**: Improved Flash Attention kernel code reuse and compiler optimization potential
- **SYCL Debug Trace Compatibility ([#518](https://github.com/intel/cutlass-sycl/pull/518))**: Enhanced debugging capabilities and trace compatibility
- **CuTe Tutorial Updates**: Added `tiled_copy_if` SYCL tutorial

### Testing and Development Infrastructure
- **Enhanced Unit Testing**: Added comprehensive unit tests for 16-bit x 8-bit grouped GEMM operations
- **Code Restructuring**: Refactored examples and codebase to focus on SYCL implementation

### Bug Fixes
- **Variable Name Bug Fix ([#491](https://github.com/intel/cutlass-sycl/pull/491))**: Fixed variable name bugs in CuTe architecture
- **2D Block Prefetch OOB Fix ([#488](https://github.com/intel/cutlass-sycl/pull/488))**: Fixed 2D block prefetch out-of-bounds issues in CuTe arch
- Various minor bug fixes and code improvements

### Notes and Known Issues
- CUTLASS APIs (Gemm/Collectives) are not updated with rearchitected Xe Cute atoms.

## [Cutlass 3.9.2 SYCL backend Version 0.3](https://github.com/codeplay/cutlass-fork/releases/tag/v3.9.2-0.3) (2025-06-30)
- Add support for GEMM FP8 (E5M2 and E4M3)
- Add example for GEMM FP8 with support for channel-wise and group-wise quantization
- Add support for Grouped GEMM FP8
- Improve performance for FP8 to FP16 conversion
- Add support for epilogue data conversion
- Add support for FP16 GEMM with FP16 accumulator
- Add support for BF16 GEMM with BF16 accumulator
- Add support for mixed dtype GEMM with support for tensor-wise, channel-wise and group-wise quantization
- Add example of mixed dtype BF16 + INT8 using channel-wise and group-wise quantization
- Add example of mixed dtype FP16 + INT8 using tensor-wise quantization
- Add example of mixed dtype FP16 + INT4 using channel-wise and group-wise quantization
- Add support for zero-point quantization in INT4 and INT8 data types
- Add support for Flash Attention prefill FP8 with and without KV cache
- Add support for Flash Attention decode FP8 with and without KV cache

## [Cutlass 3.9.2 SYCL backend Version 0.2](https://github.com/codeplay/cutlass-fork/releases/tag/v3.9.2-0.2) (2025-05-30)
- GEMM/StreamK/SplitK with support for FP16 data type
- Flash attention prefill with Paged KV cache with support for FP16 data type
- Performance improvements for flash attention prefill and decode

## [Cutlass 3.9 SYCL backend Version 0.1](https://github.com/codeplay/cutlass-fork/releases/tag/v3.9-0.1) (2025-04-30)
- Support for Intel GPU Data Center Max (1100 and 1550) 
- Support for Intel Arc B580 Battlemage 
- GEMM/StreamK/SplitK with support for bfloat16 data type
- Flash attention prefill and decode with KV cache with support for bfloat16 data type
- Support for epilogue operations:
  - Element-wise, row-wise and column-wise bias
  - ReLU, SiLU, GELU activation fns
  - Softmax
- Mixed precision GEMM (bfloat16/int8, half/int4) with dequantization support
- Dual GEMM & Grouped GEMM
