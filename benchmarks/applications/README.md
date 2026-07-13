# Benchmarks

```
cd cutlass-fork/build/
```

## Compiling GEMM benchmarks with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_pvc | intel_gpu_bmg_g21
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja cutlass_benchmarks_gemm_sycl
./benchmarks/00_gemm/cutlass_benchmarks_gemm --config_file=../benchmarks/config_files/00_gemm/pvc/bf16.in
```

## Compiling and Running GEMM benchmarks with default configurations with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_pvc | intel_gpu_bmg_g21
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja benchmarks_gemm_sycl
```

## Compiling Flash Attention v2 benchmarks with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_pvc | intel_gpu_bmg_g21
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja cutlass_benchmarks_flash_attention
./benchmarks/02_flash_attention/flash_attention_prefill/cutlass_benchmarks_flash_attention_prefill_xe --config_file=../benchmarks/config_files/02_flash_attention/bmg/prefill/legacy/sglang_extend_nokvcache.in
./benchmarks/02_flash_attention/flash_attention_prefill_cachedKV/cutlass_benchmarks_flash_attention_prefill_cachedkv_xe --config_file=../benchmarks/config_files/02_flash_attention/bmg/prefill/legacy/sglang_extend_kvcache.in
./benchmarks/02_flash_attention/flash_attention_decode/cutlass_benchmarks_flash_attention_decode_xe --config_file=../benchmarks/config_files/02_flash_attention/bmg/decode/legacy/sglang_kvcache.in
```

## Compiling and Running Flash Attention v2 benchmarks with default configurations with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_pvc | intel_gpu_bmg_g21
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja benchmarks_flash_attention
```

## Compiling GDN Attention benchmarks with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_bmg_g21 | intel_gpu_cri
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja cutlass_benchmarks_gdn_xe
# Pick the config matching your target. Both CRI and BMG sweep the same
# sequence lengths (4096-128000); they reduce the batch count (to 1 and 4)
# from the full model to stay within memory limits and CI/simulator timeouts.
./benchmarks/03_gdn/cutlass_benchmarks_gdn_xe --config_file=../benchmarks/config_files/03_gdn/bmg/bf16.in   # intel_gpu_bmg_g21
./benchmarks/03_gdn/cutlass_benchmarks_gdn_xe --config_file=../benchmarks/config_files/03_gdn/cri/bf16.in   # intel_gpu_cri
```

## Compiling and Running GDN Attention benchmarks with default configurations with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_bmg_g21 | intel_gpu_cri
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja benchmarks_gdn
```

## Compiling and Running all benchmarks with default configurations with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_pvc | intel_gpu_bmg_g21
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja benchmarks
```