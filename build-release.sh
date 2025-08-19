script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
rm -rf ${script_dir}/build && mkdir ${script_dir}/build && cd ${script_dir}/build

. /opt/intel/oneapi/setvars.sh
. /home/taozha2/workspace/crisim/env.sh

export CC=icx
export CXX=icpx

export PrintDebugMessages=1
export NEOReadDebugKeys=1

export IGC_ShaderDumpEnable=1
export IGC_DumpToCustomDir=${script_dir}/build/mm_dumps

export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
#export IGC_VISAOptions="-perfmodel"
#export IGC_VectorAliasBBThreshold=100000000000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
export OCL_ICD_VENDORS=$HOME

output=spir64

# GPU time
cmake .. -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DSYCL_INTEL_TARGET=ON -DCUTLASS_SYCL_PROFILING_ENABLED=ON -DDPCPP_SYCL_TARGET=$output -DCUTLASS_ENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-gline-tables-only $1 $2" -DCUTLASS_SYCL_BUILTIN_ENABLE=ON

target=./examples/sycl/00_bmg_gemm/00_bmg_gemm

ninja $target && $target --m=256 --k=1024 --n=256 --iterations=1
