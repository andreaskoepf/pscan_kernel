from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


PACKAGE_NAME = "barrel_pscan"

cc_flag = []

cc_flag.append("-gencode")
cc_flag.append("arch=compute_70,code=sm_70")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")
# cc_flag.append("-gencode")
# cc_flag.append("arch=compute_90,code=sm_90")

setup(
    name=PACKAGE_NAME,
    ext_modules=[
        CUDAExtension(
            name=PACKAGE_NAME,
            sources=[
                "csrc/pscan_fwd_fp32.cu",
                "csrc/pscan.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                    "--threads",
                    "4",
                ]
                + cc_flag,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
