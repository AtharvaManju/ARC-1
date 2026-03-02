from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")

WITH_URING = os.environ.get("AIMEMORY_WITH_URING", "0") == "1"
WITH_GDS = os.environ.get("AIMEMORY_WITH_GDS", "0") == "1"

define_macros = []
libraries = []
extra_cxx = ["-O3", "-std=c++17", "-fPIC", "-Wall", "-Wextra"]
include_dirs = [os.path.join(CUDA_HOME, "include")]
library_dirs = [os.path.join(CUDA_HOME, "lib64")]

libraries.append("cudart")

define_macros.append(("AIMEMORY_WITH_URING", "1" if WITH_URING else "0"))
if WITH_URING:
    libraries.append("uring")

define_macros.append(("AIMEMORY_WITH_GDS", "1" if WITH_GDS else "0"))
if WITH_GDS:
    libraries.append("cufile")

setup(
    name="aimemory-engine",
    version="0.10.2",
    description="AIMemory Engine (optional): native NVMe IO + pinned allocator",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[
        CppExtension(
            name="sarc_engine",
            sources=["sarc_engine.cpp"],
            define_macros=define_macros,
            extra_compile_args={"cxx": extra_cxx},
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
