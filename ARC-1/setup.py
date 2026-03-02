from setuptools import setup, find_packages

setup(
    name="aimemory",
    version="0.10.2",
    description="AIMemory: autograd saved-tensor spill/restore with plug-and-play fallbacks",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "numpy",
        "psutil",
        "cryptography",
    ],
    extras_require={
        "engine": ["aimemory-engine==0.10.2"],
        "dashboard": ["rich", "pynvml"],
    },
    entry_points={"console_scripts": ["aimemory=aimemory.__main__:main"]},
)
