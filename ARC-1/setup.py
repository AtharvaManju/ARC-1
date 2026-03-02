from setuptools import setup, find_packages

setup(
    name="arc1",
    version="0.10.2",
    description="ARC-1: autograd saved-tensor spill/restore with plug-and-play fallbacks",
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
        "engine": ["arc1-engine==0.10.2"],
        "dashboard": ["rich", "pynvml"],
    },
    entry_points={"console_scripts": ["arc1=aimemory.__main__:main", "aimemory=aimemory.__main__:main"]},
)
