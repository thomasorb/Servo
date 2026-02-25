from setuptools import setup, Extension, find_packages
import numpy as np

ext_modules = [
    Extension(
        "servo.faster",
        sources=["src/faster/faster.c"],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "-O3", "-ffast-math", "-fno-math-errno", "-fno-trapping-math", "-march=native"
        ],
    )
]

setup(
    name="servo",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=[],
    entry_points={
        "console_scripts": [
            "servo=servo.cli:main",
        ]
    },
)

