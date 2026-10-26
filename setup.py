from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            name="nect.sampling.ct_sampling",
            sources=["nect/sampling/ct_sampling.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)