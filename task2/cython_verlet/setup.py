from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

ext_modules=[
    Extension("cythonVerlet",
              ["cythonVerlet.pyx"],
              include_dirs=[get_include()]
              )
]

setup(
  name="cythonVerlet",
  cmdclass={"build_ext": build_ext},
  ext_modules=ext_modules
)

