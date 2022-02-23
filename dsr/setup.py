from distutils.core import setup
from setuptools import find_packages
from Cython.Build import cythonize
import numpy
import os

# To build cython code using setup try:
# python setup.py build_ext --inplace

setup(  name='dsr',
        version='1.0dev',
        description='Deep symbolic regression.',
        author='LLNL',
        # packages=['dsr'],
        packages=find_packages(),
        ext_modules=cythonize([os.path.join('dsr','cyfunc.pyx')]), 
        include_dirs=[numpy.get_include()]
        )
