from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension("analysis_code.md",["analysis_code/md.pyx"],\
            include_dirs=[numpy.get_include()], \
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]), \
            Extension("analysis_code.timeseries",["analysis_code/timeseries.pyx"],\
            include_dirs=[numpy.get_include()], \
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),\
        Extension("analysis_code.rdf",["analysis_code/rdf.pyx"],\
            include_dirs=[numpy.get_include()], \
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]), \
        Extension("analysis_code.Liquid_crystal.Liquid_crystal",["analysis_code/Liquid_crystal/Liquid_crystal.pyx"],\
            include_dirs=[numpy.get_include()], \
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]), \
        Extension("analysis_code.Liquid_crystal.INDUS_utils",["analysis_code/Liquid_crystal/INDUS_util.pyx"],\
            include_dirs=[numpy.get_include()], \
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),\
        Extension("analysis_code.Liquid_crystal.GAFF_LC",["analysis_code/Liquid_crystal/GAFF_LC.pyx"],\
            include_dirs=[numpy.get_include()], \
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])]

setup(name='analysis_code',ext_modules = cythonize(extensions,compiler_directives={'language_level' : "3"}))
