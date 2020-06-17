# cython: language_level=3
# distutils: Purpose to distributre the python models

# write a setup script (setup.py by convention)

# (optional) write a setup configuration file

# create a source distribution

# (optional) create one or more built (binary) distributions


from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
# import cython_utils

import os
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

# All of this is done through another keyword argument to setup(), the ext_modules option. ext_modules is just a list of Extension instances, each of which describes a single extension module. 
# Cython.Build.cythonize(module_list, exclude=None, nthreads=0, aliases=None, quiet=False, force=False, language=None, exclude_failures=False, **options)¶
# Compile a set of source modules into C/C++ files and return a list of distutils Extension objects for them.

# module_list – As module list, pass either a glob pattern, a list of glob patterns or a list of Extension objects. The latter allows you to configure the extensions separately through the normal distutils options. 

setup(ext_modules = cythonize(["graphsaint_cython/cython_sampler.pyx","graphsaint_cython/cython_utils.pyx","graphsaint_cython/norm_aggr.pyx"]), include_dirs = [numpy.get_include()])
# numpy.get_include():  Make distutils look for numpy header files in the correct place


# to compile: python graphsaint/setup.py build_ext --inplace
# The --inplace option creates the shared object file (with .so suffix) in the current directory.

"""
Module files all in .pyx format: Pyrex langauage

Source code file written in Pyrex, a Python-like language used 
for writing Python extension modules with C-like syntax; 
may contain references to existing C modules; compiles code
 that increases the execution time of Python programs.

Since Pyrex is used for writing modules for Python software, PYX files may sometimes be found with Python applications.

NOTE: Cython is an extension of Pyrex that contains several enhancements, and it is often preferred to Pyrex.


"""