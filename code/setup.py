
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
import shutil

#shutil.move('cagent.pxd', 'cagent_b.pxd')

setup(
    name = "Cwrite_observations",
    include_dirs = [np.get_include()],
    ext_modules = cythonize('Cwrite_observations.pyx')
)

#shutil.move('cagent_b.pxd', 'cagent.pxd')


'''
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext = Extension("cflock", ["cagent.pyx"],
    include_dirs = [numpy.get_include()])
                
setup(ext_modules=[ext],
      cmdclass = {'build_ext': build_ext})
'''