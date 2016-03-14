#!/usr/bin/env python
from setuptools import setup, find_packages, Extension

ext_objective_sparse = Extension("mle_rev.objective_sparse",
                                 sources=["mle_rev/objective_sparse.pyx",],
                                 libraries=["m",])

setup(name="mle_rev",
      version = "0.0.1",
      description = "Python reversible MLE solver",
      author = "Benjamin Trendelkamp-Schroer",
      author_email = "benjamin.trendelkampschroer@gmail.com",
      packages = find_packages(),
      ext_modules=[ext_objective_sparse,],
      install_requires = ['numpy>=1.7.1', 
                          'scipy>=0.11']
      )
    
