#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name="mle_rev",
      version = "0.0.1",
      description = "Python reversible MLE solver",
      author = "Benjamin Trendelkamp-Schroer",
      author_email = "Benjamin.Trendelkamp-Schroer@fu-berlin.de",
      packages = find_packages(),
      install_requires = ['numpy>=1.7.1', 
                          'scipy>=0.11']
      )
    
