#!/usr/bin/env python

from distutils.core import setup
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))


# def read(*parts):
#     # intentionally *not* adding an encoding option to open
#     return codecs.open(os.path.join(here, *parts), 'r').read()

# long_description = read('README.rst')

setup(name='hdpg1d',
      version='1.0',
      description='An 1D finite element solver using hybridizable discontinuous\
      Petrov-Galerkin method',
      author='Keyi Ni',
      author_email='keyi.ni@mail.utoronto.ca',
      url='test',
      license='MIT',
      packages=['hdpg1d'],
      requires=[
          'numpy',
          'matplotlib',
          'scipy'
      ],)
