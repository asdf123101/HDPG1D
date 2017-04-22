#!/usr/bin/env python

from setuptools import setup, find_packages
import os

# here = os.path.abspath(os.path.dirname(__file__))


setup(name='hdpg1d',
      version='1.2',
      description='An 1D finite element solver using hybridizable discontinuous\
      Petrov-Galerkin method',
      author='Keyi Ni',
      author_email='keyi.ni@mail.utoronto.ca',
      url='test',
      license='MIT',
      packages=find_packages(),
      entry_points={
          'console_scripts': [
              'PGsolve = hdpg1d.cmd:main'
          ],
      },
      requires=[
          'numpy',
          'matplotlib',
          'scipy'
      ],)
