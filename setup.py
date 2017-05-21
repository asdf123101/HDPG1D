#!/usr/bin/env python

from setuptools import setup, find_packages


setup(name='hdpg1d',
      version='3.0.1',
      description='An 1D finite element solver using hybridizable discontinuous\
      Petrov-Galerkin method',
      author='Keyi Ni',
      author_email='keyi.ni@mail.utoronto.ca',
      url='https://github.com/asdf123101/HDPG1D',
      license='MIT',
      packages=find_packages(),
      data_files=[('config', ['hdpg1d/config/config.json'])],
      include_package_data=True,
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
