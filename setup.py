from setuptools import setup, find_packages

setup(
        name='spacelib',
        version='0.0.2',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'torch',
            'gym',
            'numba'
        ]
)
