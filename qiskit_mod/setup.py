import os, sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    """
    A test command to run pytest on a the full repository.
    This means that any function name test_XXXX
    or any class named TestXXXX will be found and run.
    """
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main([".", "-vv"])
        sys.exit(errno)

setup(
    name="neasqcvariational",
    version="0.0.2",
    python_requires='>=3.9',
    author="Arseny Kovyrshin, Giorgio Silvi",
    license="European Union Public License 1.2",

    packages=find_packages(),
    #install_requires=["numpy","pytest","qiskit_nature=0.20.0","qiskit_terra=0.18.2","qiskit_aer=0.9.0"],
    install_requires=["numpy","pytest","qiskit_nature","qiskit_terra","qiskit_aer"],
    # Don't change these two lines
    tests_require=["pytest"],
    cmdclass={'test': PyTest},
)

# # Copyright © 2021 HQS Quantum Simulations GmbH.

# """Install questhqstools package to provide modified solutions of chemical problems with Qiskit."""
# from setuptools import find_packages, setup
# import os
# path = os.path.dirname(os.path.abspath(__file__))

# # obtain current version
# __version__ = None
# with open(os.path.join(path, 'questhqstools/__version__.py')) as f:
#     lines = f.readlines()
# __version__ = lines[-1].strip().split("'")[1].strip()

# setup(name='questhqstools',
#     version=__version__,
#     python_requires='>=3.7',
#     packages=find_packages(exclude=('docs')),
#     )

# install_requires = [
#     'numpy',
#     'pytest',
#     'qiskit_nature',
#     'qiskit',
# ]
