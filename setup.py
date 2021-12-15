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

    packages=find_packages(exclude=("my_lib","myqlm-interop")),
    #install_requires=["numpy","pytest","qiskit_nature=0.20.0","qiskit_terra=0.18.2","qiskit_aer=0.9.0"],
    install_requires=["numpy","pytest","qiskit_nature","qiskit_terra","qiskit_aer"],
    # Don't change these two lines
    tests_require=["pytest"],
    cmdclass={'test': PyTest},
)
