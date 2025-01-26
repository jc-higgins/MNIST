from setuptools import setup, find_packages

setup(
    name="mnist_pkg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=['torch', 'torchvision', 'matplotlib'],
    description='Small MNIST Package',
    author='John Higgins',
    url="https://github.com/jc-higgins/MNIST"
)