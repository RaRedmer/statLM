# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setup(
    name="statLM", # Replace with your own username
    version="0.0.1",
    author="Raphael Redmer",
    author_email="ra.redmer@outlook.com",
    description="Language models to predict words",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/RaRedmer/statLM",
    license=license,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
