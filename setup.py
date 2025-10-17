#!/usr/bin/env python3
"""Setup script for social-deduction-bench package."""

from setuptools import setup, find_packages

setup(
    packages=find_packages(where=".", include=["sdb*", "scripts*"]),
    package_dir={"": "."},
)

