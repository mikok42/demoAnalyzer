"""Shim for older pip (e.g. macOS system 21.x) that cannot do editable installs from pyproject.toml alone."""
from setuptools import setup

setup()
