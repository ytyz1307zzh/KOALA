from setuptools import setup, find_packages
import sys

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='KOALA',
    version='1.0',
    author='Zhihan Zhang',
    author_email='zhangzhihan@pku.edu.cn',
    description='Knowledge-Aware Procedural Text Understanding',
    long_description=readme,
    python_requires='>=3.7',
    packages=find_packages(),
    install_requires=reqs.strip().split('\n'),
)
