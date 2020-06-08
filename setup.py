#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
import os

if os.path.exists('README.rst'):
    with open('README.rst') as readme_file:
        readme = readme_file.read()
else:
    readme = ''

if os.path.exists('HISTORY.rst'):
    with open('HISTORY.rst') as history_file:
        history = history_file.read()
else:
    history = ''

requirements = [s.strip() for s in open('requirements.txt', 'r').readlines()] 

setup_requirements = [ ]

test_requirements = [ ]

package_data = { '': ['*.config'] }

setup(
    author="Guillermo Blanco",
    author_email='',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Multiple-Choice Question-Answering Utilities to process and evaluate datasets.",
    install_requires=requirements,
    license="MIT license",
    long_description='',
    include_package_data=True,
    package_data=package_data,
    keywords='mcqa_utils',
    name='mcqa_utils',
    packages=find_packages(include=['mcqa_utils']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/geblanco/mcqa-utils',
    version='0.1.0',
    zip_safe=False,
)
