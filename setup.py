#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [s.strip() for s in open('requirements.txt', 'r').readlines()]

setup_requirements = []

test_requirements = []

setup(
    author="Guillermo E. Blanco",
    author_email='geblanco@lsi.uned.es',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    description="Multiple Choice Evaluation utilities",
    entry_points={
        'console_scripts': [
            'mcqa_utils=mcqa_utils.mcqa_utils:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='mcqa_utils',
    name='mcqa_utils',
    packages=find_packages(include=['mcqa_utils', 'mcqa_utils.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/geblanco/mcqa_utils',
    version='0.2.2',
    zip_safe=False,
)
