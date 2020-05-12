# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Installation function for the higher project."""

import pathlib
from setuptools import setup

# current directory
HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()
README = README.replace(
    "](./", 
    "](https://github.com/facebookresearch/higher/blob/master/"
)
DESC = ('A pytorch library allowing users to obtain higher order gradients '
        'over losses spanning training loops rather than individual training '
        'steps.')

REQUIRES_PYTHON = '>=3.6.0'
VERSION = (HERE / "version.txt").read_text().strip()

# run setup
setup(
    name='higher',
    version=VERSION,
    description=DESC,
    long_description=README,
    long_description_content_type="text/markdown",
    author='Edward Grefenstette',
    author_email='egrefen@fb.com',
    python_requires=REQUIRES_PYTHON,
    url="https://github.com/facebookresearch/higher",
    keywords='second-order, gradient descent, optimization, meta-learning',
    packages=['higher'],
    install_requires=['torch'],
    include_package_data=True,
    license='Apache',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='nose.collector',
    tests_require=['nose', 'parameterized'],
)
