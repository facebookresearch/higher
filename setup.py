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

from setuptools import setup

setup(
    name='higher',
    author='Edward Grefenstette',
    author_email='egrefen@fb.com',
    version='0.1.5',
    keywords='second-order, gradient descent, optimization, meta-learning',
    packages=['higher'],
    install_requires=['torch'],
    test_suite='nose.collector',
    tests_require=['nose', 'parameterized'],
)
