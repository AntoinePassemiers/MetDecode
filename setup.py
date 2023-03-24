# -*- coding: utf-8 -*-
#
#  setup.py
#
#  Copyright 2023 Antoine Passemiers <antoine.passemiers@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

from setuptools import setup


packages = [
    'metdecode',
]

setup(
    name='metdecode',
    version='1.0.0',
    description='Reference-based Non-linear Deconvolution of Whole-Genome Methylation Sequencing data',
    url='https://github.com/AntoinePassemiers/MetDecode',
    author='Antoine Passemiers',
    packages=packages,
    include_package_data=False,
    install_requires=['numpy >= 1.13.3', 'torch'])