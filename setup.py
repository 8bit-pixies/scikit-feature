#from distutils.core import setup
from setuptools import setup

"""
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

NAME = "skfeature-chappers"

DESCRIPTION = "Unofficial Fork of Feature Selection Repository in Python (DMML Lab@ASU)"

KEYWORDS = "Feature Selection Repository"

AUTHOR = "Jundong Li, Kewei Cheng, Suhang Wang"

AUTHOR_EMAIL = "jundong.li@asu.edu, kcheng18@asu.edu, suhang.wang@asu.edu"

MAINTAINER="Chapman Siu"

MAINTAINER_EMAIL="chpmn.siu@gmail.com"

URL = "https://github.com/chappers/scikit-feature"

VERSION = "1.0.2"


setup(
    name = NAME,
    version = VERSION,
    description = DESCRIPTION,
    keywords = KEYWORDS,
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    maintainer=MAINTAINER, 
    maintainer_email=MAINTAINER_EMAIL,
    url = URL,
    packages =['skfeature', 'skfeature.utility','skfeature.function','skfeature.function.information_theoretical_based','skfeature.function.similarity_based','skfeature.function.sparse_learning_based','skfeature.function.statistical_based','skfeature.function.streaming','skfeature.function.structure','skfeature.function.wrapper',] ,
)


# pushing to pypi:
# python setup.py sdist
# python setup.py bdist_wheel --universal
# twine upload dist/*


