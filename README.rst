.. -*- mode: rst -*-

.. |Travis| image:: https://travis-ci.org/operalib/operalib.svg?branch=master
.. _Travis: https://travis-ci.org/operalib/operalib

.. |Codecov| image:: https://codecov.io/gh/operalib/operalib/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/operalib/operalib

.. |CircleCI| image:: https://circleci.com/gh/operalib/operalib/tree/master.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/operalib/operalib

.. |Python27| image:: https://img.shields.io/badge/python-2.7-blue.svg
.. _Python27: https://github.com/operalib/operalib

.. |Python36| image:: https://img.shields.io/badge/python-3.6-blue.svg
.. _Python36: https://github.com/operalib/operalib

.. |PyPi| image:: https://badge.fury.io/py/operalib.svg
.. _PyPi: https://badge.fury.io/py/operalib

Operalib
========
|PyPi|_ |Travis|_ |Codecov|_ |CircleCI|_ |Python27|_ |Python36|_

Operalib is a library for structured learning and prediction for
`python <https://www.python.org>`_ based on operator-valued kernels (OVKs).
OVKs are an extension of scalar kernels to matrix-valued kernels.
The idea is to predict silmultaneously several targets while, for instance,
encoding the output structure with the operator-valued kernel.

We aim at providing an easy-to-use standard implementation of operator-valued
kernel methods. Operalib is designed for compatilibity to
`scikit-learn <http://scikit-learn.org>`_ interface and conventions.
It uses `numpy <http://www.numpy.org>`_,
`scipy <http://www.scipy.org>`_ and `cvxopt <http://www.cvxopt.org>`_ as
underlying libraries.

The project is developed by the
`AROBAS <https://www.ibisc.univ-evry.fr/arobas>`_ group of the
`IBISC laboratory <https://www.ibisc.univ-evry.fr/en/start>`_ of the
University of Evry, France.

Documentation
=============
Is available at: http://operalib.github.io/operalib/documentation/.

Install
=======
The package is available on PyPi, and the installation should be as simple as::

  pip install operalib

To install from the sources in your home directory, use::

  pip install .

To install for all users on Unix/Linux::

  python setup.py build
  python setup.py install

.. For more detailed installation instructions,
.. see the web page http://scikit-learn.org/stable/install.html

GIT
~~~

You can check the latest sources with the command::

    git clone https://github.com/operalib/operalib

or through ssh, instead of https, if you have write privileges::

    git clone git@github.com:operalib/operalib.git

References
==========
A non-exhaustive list of publications related to operator-valued kernel is
available here:

http://operalib.github.io/operalib/documentation/reference_papers/index.html.
