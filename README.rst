.. -*- mode: rst -*-

.. |Travis| image:: https://travis-ci.org/RomainBrault/operalib.svg?branch=master
.. _Travis: https://travis-ci.org/RomainBrault/operalib

.. |Coveralls| image:: https://coveralls.io/repos/github/RomainBrault/operalib/badge.svg?branch=master
.. _Coveralls: https://coveralls.io/github/RomainBrault/operalib?branch=master

.. |CircleCI| image:: https://circleci.com/gh/RomainBrault/operalib/tree/master.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/RomainBrault/operalib

.. |Python27| image:: https://img.shields.io/badge/python-2.7-blue.svg
.. _Python27: https://github.com/RomainBrault/operalib

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://github.com/RomainBrault/operalib

Operalib
========
|Travis|_ |Coveralls|_ |CircleCI|_ |Python27|_ |Python35|_

Operalib is a library for structured learning and prediction for
`python <https://www.python.org>`_ based on operator-valued kernels (OVKs).
OVKs are an extension of scalar kernels to matrix-valued kernels.
The idea is to predict silmultaneously several targets while, for instance,
encoding the output structure with the operator-valued kernel.

We aim at providing an easy-to-use standard implementation of operator-valued
kernel methods. Operalib is designed for compatilibity to
`scikit-learn <http://scikit-learn.org>`_ interface and conventions.
It uses `numpy <http://www.numpy.org>`_, and
`scipy <http://www.scipy.org>`_ as underlying libraries.

The project is developed by the
`AROBAS <https://www.ibisc.univ-evry.fr/arobas>`_ group of the
`IBISC laboratory <https://www.ibisc.univ-evry.fr/en/start>`_ of the
University of Evry, France.

Documentation
========
Is available at: http://romainbrault.github.io/operalib/documentation.

Install
=======
This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

  python setup.py install --user

To install for all users on Unix/Linux::

  python setup.py build
  sudo python setup.py install

.. For more detailed installation instructions,
.. see the web page http://scikit-learn.org/stable/install.html

GIT
~~~

You can check the latest sources with the command::

    git clone https://github.com/RomainBrault/operalib.git

or if you have write privileges::

    git clone git@github.com:RomainBrault/operalib.git

References
==========
A non-exhaustive list of publications related to operator-valued kernel is
available here:

http://romainbrault.github.io/operalib/documentation/reference_papers/index.html.
