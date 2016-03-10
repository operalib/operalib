.. -*- mode: rst -*-

|Travis|_ |Python27|_ |Python35|_

.. |Travis| image:: https://travis-ci.org/RomainBrault/operalib.svg?branch=master
.. _Travis: https://travis-ci.org/scikit-learn/scikit-learn

.. |Python27| image:: https://img.shields.io/badge/python-2.7-blue.svg
.. _Python27: https://badge.fury.io/py/scikit-learn

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/scikit-learn

operalib
========
Operalib is a structured learning and prediction library for
[Python](https://www.python.org/) utilising operator-valued kernels (OVKs).
OVKs are an extension of scalar kernels into matrix-valued kernels,
allowing prediction of several targets simultaneously while, for instance,
encoding the output structure with the operator-valued kernel.

We aim at providing an easy-to-use standard implementation of operator-valued
kernel methods. Operalib is designed for compatilibity to
[Scikit-learn](http://scikit-learn.org/) interface and conventions.
It utilises [Numpy](http://www.numpy.org/), and
[Scipy](http://www.scipy.org/) as underlying libraries.

For structured learning using max-margin methods,
conditional random fields or structured SVM's check the excellent
[PyStruct](https://pystruct.github.io) library.

The project is developed by the
[AROBAS](https://www.ibisc.univ-evry.fr/arobas) group of the
[IBISC laboratory](https://www.ibisc.univ-evry.fr/en/start) of the
University of Evry, France.

References
========
* Néhémy Lim, Florence d'Alché-Buc, Cédric Auliac, George Michailidis (2014): Operator-valued Kernel-based Vector Autoregressive Models for Network Inference, (in revision)
* Lim, Senbabaoglu, Michalidis and d'Alche-Buc (2013): OKVAR-Boost: a novel boosting algorithm to infer nonlinear dynamics and interactions in gene regulatory networks. Bioinformatics 29 (11):1416-1423.
* Brouard, d'Alché-Buc and Szafranski (2011): Semi-Supervized Penalized Output Kernel Regression for Link Prediction. In ICML 2011.


Further litterature on Operator-valued kernels
========
- On operator-valued kernels: http://www0.cs.ucl.ac.uk/staff/M.Pontil/reading/vecval.pdf