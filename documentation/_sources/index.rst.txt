.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Operalib's documentation!
====================================

Operalib is a Python library devoted to multiple output regression and structured output prediction. It is mainly based on
Operator-Valued Kernels (OVKs) and Output Kernels (OKs). 

OVKs allow, for instance, to model vector-valued functions. In this senario it is possible to predict silmultaneously several targets while taking into account the dependence relationship between the coordinates of the output vector. First introduced in Machine-Learning to solve Multi-Task Regression problems, OVKs have been developed in the context of vector-valued Reproducing Kernel Hilbert Spaces (vv-RKHS) to address various tasks involving vectorial outputs: multiple structured classification, vector field learning, vector autoregression, functional regression, structured output prediction, link prediction...

Our aim is to provide an easy-to-use standard implementation of operator-valued
kernel methods in the spirit of `scikit-learn <http://scikit-learn.org/>`_. Hence Operalib is designed for compatilibity to scikit-learn interface and conventions. It uses numpy, and scipy as underlying libraries.

The project is developed by:

* LTCI, Télécom ParisTech, 
* Université Paris-Saclay, 
* IBISC, University of Evry, France.


    .. toctree::
       :maxdepth: 2

       api
       auto_examples/index
       reference_papers/index
       contact
       ...

See the `README <https://github.com/RomainBrault/operalib/blob/master/README.rst>`_
for additional information.

Sources
=======

Are available on `github <https://github.com/RomainBrault/operalib/>`_.

Issues
======

Signal any issue with the library or the documentation
`here <https://github.com/RomainBrault/operalib/issues>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

