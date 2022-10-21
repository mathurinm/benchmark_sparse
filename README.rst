
Benchopt for Lq least squares
=============================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver of Lq least squares:


$$\\min_{w} \\frac{1}{2} \\Vert y - Xw \\Vert^2 + \\lambda \\sum_1^p |w_j| q$$


where $q = 2/3$ for now, $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features and


$$X \\in \\mathbb{R}^{n \\times p} \\ , \\quad w \\in \\mathbb{R}^p$$


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/mathurinm/benchmark_sparse
   $ benchopt run benchmark_sparse -r 1

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_sparse -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/mathurinm/benchmark_sparse/workflows/Tests/badge.svg
   :target: https://github.com/mathurinm/benchmark_sparse/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
