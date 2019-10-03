.. higher documentation master file, created by
   sphinx-quickstart on Mon Sep  9 13:47:16 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/facebookresearch/higher


higher documentation
====================

``higher`` is a library providing support for higher-order optimization, e.g.
through unrolled first-order optimization loops, of "meta" aspects of these
loops. It provides tools for turning existing `torch.nn.Module` instances
"stateless", meaning that changes to the parameters thereof can be tracked, and
gradient with regard to intermediate parameters can be taken. It also provides a
suite of differentiable optimizers, to facilitate the implementation of various
meta-learning approaches.


.. toctree::
   :maxdepth: 2
   :caption: Library Reference:

   toplevel
   patch
   optim
   utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
