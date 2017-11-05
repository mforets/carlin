==================================
Carleman linearization in SageMath
==================================

.. image:: https://travis-ci.org/mforets/carlin.svg?branch=master
   :target: https://travis-ci.org/mforets/carlin
  
.. image:: https://img.shields.io/badge/docs-latest-blue.svg
   :target: http://mforets.github.io/carlin/doc/html

.. image:: https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000
   :target: https://github.com/mforets/carlin/blob/master/LICENSE

.. image:: https://zenodo.org/badge/84454287.svg
   :target: https://zenodo.org/badge/latestdoi/84454287
   
Installation
~~~~~~~~~~~~

This package requires sage v.7.6. or greater, see the `download page <http://www.sagemath.org/>`_ for further information about installing SageMath in your own computer, or use it online from `CoCalc <https://cocalc.com/>`_.

To install `carlin`, use the following commands::

   sage -pip install --user --upgrade -v git+https://github.com/mforets/polyhedron_tools.git 
   sage -pip install --user --upgrade -v git+https://github.com/mforets/carlin.git

Documentation
~~~~~~~~~~~~~

There is an online `HTML documentation <http://mforets.github.io/carlin/doc/html/>`_.

For a local build of the HTML documentation, clone this repository and run::

   sage -sh -c "make html"
    
The documentation in PDF format can be built with::

   sage -sh -c "make latexpdf"

These commands shall be executed inside the ``/docs`` directory.

Examples
~~~~~~~~


`Browse the Jupyter notebooks <http://nbviewer.jupyter.org/github/mforets/carlin/tree/master/examples/>`_ which are available in the ``/examples`` folder in this repository. 

These can be displayed in a window embedded in github, but it is recommended to use the 
external `nbviewer <http://nbviewer.jupyter.org/github/mforets/>`_.

