==================================
Carleman linearization in SageMath
==================================

.. image:: https://api.travis-ci.org/mforets/carlin.svg
   :target: https://travis-ci.org/mforets/carlin

Installation
~~~~~~~~~~~~

To install the package use the following command::

   sage -pip install --upgrade -v git+https://github.com/mforets/polyhedron_tools.git git+https://github.com/mforets/carlin.git

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

Browse the Jupyter notebooks available in the ``/examples`` folder in this repository. These can be displayed in a window embedded in github, but it is recommended to use the external nbviewer (there is a link at the top right of that window).
