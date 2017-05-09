.. nodoctest

How to install through Sage's pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install the package use the following command::

   sage -pip install --upgrade -v git+https://github.com/mforets/carlin.git

How to build the project locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a local build of the HTML documentation::

   sage -sh -c "make html"
    
The PDF format can be built with::

   sage -sh -c "make latexpdf"

These commands shall be executed inside the ``/docs`` directory.