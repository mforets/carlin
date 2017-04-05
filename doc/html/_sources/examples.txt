.. nodoctest

Examples folder
~~~~~~~~~~~~~~~

Browse the Jupyter notebooks available in the ``/examples`` folder in this repository. 
These can be displayed in a window embedded in github, but it is recommended 
to use the external nbviewer (there is a direct link at the top right of that window).


Van der Pol oscillator
~~~~~~~~~~~~~~~~~~~~~~

A model file is text file that contains the description of an ODE, as in written math. Consider the 
``vanderpol.sage`` model, which can be found in the ``examples/`` folder.

The following script produces the file ``vanderpol_N_4.mat``. The file format MAT, originally from MATLAB software, 
is well known, and there are tools that allow to conversion to other formats, such as NumPy arrays. 

To linearize the Van der Pol model and export to a MAT file, use code like the following::

    from carlin.transformation import linearize
    
    # truncation order
    N = 4

    # initial condition, x(0) = x0
    x0 = [0.5, 0.5]

    # set model filename, it should be in the working folder of the calling script
    model_filename = 'vanderpol.sage'

    # set target filename
    target_filename = 'vanderpol_N_4.mat'
    
    linearize(model_filename, target_filename, N, x0)
    
If the computation is successful, the output is similar to::

    Obtaining the canonical representation... done
    Computing matrix BN... done
    Computing the quadratic reduction... done
    Computing the characteristics of the model... done
    Exporting to  vanderpol_N_4.mat ... done