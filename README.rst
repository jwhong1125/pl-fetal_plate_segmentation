pl-fetal_plate_segmentation
================================

.. image:: https://badge.fury.io/py/fetal_plate_segmentation.svg
    :target: https://badge.fury.io/py/fetal_plate_segmentation

.. image:: https://travis-ci.org/FNNDSC/fetal_plate_segmentation.svg?branch=master
    :target: https://travis-ci.org/FNNDSC/fetal_plate_segmentation

.. image:: https://img.shields.io/badge/python-3.5%2B-blue.svg
    :target: https://badge.fury.io/py/pl-fetal_plate_segmentation

.. contents:: Table of Contents


Abstract
--------

An app to segment the cortical plate of fetal T2 MRI using deep leraning.


Synopsis
--------

.. code::

    python fetal_plate_segmentation.py                                           \
        [-v <level>] [--verbosity <level>]                          \
        [--version]                                                 \
        [--man]                                                     \
        [--meta]                                                    \
        <inputDir>
        <outputDir> 

Description
-----------

``fetal_plate_segmentation.py`` is a ChRIS-based application that...

Agruments
---------

.. code::

    [-v <level>] [--verbosity <level>]
    Verbosity level for app. Not used currently.

    [--version]
    If specified, print version number. 
    
    [--man]
    If specified, print (this) man page.

    [--meta]
    If specified, print plugin meta data.


Run
----

This ``plugin`` can be run in two modes: natively as a python package or as a containerized docker image.

Using PyPI
~~~~~~~~~~

To run from PyPI, simply do a 

.. code:: bash

    pip install fetal_plate_segmentation

and run with

.. code:: bash

    fetal_plate_segmentation.py --man /tmp /tmp

to get inline help. The app should also understand being called with only two positional arguments

.. code:: bash

    fetal_plate_segmentation.py /some/input/directory /destination/directory


Using ``docker run``
~~~~~~~~~~~~~~~~~~~~

To run using ``docker``, be sure to assign an "input" directory to ``/incoming`` and an output directory to ``/outgoing``. *Make sure that the* ``$(pwd)/out`` *directory is world writable!*

Now, prefix all calls with 

.. code:: bash

    docker run --rm -v $(pwd)/out:/outgoing                             \
            fnndsc/pl-fetal_plate_segmentation fetal_plate_segmentation.py                        \

Thus, getting inline help is:

.. code:: bash

    mkdir in out && chmod 777 out
    docker run --rm -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing      \
            fnndsc/pl-fetal_plate_segmentation fetal_plate_segmentation.py                        \
            --man                                                       \
            /incoming /outgoing

Examples
--------





