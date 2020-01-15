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

A ChRIS app to segment the cortical plate of fetal T2 MRI using deep learning.


Synopsis
--------

.. code::

        python3 fetal_plate_segmentation.py                                         \\
            [-h] [--help]                                               \\
            [-td] [--tempdir]                                           \\
            [-vd] [--verifydir]                                         \\
            [--json]                                                    \\
            [--man]                                                     \\
            [--meta]                                                    \\
            [--savejson <DIR>]                                          \\
            [-v <level>] [--verbosity <level>]                          \\
            [--version]                                                 \\
            <inputDir>                                                  \\
            <outputDir>

Description
-----------

`fetal_plate_segmentation.py` basically does a segment left / right fetal cortical plate, and inner region of the cortical plate.

This script part of the automatic fetal brain process pipeline at BCH.

Agruments
---------

.. code::

        [-h] [--help]
        If specified, show help message and exit.

        [-td] [--tempdir]
        If specified, intermediate result saved at tempdir.

        [-vd] [--verifydir]
        If specified, segmentation result verify image saved at verifydir.

        [--json]
        If specified, show json representation of app and exit.

        [--man]
        If specified, print (this) man page and exit.

        [--meta]
        If specified, print plugin meta data and exit.

        [--savejson <DIR>]
        If specified, save json representation file to DIR and exit.

        [-v <level>] [--verbosity <level>]
        Verbosity level for app. Not used currently.

        [--version]
        If specified, print version number and exit.


Run
----

This ``plugin`` can be run in two modes: natively as a python package or as a containerized docker image.

Using PyPI
~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~

To run using ``docker``, be sure to assign an "input" directory to ``/incoming`` and an output directory to ``/outgoing``. *Make sure that the* ``$(pwd)/out`` *directory is world writable!*

Now, prefix all calls with

.. code:: bash

    docker run --rm -v $(pwd)/out:/outgoing                             \
            jwhong1125/pl-fetal_plate_segmentation fetal_plate_segmentation.py                        \

Thus, getting inline help is:

.. code:: bash

    mkdir in out && chmod 777 out
    docker run --rm -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing      \
            jwhong1125/pl-fetal_plate_segmentation fetal_plate_segmentation.py                        \
            --man                                                       \
            /incoming /outgoing

