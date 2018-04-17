# Documentation
this repository contains a documentation of the Snap ML library. It describes its functionalities and the different APIs.

## Install Sphinx

The documentation is built using the Sphinx Python Documentation Generator. 
To install Sphinx use the following command

```
    $ sodu pip install sphinx
```

The following built-in Sphinx extensions are required and should be included in the standard installation:

*  autodoc
*  imgmath
*  napoleon
*  intersphinx

## Build Documentation

To build the ducumentation you need to run

```
    $ mkdir build
    $ sphinx-build -b html source build
```

To see the display the generated documentation point your browser to *build/index.html*.

**Note** that Sphinx needs to run the code in order to auto-generate the code documentation. Therefore make sure that all git submodules are loaded and libglm is compiled.

To load the submodules use

```
    $ cd code/snap-ml-local
    $ git submodule init
    $ git submodule update --recursive
    $ cd ..
    $ cd code/snap-ml-spark
    $ git submodule init
    $ git submodule update --recursive

```

To compile libglm

```
    $ cd code/snap-ml-local/snap_ml/wrapper/
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make

```






