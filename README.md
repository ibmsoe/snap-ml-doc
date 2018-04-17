# Documentation
this repository contains a documentation of the Snap ML library. It describes its functionalities and the different APIs.


## Build the Documentation


### Install Sphinx

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

### Compile Documentation

To build the ducumentation you need to run

```
    $ sphinx-build -b html source build
```

Note that Sphinx needs to run the code in order to auto-generate the code documentation. Therefreo make sure that all git submodules are loaded and libglm is compiled.

To load the submodules use

```
    $ cd code/snap-ml-local
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






