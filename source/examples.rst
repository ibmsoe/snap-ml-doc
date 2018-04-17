
.. _examples:

==================================
Examples
==================================

Here you can find some application examples of Snap ML.

.. _adclickExample:

Ad click Prediction
----------------------------------

:note: add example on kaggle dataset

.. code-block:: python

    >>> from snap_ml import LinearRegression
    >>> from sklearn import 
	
    >>> data = DatasetReader(session)
    >>> reg = LinearRegression(maxIter = 100, regParam= 0.01)
    >>> reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    >>> reg.model
    array([ 0.5,  0.5])


