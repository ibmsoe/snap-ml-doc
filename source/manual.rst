
.. _manual:

Manual
==================================

The Snap Machine Learning (Snap ML) Library is designed to offer fast training of generalized linear models.
It currently supports the following machine learning models:

* :ref:`linear-regression`
* :ref:`svm`
* :ref:`logistic-regression`


.. _linear-regression:

---------------------------
Linear Regression
---------------------------
.. currentmodule:: snap_ml.RidgeRegression

:class:`RidgeRegression` fits a linear model with coefficients :math:`w = (w_1, ..., w_d)` to minimize the residual sum
of squares between the predicted responses :math:`\hat{y}` and the true labels :math:`y` of the training data. 

In order to prevent the model from overfitting ridge regression imposes an 
:math:`L_2` norm penalty on the size of the coefficients. Mathematically Ridge Regression
solves the following problem:

.. math:: \underset{w}{\min\,} \frac 1 {2}|| X^\top w - y||_2^2 + \frac \lambda 2 \|w\|_2^2

where :math:`X=[x_1,...,x_n]` is the training data matrix with samples :math:`\{x_i\}_{i\in [n]}` in its columns and :math:`y_i` are the corresponding labels which take values :math:`\pm 1`. The regularization strength is controlled by the regularization parameter :math:`\lambda\geq 0`: 
The larger :math:`\lambda` the more robust the model becomes to overfitting.


.. note::
	In order to find an appropriate regularization parameter cross validation should be performed

*Snap ML* implements variants of stochastic coordinate descent `[SCD] <References_>`_ and stochastic dual coordinate ascent `[SDCA] <References_>`_ as an algorithm to fit the model parameters :math:`w`. 
In order to optimally support GPUs for training *Snap ML* also implements a parallel asynchronous version of these solvers especially designed to leverage the massive parallelism of modern GPUs `[TPASCD17] <References_>`_. 

To train the model the ``fit`` method is used; it takes the training data and the labels :math:`X,y` as input and stores the learnt coefficients of the model :math:`w` in its  ``coef_`` member function. The trained model can then be used to make predictions by calling the ``predict`` method on unlabeled data. 

.. code-block:: python

    >>> from snap_ml import LinearRegression
    >>> reg = LinearRegression(maxIter = 100, regularizer= 0.1)
    >>> reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    >>> reg.coef_
    array([ 0.5,  0.5])
    >>> reg.predict([[3, 3], [0, 1]])
    array([3, 0.5])

	
A full example of training a :class:`RidgeRegression` model on a real application can be found here :ref:`examples` and for more details about the API we refer to the :ref:`python-api-documentation`. 
 
.. _svm:

--------------------------
Support Vector Machines
--------------------------

.. currentmodule:: snap_ml.SupportVectorMachine

Support Vector Machine (SVM) is a supervised learning method which can be applied for regression as well as classification.
Currently *Snap ML* implements :class:`SupportVectorMachine` (SVMs) with a linear kernel function and offers 
:math:`L_2` regularization to prevent the model from overfitting.

Mathematically it solves the following optimization problem:

.. math:: \underset{w}{\min\,}  \sum_i [x_i^\top w  - y_i]_+ +\frac \lambda 2 \|w\|_2^2

where :math:`[u]_+=\max(u,0)` denotes the hinge loss with :math:`\{x_i\}_{i\in [n]}` being the training samples and :math:`y_i \in \{+1,-1\}` the corresponding labels. The regularization strength :math:`\lambda>0`  can be controlled by the user through the ``regularizer`` parameter.
The larger :math:`\lambda` the more robust the model becomes to overfitting.
 
*Snap ML* implements stochastic dual coordinate `[SDCA] <References_>`_ and the GPU optimized `[TPASCD17] <References_>`_ as an algorithm to train the SVM classifier. 
SDCA runs on the equivalent SVM dual problem formulation: 

.. math:: \underset{\alpha}{\min\,}  \sum_i -\alpha_i y_i +\frac 1 {2\lambda} \|X\alpha\|_2^2

with :math:`\alpha_i y_i\in[0,1]`.

To train the model the ``fit`` method is used; it takes the training data and the labels :math:`X,y` as input and stores the learnt coefficients of the model :math:`w` in its  ``coef_`` member function. The trained model can then be used to make predictions by calling the ``predict`` method on unlabeled data. 

.. code-block:: python

    >>> from snap_ml import SupportVectorMachine
    >>> reg = SupportVectorMachine(maxIter = 100, regularizer= 1.0)
    >>> reg.fit ([[1, 2], [1, 1], [2, 1]], [-1, -1, 1])
    >>> reg.coef_
    array([ 0.5,  0.5])
    >>> reg.predict([[3, 3], [0, 1]])


	
A full example of training a :class:`SupportVectorMachine` model in a real application can be found here :ref:`examples` and for more details about the API we refer to the :ref:`python-api-documentation`. 

    

.. _logistic-regression:

---------------------------
Logistic Regression
---------------------------

.. currentmodule:: snap_ml.LogisticRegression

:class:`LogisticRegression` is a linear model for classification. A logistic model is used to estimate the probability of an outcome based on the features of the input data.

In order to prevent the model from overfitting :math:`L_2` regularization is implemented. Mathematically Logistic Regression solves the following optimization problem composing of the logistic loss and an :math:`L_2` regularizer:

.. math:: \underset{w}{\min\,} \frac 1 {2}|| X^\top w - y||_2^2 + \frac \lambda 2 \|w\|_2^2

where :math:`X=[x_1,...,x_n]` is the training data matrix with samples :math:`\{x_i\}_{i\in [n]}` in its columns and :math:`y_i \in \{+1,-1\}` denote the corresponding labels. 
The regularization strength is controlled by the regularization parameter :math:`\lambda\geq 0`: the larger :math:`\lambda` the more robust the model becomes to overfitting. :math:`\lambda` can be specified by the user through the ``regularizer`` input parameter.

*Snap ML* implements stochastic coordinate descent `[SCD] <References_>`_ and stochastic dual coordinate ascent `[SDCA] <References_>`_ as an algorithm to fit the model :math:`w`. In order to support GPU acceleration *Snap ML* also implements a parallel asynchronous version of these solvers especially designed to leverage the massive parallelism of moderne GPUs `[TPASCD17] <References_>`_. 

The model can be trained using the ``fit`` method which takes the training data and the labels :math:`X,y` as input and stores the coefficients of the learnt model :math:`w` in its  ``coef_`` attribute. This model can then be used to make predictions by calling the ``predict`` method on unlabeled data. 

.. code-block:: python

    >>> from snap_ml import LogisticRegression
    >>> lr = LogisticRegression(maxIter = 100, regParam= 0.01)
    >>> lr.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

	
A full example of training a :class:`LogisticRegression` model in Snap ML can be found here :ref:`examples` and for more details about the API we refer to the :ref:`python-api-documentation`. 

=======================
References
=======================

:[SCD]: *Y. Nesterov*. Efficiency of coordinate descent methods on huge-scale optimization problems. SIAM Journal on Optimization, 2012.

:[SDCA]: *Shai Shalev-Shwartz and Tong Zhang*. Stochastic Dual Coordinate Ascent Methods for Regularized Loss Minimization. arXiv stat.ML, 2013.

:[TPASCD17]: *Thomas Parnell, Celestine Dünner, Kubilay Atasu, Manolis Sifalakis, and Haris Pozidis*. Large-Scale Stochastic Learning using GPUs. IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW), 2017.


