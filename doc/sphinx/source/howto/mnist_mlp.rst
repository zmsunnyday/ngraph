.. mnist_mlp.rst

################
Training a Model
################

In this section, we construct an MNIST-MLP model for inference and
then extend it for training. We also include a simple data loader for
the MNIST data so that training can be tested.

* :ref:`model_overview`
* :ref:`code_structure`
  - :ref:`inference`
  - :ref:`loss`
  - :ref:`backprop`
  - :ref:`update`

.. _model_overview:

Model Overview
==============

The ngraph API was designed for automatic graph construction under
direction of a framework. Without a framework, the process is somewhat
tedious, so we have selected a relatively simple model, a fully
connected topology with one hidden layer followed by softmax.

Since the graph is stateless, there are parameters for the input and
the variables. The training function will return the tensors for the
updated variables.

.. _code_structure:

Code Structure
==============


.. _inference:

Inference
---------

We begin by building the graph, starting with the input parameter
``X``. We define a fully-connected layer, including a parameter for
weights and bias. We repeat the process for the next layer, which we
cap with a ``softmax``.

.. _loss:

Loss
----

We use cross-entropy to compute the loss.

.. _update:

Update
------





