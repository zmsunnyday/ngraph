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
weights and bias.

.. literalinclude:: ../../../examples/mnist_mlp.cpp
   :language: cpp
   :lines: 267-275


We repeat the process for the next layer, which we
normalize with a ``softmax``.

.. literalinclude:: ../../../examples/mnist_mlp.cpp
   :language: cpp
   :lines: 277-286


.. _loss:

Loss
----

We use cross-entropy to compute the loss. nGraph does not currenty
have a cross-entropy op, so we implement it directly, adding clipping
to prevent underflow.

.. literalinclude:: ../../../examples/mnist_mlp.cpp
   :language: cpp
   :lines: 288-299


.. _backprop:

Backprop
--------

To compute the updates, we need a computation that computes an
adjustment for the weights from an adjustment to the loss. In nGraph,
``loss`` is the computation that computes the loss; it is equivalent
to what some descriptions of the autodiff algorithm call "the tape."
If each step of a computation between a weight and the loss has a
derivative, we can use the reverse mode autodiff to compute an update
for the weight from an update for the loss; in fact, we can compute
updates for all the weights, sharing much of the update computation,
and this is what some frameworks do. But it is just as easy for us to
instead create the update computations for all of the weights, which
lets compilation optimize across steps in the computation.

We'll call the adjustment to the loss

.. code-block:: cpp

   auto delta = -learning_rate * loss;

For any node ``N``, if the update for ``loss`` is ``delta``, the
update computation for ``N`` will be given by the node

.. code-block:: cpp

   auto update = loss->backprop_node(N, delta);

The different update nodes will
share intermediate computations. So to get the updated value for ``W0`` we
just say

.. code-block:: cpp

   auto W0_next = W0 + loss->backprop_node(W0, delta);

.. _update:

Update
------

Since nGraph is stateless, we train by making a function that has the
original weights among its inputs and the updated weights among the
results. For training, we'll also need the labeled training data as
inputs, and we'll return the loss as an additional result.  We'll also
want to track how well we are doing; this is a function that returns
the loss and has the labeled testing data as input. Although we can
use the same nodes in different functions, nGraph currently does not
allow the same nodes to be compiled in different functions, so we
compile clones of the nodes.






