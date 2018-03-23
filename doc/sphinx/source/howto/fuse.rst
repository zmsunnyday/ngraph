.. fuse.rst  


#########
Fuse ops
#########

* :ref:`pattern_matching`
* :ref:`graph_rewrite`

In working with a :term:`function graph`, there are many ways to describe what 
happens when we need to do something with the ops (AKA "operational functions") 
from that graph. *Fusing* is the term we will use here, but it's also described 
as: *combining*, *folding*, *collapsing*, or *merging* graph functions. One 
common use case for nGraph is to fuse a subgraph from the function graph into 
:doc:`one of the core ops <../ops/index>`. In other words, nGraph will take a 
sub-set of computations from the graph and make it more efficient via 
:term:`fusion`.

nGraph is an *optimizing* compiler. As such, it performs a series of 
optimization passes over a given function graph to transform it into a 
different graph that is not only *semantically equivalent* to the original, 
but also *inherently optimized*. The new graph has superior runtime 
characteristics for any given backend.  

The optimization passes may include algebraic simplifications, domain-specific 
simplifications (ReshapeElimination TODO: add more), and fusion. Most passes 
share the same mode of operation (or the same operational structure) and consist 
of two stages:

#. Locating a list of potential transformation candidates (usually, subgraphs) 
   in the given graph.
#. Transforming the selected candidates into semantically-equivalent subgraphs 
   that (usually) run faster.

Let's consider an example. A user would like to execute a simple graph that 
describes the following arithmetic expression:

:math:`a + b * 1` or :math:`Add(a, Mul(b, 1))` 

In the above expressions, `1` is an :term:`identity element`: any element 
multiplied by the identity element is equal to itself. This is the same as saying

:math:`b * 1 = b` 

The writer of an algebraic-simplification pass would probably want to ``locate`` 
all multiplication expressions where multiplicands are multiplied by `1` (for 
stage 1) and ``transform``, `` simplify``, or ``replace`` those expressions with 
just their multiplicands (for stage 2).

To make the work of an optimization pass writer easier, the nGraph library 
includes facilities that enable the *finding* of relevant candidates using pattern 
matching (via `pattern/matcher.hpp`), and the *transforming* of the original graph 
into a condensed version ( via `pass/graph_rewrite.hpp`).

Let's consider each of the two in more detail and many ways they can help the 
work of the optimization pass writer.


.. _pattern_matching: 

Apply pattern matching to fuse ops
------------------------------------

Before delving into the details of pattern matching, it's worthwhile to point 
out that the sole purpose of pattern matching is to **find** patterns in the 
given graph. What is a :term:`pattern`? In the context of ngraph, the *pattern* 
is simply a subgraph that could contain any operation ngraph's IR defines 
(addition, subtraction, etc) *along with* some special wildcard nodes which will 
be discussed in just a moment.

A good analogy would be regular expressions. One writes a regex (pattern) and 
runs it through some text (our input graph) to find and/or replace the 
occurences of the pattern in the given text. Similarly, the pass writer 
constructs patterns which are just regular ngraph graphs and then runs those 
patterns through given graphs.


.. doxygenclass:: ngraph::pattern::Matcher
   :project: ngraph
   :members:




.. TODO





.. _graph_rewrite:

Using GraphRewrite to fuse ops
--------------------------------

.. TODO 