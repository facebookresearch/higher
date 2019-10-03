# Examples

This directory presents concise examples of using `higher`
to re-implement techniques that use unrolled gradients.

## Model Agnostic Meta Learning (MAML)
[./maml-omniglot.py](./maml-omniglot.py)
does few-shot Omniglot classification with MAML.
For more details see [the original MAML paper](https://arxiv.org/abs/1703.03400).

TODO: Check the best way of noting that we are modifying and redistributing
this w.r.t. the original MIT licensing.
This code has been modified from Jackie Loong's PyTorch MAML implementation:
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py

## Structured Prediction Energy Networks (SPENs)
[./spen-mnist.py](./spen-mnist.py)
uses SPENs for for MNIST classification.
For more details see
[The original SPEN paper](https://arxiv.org/abs/1511.06350)
and
[End-to-End Learning for SPENs](https://arxiv.org/abs/1703.05667).
