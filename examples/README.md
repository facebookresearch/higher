# Examples

This directory presents concise examples of using `higher`
to re-implement techniques that use unrolled gradients.

## Model Agnostic Meta Learning (MAML)
[./maml-omniglot.py](./maml-omniglot.py)
does few-shot Omniglot classification with MAML.
For more details see [the original MAML paper](https://arxiv.org/abs/1703.03400).
Our MAML++ fork and experiments are available [here](https://github.com/bamos/HowToTrainYourMAMLPytorch).

## Deep Energy Models
[./deep-energy-mnist.py](./deep-energy-mnist.py)
uses SPENs/ICNNS for for MNIST classification.
For more details see
[The original SPEN paper](https://arxiv.org/abs/1511.06350),
[End-to-End Learning for SPENs](https://arxiv.org/abs/1703.05667),
and
[Input Convex Neural Networks](https://arxiv.org/abs/1609.07152).
