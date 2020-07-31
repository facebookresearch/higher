![higher logo](https://github.com/facebookresearch/higher/raw/master/resources/higher_logo.png)
--------------------------------------------------------------------------------

`higher` is a library providing support for higher-order optimization, e.g. through unrolled first-order optimization loops, of "meta" aspects of these loops. It provides tools for turning existing `torch.nn.Module` instances "stateless", meaning that changes to the parameters thereof can be tracked, and gradient with regard to intermediate parameters can be taken. It also provides a suite of differentiable optimizers, to facilitate the implementation of various meta-learning approaches.

Full documentation is available at https://higher.readthedocs.io/en/latest/.

# Requirements and Installation

* Python version >= 3.5
* PyTorch version >= 1.3

To install `higher` from [PyPi](https://pypi.org/project/higher/):
```bash
pip install higher
```

To install `higher` from source:
```bash
git clone git@github.com:facebookresearch/higher.git
cd higher
pip install .
```

Alternatively `python setup.py install` will do the same thing.

# Citation

If you use `higher` in your research and found it helpful, please consider citing the following paper:

```bib
@article{grefenstette2019generalized,
  title={Generalized Inner Loop Meta-Learning},
  author={Grefenstette, Edward and Amos, Brandon and Yarats, Denis and Htut, Phu Mon and Molchanov, Artem and Meier, Franziska and Kiela, Douwe and Cho, Kyunghyun and Chintala, Soumith},
  journal={arXiv preprint arXiv:1910.01727},
  year={2019}
}
```

# Use case

## Your needs
You have a `model` with parameters `P`, where `P[t]` denotes the parameters at update timestep `t`.
You want to update the model through `k` steps of optimization, and compute gradients through the optimization process,
i.e. compute `torch.autograd.grad(P[k], P[0])` or obtain gradients that depend on this gradient pathway existing.

## Your obstacles

You are using some existing code for your `model`, so the parameters are stateful, preventing you from forming a graph with `P[t]` as nodes.
Even if you roll your own solution, you want to use optimization techniques beyond normal SGD, and `torch.optim` optimizers don't let you optimize "through" them.

## Your solution
Good news: `higher` has got you covered! Using our growing set of tools and utility functions, you can backpropagate through an unbounded number of model update steps for all your meta-learning needs.
This library includes:

* Helper functions for monkey-patching `torch.nn` modules to make them functional (non-stateful), i.e. feed their parameters as an extra argument during the forward pass.
* Classes implementing differentiable versions of `torch.optim.Adam` (and SGD), designed to track or branch out from the state of a "normal" `Adam` instance.

# Example Usage

Say your training code looks like this:

```python
model = MyModel()
opt = torch.optim.Adam(model.parameters())

for xs, ys in data:
    opt.zero_grad()
    logits = model(xs)
    loss = loss_function(logits, ys)
    loss.backward()
    opt.step()
```

To turn this into a differentiable version, the following changes should be introduced:

```python
model = MyModel()
opt = torch.optim.Adam(model.parameters())

# When you want to branch from the current state of your model and unroll
# optimization, follow this example. This context manager gets a snapshot of the
# current version of the model and optimizer at the point where you want to
# start unrolling and create a functional version `fmodel` which executes the
# forward pass of `model` with implicit fast weights which can be read by doing
# `fmodel.parameters()`, and a differentiable optimizer `diffopt` which ensures
# that at each step, gradient of `fmodel.parameters()` with regard to initial
# fast weights `fmodel.parameters(time=0)` (or any other part of the unrolled
# model history) is defined.

with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
    for xs, ys in data:
        logits = fmodel(xs)  # modified `params` can also be passed as a kwarg
        loss = loss_function(logits, ys)  # no need to call loss.backwards()
        diffopt.step(loss)  # note that `step` must take `loss` as an argument!
        # The line above gets P[t+1] from P[t] and loss[t]. `step` also returns
        # these new parameters, as an alternative to getting them from
        # `fmodel.fast_params` or `fmodel.parameters()` after calling
        # `diffopt.step`.

        # At this point, or at any point in the iteration, you can take the
        # gradient of `fmodel.parameters()` (or equivalently
        # `fmodel.fast_params`) w.r.t. `fmodel.parameters(time=0)` (equivalently
        # `fmodel.init_fast_params`). i.e. `fast_params` will always have
        # `grad_fn` as an attribute, and be part of the gradient tape.

    # At the end of your inner loop you can obtain these e.g. ...
    grad_of_grads = torch.autograd.grad(
        meta_loss_fn(fmodel.parameters()), fmodel.parameters(time=0))
```

**Beware** that when unrolling your optimisation like this for `k`, all gradients and all activations of your model at each step is kept in memory,
meaning the memory footprint of your model is `k` times greater.

# Adding your own optimizers

It is possible to use optimizers other that those found in `torch.optim`. A differentiable version must be implemented first. This can be done by subclassing `higher.optim.DifferentiableOptimizer` and overriding the `_update` method, following the arguments of the original. Assuming the logic of the optimizer being added follows the logic of those found in `torch.optim`, the steps to follow are more or less:

1. Remove the following code (no support for closures).
    ```
    loss = None
    if closure is not None:
        loss = closure()
    ```
2. Replace
    ```
    for group in self.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.data
    ```
    with
    ```
    zipped = zip(self.param_groups, grouped_grads)
    for group_idx, (group, grads) in enumerate(zipped):
        for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
          if g is None:
              continue
    ```
3. Replace `state = self.state[p]` with `state = self.state[group_idx][p_idx]`.
4. Replace any in-place op with a non in-place op, e.g. `t.add_(a, x).mul_(y)` should become `t = t.add(a, x).mul(y)` (note the assignment). Be careful to also track where dictionaries are being implicitly updated by such ops, e.g. if there is code of the form:
    ```
    p = state['k']
    ...
    p.add_(a, x)
    ```
    in the original optimizer, this code should be converted to
    ```
    p = state['k']
    ...
    state['k'] = p = p.add(a, x)
    ```
    to ensure the corresponding dictionary is.
5. Except where used for shape inference, replace instances of `t.data` with `t` for all `t`.
6. Be sure to update `group['params'][p_idx]` for each `p_idx` in need of update (those ignored will yield the original parameters in the fast weight collection). The latest fast weights will be returned by the inherited `step` function.
7. **Importantly**, you need to register your new differentiable optimizer with `higher` using `higher.register_optim` to ensure that it is recognized as an option by the library's methods. You can do this at any point after the definition of an optimizer, and before any `higher` code involving that optimizer is called. For example, if you have implemented `MyDiffOpt` as a differentiable version of some optimizer `MyOpt`, register it by adding the line `higher.register_optim(MyOpt, MyDiffOpt)` after the classes are defined.

You can find examples of how to test for gradient correctness using finite difference methods in `tests/test_optim.py`. Please note that some stability tricks may be needed to avoid `nan`s in the gradients. See the `higher.optim.DifferentiableAdam` implementation for examples of mitigation strategies, e.g. identify operations that yield exploding gradients, e.g. typically those taking the square roots of moving averages (which are intially zero), and register a backward hook using `x.register_hook` on the inputs `x` to those functions, using the helper function `_get_mask_closure` from `higher.optim`.

# Related Projects

The following papers and codebases reference or directly use `higher`:

* [Bechtle, Sarah, et al. "Meta-learning via learned loss." (2019).](https://arxiv.org/abs/1906.05374)
* [Wang, Haoxiang, Ruoyu Sun, and Bo Li. "Global convergence and induced kernels of gradient-based meta-learning with neural nets." (2020)](https://arxiv.org/abs/2006.14606)
* [Morse, Kristen, et al. "Learning State-Dependent Losses for Inverse Dynamics Learning." (2020)](https://arxiv.org/abs/2003.04947)
* [Holla, Nithin, et al. "Learning to Learn to Disambiguate: Meta-Learning for Few-Shot Word Sense Disambiguation." (2020)](https://arxiv.org/abs/2004.14355)
* [De Angeli, Nicola. "State of the Art on: Meta-learning for Few-Shot Classification." (2020)](https://pdfs.semanticscholar.org/8ec2/a4f069a1b73a5a91be6a43d6e1af028b8ca1.pdf)
* [Zhou, Allan, Tom Knowles, and Chelsea Finn. "Meta-Learning Symmetries by Reparameterization." (2020)](https://arxiv.org/abs/2007.02933)

Is yours missing? Raise an [issue](https://github.com/facebookresearch/higher/issues/new) or add it via a [pull request](https://github.com/facebookresearch/higher/compare)!

# Release Notes
See the [changelog](./CHANGELOG.md) for release notes.

# Known/Possible Issues
* See the [issues tracker](https://github.com/facebookresearch/higher/issues) for an up-to-date list.
* No support (or planned support) for `torch.nn.DataParallel` at this time. This would require a rewrite of `DataParallel`. Please raise an issue on the pytorch issue tracker if this matters to you.
* Some of the adaptative gradient-style differentiable optimizers may be unstable and yield NaNs when taking higher order gradients. Some tricks have been used to mitigate this risk. Please raise an issue if these are not sufficient in practice.
* Second-order gradients may not work with some CUDNN modules (mostly RNNs). From PyTorch v1.3 onwards, wrapping the code where models are used with `higher` using the following context manager should solve the issue:
```python
with torch.backends.cudnn.flags(enabled=False):
    # Your meta-learning code here...
```

# License
`higher` is released under Apache License Version 2.0.

# Thanks
Thanks to [Adam Paszke](https://gist.github.com/apaszke)
whose [gist](https://gist.github.com/apaszke/4c8ead6f17a781d589f6655692e7f6f0)
was the source of inspiration (and starting point) for our method for monkey
patching arbitrary `torch.nn` modules.

Thanks for the many interns, researchers, and engineers who helped road-test early versions of this library.
