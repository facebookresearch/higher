Changelog
=========

Version 0.2.1
-------------
Fixes:
- Surpress warnings about not flattening parameters when using RNNs on GPU.

Version 0.2
-----------
New:
- Patched model parameters can be modified directly, while still tracking
updates for the purpose of computing higher order gradients. This allows practices like:
```python
fmodel = monkeypatch(model)
weight = fmodel.linear.weight
new_weight = some_differentiable_function(weight)
fmodel.linear.weight = new_weight
```
- Support calling submodules of patched module directly, e.g.:
```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.submodule = nn.Linear(3,4)

    def forward(self, inputs):
        return self.submodule(inputs)

model = Model()
fmodel = higher.monkeypatch(model)
inputs = torch.rand(2,3)

models = (model, fmodel, model.submodule, fmodel.submodule)
for m1 in models:
    for m2 in models:
        assert torch.equal(m1(inputs), m2(inputs))
```
- Add property `track_higher_grads` to patched module, allowing them to behave like normal (unpatched) modules at test time. This makes their performance roughly equivalent to running the unpatched module, reducing the need to write more code for test loops.

Fixes:
- Fix monkey-patching logic for RNN variants to support PyTorch v1.4.
- Incorporate `eps` hyperparameter in differentiable Adagrad implementation.
- Release references to fast weights in `params_box[0]` after each `forward` call. This should avoid memory leaks in certain use cases.
- Fix how `fmodel.parameters()` returns iterables which avoids logic errors when running patched modules in test mode.
- Fix memory leaks/efficiency issues when running loops with `track_higher_grads=False`.

Improvements:
- Extended test coverage for RNN variants.
- Minor improvements to various unit tests.
- General codebase clean-up (removing deprecated functions, fixing typos).

Version 0.1.5
-------------
New:
- `DifferentiableOptimizer` instances take a `grad_callback` kwarg at creation and/or when `step` is called, which allows users to specify a callable which will be called on the list of gradients w.r.t. the model parameters tracked by the optimizer, so as to permit transformations of such gradients. Users are responsible for ensuring that such transformations do not change the shape/order of gradients or prevent the formation of second-order gradients.

Fixes:
- Removed a memory leak due to refcount cycle in `MonkeyPatched` classes.

Version 0.1.4
-------------
Fixes:
- Differentiable optimizers now deal with models with frozen parameters appropriately.

Version 0.1.3
-------------
Fixes:
- Can now run differentiable optimizers which implement instability mitigation with `track_higher_grad=False`.
- Various small bugfixes.

Version 0.1.2
-------------
New:
- Added `higher.create_diff_optim` utility function, which helps construct differentiable optimizers without the requirement of there being an existing optimizer instance. This is useful if differentiable optimizers to optimize something other than the unrolled model (e.g. inputs to the training loop).
- Added functionality to override hyperparameters with arbitrary tensors (which can require gradient) at each invocation of the step method of differentiable optimizers.

Version 0.1.1
-------------
Fixes:
- Greater differentiable optimizer stability.
- Updated warnings for unstable/unexpected differentiable optimizer behavior.
- Differentiable optimizer instances with issues raise warnings.

Improvements:
- Double patching modules now raises an error.

New:
- Added functionality to take gradient with regard to (appropriate) optimizer hyper parameters.

Version 0.1
-----------
Initial working release, with a few examples in /examples.
