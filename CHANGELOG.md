Changelog
=========

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