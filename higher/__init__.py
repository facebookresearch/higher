# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import patch  # noqa: F401
from . import optim  # noqa: F401
from . import utils  # noqa: F401

from .optim import get_diff_optim, create_diff_optim  # noqa: F401
from .optim import register_optim  # noqa: F401
from .patch import monkeypatch

from contextlib import contextmanager as _contextmanager
import typing as _typing

import torch as _torch


@_contextmanager
def innerloop_ctx(
    model: _torch.nn.Module,
    opt: _torch.optim.Optimizer,
    device: _typing.Optional[_torch.device] = None,
    copy_initial_weights: bool = True,
    override: optim._OverrideType = None,
    track_higher_grads: bool = True
):
    r"""A context manager for writing differentiable inner loops.

    Args:
        model: a ``torch.nn.Module`` subclass instance.
        opt: an existing optimizer, assumed to be an instance of
            ``torch.optim.Optimizer``, of a supported type which is either 
            defined in ``torch.optim``, or a custom implemantation which has 
            been added to higher at runtime by using ``higher.register_optim``. 
            We assume this optimizer tracks the parameters (or some subset 
            thereof) of a single ``torch.nn.Module`` instance, with support for 
            parameter groups.
        device (optional): a device to cast the fast weights and state to. If
            not specified, the device used for corresponding weights of 
            ``model`` will be used.
        copy_initial_weights: if true, the weights of the patched module are
            copied to form the initial weights of the patched module, and thus
            are not part of the gradient tape when unrolling the patched module.
            If this is set to False, the actual module weights will be the
            initial weights of the patched module. This is useful when doing
            MAML, for example.
        override (optional): a dictionary mapping optimizer settings (i.e. those
            which would be passed to the optimizer constructor or provided
            within parameter groups) to either singleton lists of override
            values, or to a list of override values of length equal to the
            number of parameter groups. If a single override is provided for a
            keyword, it is used for all parameter groups. If a list is provided,
            the ``i``\ th element of the list overrides the corresponding 
            setting in the ``i``\ th parameter group. This permits the passing 
            of tensors requiring gradient to differentiable optimizers for use 
            as optimizer settings.
        track_higher_grads: if True, during unrolled optimization the graph be
            retained, and the fast weights will bear grad funcs, so as to permit
            backpropagation through the optimization process. Setting this to
            False allows ``innerloop_ctx`` to be used in "test mode", without
            potentially tracking higher order gradients. This can be useful when
            running the training loop at test time, e.g. in k-shot learning
            experiments, without incurring a significant memory overhead.

    Yields:
        A ``(fmodule, diffopt)`` tuple. where ``fmodule`` is a "stateless" 
        version of the original module, for which calls to forward take the 
        additional kwarg-only parameter ``params``, which should be a list of 
        torch tensors requiring gradients, ideally provided by this function 
        (see below) or by an update step from one of the optimizers in 
        ``higher.optim``. And ``diffopt`` is an initialized 
        ``DifferentiableOptimizer`` instance of the right subtype.
    """
    fmodel = monkeypatch(
        model, 
        device, 
        copy_initial_weights=copy_initial_weights,
        track_higher_grads=track_higher_grads
    )
    diffopt = optim.get_diff_optim(
        opt,
        model.parameters(),
        fmodel=fmodel,
        device=device,
        override=override,
        track_higher_grads=track_higher_grads
    )
    yield fmodel, diffopt


__all__: list = ["innerloop_ctx"]
