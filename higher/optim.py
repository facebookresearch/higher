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

'''Differentiable optimizer wrappers around ``torch.optim`` instances.'''

import abc as _abc
import collections as _collections
import copy as _copy
import math as _math
import typing as _typing
import warnings as _warnings

import torch as _torch

from . import patch as _patch
from . import utils as _utils

_GroupedGradsType = _typing.List[_typing.List[_torch.Tensor]]
_StateType = _typing.List[_typing.DefaultDict[int, _typing.Any]]
_GradClosureType = _typing.Callable[[_torch.Tensor], _torch.Tensor]
_OverrideType = _typing.Dict[str, _typing.List[_typing.Any]]
_GradCallbackType = _typing.Callable[[_typing.List[_torch.Tensor]],
                                     _typing.List[_torch.Tensor]]


def _get_mask_closure(mask: _torch.Tensor) -> _GradClosureType:
    def closure(grad: _torch.Tensor) -> _torch.Tensor:
        grad = _torch.where(mask, _torch.zeros_like(grad), grad)
        if grad.requires_grad:
            grad.register_hook(_get_mask_closure(mask))
        return grad
    return closure


def _maybe_mask(tensor: _torch.Tensor, mask: _torch.Tensor) -> None:
    if tensor.requires_grad:
        tensor.register_hook(_get_mask_closure(mask))


class DifferentiableOptimizer(_abc.ABC):
    def __init__(
        self,
        other: _torch.optim.Optimizer,
        reference_params: _typing.Iterable[_torch.Tensor],
        fmodel: _typing.Optional[_patch._MonkeyPatchBase] = None,
        device: _typing.Optional[_torch.device] = None,
        override: _typing.Optional[_OverrideType] = None,
        grad_callback: _typing.Optional[_GradCallbackType] = None,
        track_higher_grads: bool = True,
        **kwargs
    ) -> None:
        r"""Initialize the optimizer with the state of an existing optimizer.

        Args:
            other: an existing optimizer instance.
            reference_params: an iterable over the parameters of the original
                model.
            fmodel (optional): a patched stateless module with a view on
                weights.
            device (optional): the device to cast state tensors to.
            override (optional): a dictionary mapping optimizer settings (i.e.
                those which would be passed to the optimizer constructor or
                provided within parameter groups) to either singleton lists of
                override values, or to a list of override values of length equal
                to the number of parameter groups. If a single override is
                provided for a keyword, it is used for all parameter groups. If
                a list is provided, the ``i``\ th element of the list overrides the
                corresponding setting in the ``i``\ th parameter group. This permits
                the passing of tensors requiring gradient to differentiable
                optimizers for use as optimizer settings.
            grad_callback: (optional) a single argument function which will be
                applied to a list of gradients of parameters, which respects the
                order specified by ``reference_params``. This can be used to
                apply a function, such as gradient clipping, to all (or a
                subset) of these gradients every time the step function is
                called. If this keyword argument is provided when calling the
                step method, its value will override the default specified here.
            track_higher_grads: if True, during unrolled optimization the graph
                be retained, and the fast weights will bear grad funcs, so as to
                permit backpropagation through the optimization process. Setting
                this to False allows the differentiable optimizer to be used in
                "test mode", without potentially tracking higher order
                gradients. This can be useful when running the training loop at
                test time, e.g. in k-shot learning experiments, without
                incurring a significant memory overhead.
        """
        reference_params = list(reference_params)

        # Copy param groups and set up structures for copy state
        self.param_groups = _copy.deepcopy(other.param_groups)
        self._group_to_param_list: _typing.List[_typing.List[int]] = []
        self.state: _StateType = [
            _collections.defaultdict(dict)
            for _ in range(len(self.param_groups))
        ]

        # Deal with override
        if override is not None:
            self._apply_override(override)

        self._grad_callback = grad_callback

        # Copy and cast state
        zipped = zip(self.param_groups, other.param_groups)
        for group_idx, (group, orig_group) in enumerate(zipped):
            local_list = []
            for p_idx, p in enumerate(orig_group['params']):
                if p in other.state:
                    self.state[group_idx][p_idx] = {
                        k: _utils._recursive_copy_and_cast(v, device)
                        for k, v in other.state[p].items()
                    }
                index = _utils._find_param_in_list(p, reference_params)
                if index is None:
                    raise ValueError(
                        "Could not find parameter {} in reference parameters.".
                        format(str(p))
                    )
                local_list.append(index)
            group['params'] = [None] * len(group['params'])
            self._group_to_param_list.append(local_list)

        self._fmodel = fmodel
        self._track_higher_grads = track_higher_grads

    def _apply_override(self, override: _OverrideType) -> None:
        for k, v in override.items():
            # Sanity check
            if (len(v) != 1) and (len(v) != len(self.param_groups)):
                raise ValueError(
                    "Mismatch between the number of override tensors for "
                    "optimizer parameter {} and the number of "
                    "parameter groups.".format(k)
                )
            for group_idx, group in enumerate(self.param_groups):
                group[k] = v[0] if len(v) == 1 else v[group_idx]

    def step(
        self,
        loss: _torch.Tensor,
        params: _typing.Iterable[_torch.Tensor] = None,
        override: _typing.Optional[_OverrideType] = None,
        grad_callback: _typing.Optional[_GradCallbackType] = None,
        **kwargs
    ) -> _typing.Iterable[_torch.Tensor]:
        r"""Perform a model update.

        This would be used by replacing the normal sequence::

            opt.zero_grad()
            loss.backward()
            opt.step()

        with::

            diffopt.step(loss)


        Args:
            loss: the loss tensor.
            params (optional): the parameters with regard to which we measure
                the loss. These must be provided if the differentiable optimizer
                did not receive a patched model with a view over its own fast
                weights at initialisation. If there is such a model, and params
                are provided, they will overwrite the params of the encapsulated
                model.
            override (optional): a dictionary mapping optimizer settings (i.e.
                those which would be passed to the optimizer constructor or
                provided within parameter groups) to either singleton lists of
                override values, or to a list of override values of length equal
                to the number of parameter groups. If a single override is
                provided for a keyword, it is used for all parameter groups. If
                a list is provided, the ``i``\ th element of the list overrides
                the corresponding setting in the ``i``\ th parameter group. This
                permits the passing of tensors requiring gradient to
                differentiable optimizers for use as optimizer settings. Setting
                override here has highest precedence, i.e. it will override any
                tensors provided as override during the creation of the
                differentiable optimizer, where there is name clash.
            grad_callback: (optional) a single argument function which will be
                applied to a list of gradients of parameters, which respects the
                order specified by ``reference_params``. This can be used to
                apply a function, such as gradient clipping, to all (or a
                subset) of these gradients every time the step function is
                called. This callback overrides the default provided when
                constructing the differentiable optimizer.


        Returns:
            The updated parameters, which will individually have ``grad_fn``\ s
            of their own. If the optimizer has an encapsulated patched model,
            its view over its own fast weights will be updated with these
            params.
        """

        # Deal with override
        if override is not None:
            self._apply_override(override)

        if self._fmodel is None or self._fmodel.fast_params is None:
            if params is None:
                raise ValueError(
                    "params kwarg must be passed to step if the differentiable "
                    "optimizer doesn't have a view on a patched model with "
                    "params."
                )
        else:
            params = self._fmodel.fast_params if params is None else params

        params = list(params)

        # This allows us to gracefully deal with cases where params are frozen.
        grad_targets = [
            p if p.requires_grad else _torch.tensor([], requires_grad=True)
            for p in params
        ]

        all_grads = _torch.autograd.grad(
            loss,
            grad_targets,
            create_graph=self._track_higher_grads,
            allow_unused=True  # boo
        )

        if grad_callback is not None:
            all_grads = grad_callback(all_grads)
        elif self._grad_callback is not None:
            all_grads = self._grad_callback(all_grads)

        grouped_grads = []
        for group, mapping in zip(self.param_groups, self._group_to_param_list):
            grads = []
            for i, index in enumerate(mapping):
                group['params'][i] = params[index]
                grads.append(all_grads[index])
            grouped_grads.append(grads)

        self._update(grouped_grads)

        new_params = params[:]
        for group, mapping in zip(self.param_groups, self._group_to_param_list):
            for p, index in zip(group['params'], mapping):
                if self._track_higher_grads:
                    new_params[index] = p
                else:
                    new_params[index] = p.detach().requires_grad_()

        if self._fmodel is not None:
            self._fmodel.update_params(new_params)

        return new_params

    @_abc.abstractmethod
    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:
        pass


class DifferentiableSGD(DifferentiableOptimizer):
    r"""A differentiable version of the SGD optimizer.

    This optimizer creates a gradient tape as it updates parameters."""

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:
        zipped = zip(self.param_groups, grouped_grads)
        for group_idx, (group, grads) in enumerate(zipped):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue

                if weight_decay != 0:
                    g = _add(g, weight_decay, p)
                if momentum != 0:
                    param_state = self.state[group_idx][p_idx]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = g
                    else:
                        buf = param_state['momentum_buffer']
                        buf = _add(buf.mul(momentum), 1 - dampening, g)
                        param_state['momentum_buffer'] = buf
                    if nesterov:
                        g = _add(g, momentum, buf)
                    else:
                        g = buf

                group['params'][p_idx] = _add(p, -group['lr'], g)


class DifferentiableAdam(DifferentiableOptimizer):
    r"""A differentiable version of the Adam optimizer.

    This optimizer creates a gradient tape as it updates parameters."""

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:

        zipped = zip(self.param_groups, grouped_grads)
        for group_idx, (group, grads) in enumerate(zipped):
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']

            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):

                if g is None:
                    continue

                state = self.state[group_idx][p_idx]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = _torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = _torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. mov. avg. of sq. grad. vals
                        state['max_exp_avg_sq'] = _torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']

                if weight_decay != 0:
                    g = g + (weight_decay * p)

                # Decay the first and second moment running average coefficient
                state['exp_avg'] = exp_avg = (exp_avg * beta1) + (1 - beta1) * g
                state['exp_avg_sq'] = exp_avg_sq = (
                    (exp_avg_sq * beta2) + (1 - beta2) * g * g
                )

                # Deal with stability issues
                mask = exp_avg_sq == 0.
                _maybe_mask(exp_avg_sq, mask)

                if amsgrad:
                    # Maintains the max of all 2nd moment running avg. till now
                    state['max_exp_avg_sq'] = max_exp_avg_sq = _torch.max(
                        max_exp_avg_sq, exp_avg_sq
                    )
                    # Use the max. for normalizing running avg. of gradient
                    denom = _add(
                        max_exp_avg_sq.sqrt() / _math.sqrt(bias_correction2),
                        group['eps']
                    )
                else:
                    denom = _add(
                        exp_avg_sq.sqrt() / _math.sqrt(bias_correction2),
                        group['eps']
                    )

                step_size = group['lr'] / bias_correction1

                group['params'][p_idx] = _addcdiv(
                    p, -step_size, exp_avg, denom
                )


class DifferentiableAdamW(DifferentiableOptimizer):
    r"""A differentiable version of the AdamW optimizer.

        This optimizer creates a gradient tape as it updates parameters."""

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:

        zipped = zip(self.param_groups, grouped_grads)
        for group_idx, (group, grads) in enumerate(zipped):
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):

                if g is None:
                    continue

                # Perform stepweight decay
                p = p * (1 - group['lr'] * group['weight_decay'])

                if g.is_sparse:
                    raise RuntimeError(
                        'AdamW does not support sparse gradients')

                state = self.state[group_idx][p_idx]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = _torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = _torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. mov. avg. of sq. grad. vals
                        state['max_exp_avg_sq'] = _torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']

                # Decay the first and second moment running average coefficient
                state['exp_avg'] = exp_avg = (exp_avg * beta1) + (1 - beta1) * g
                state['exp_avg_sq'] = exp_avg_sq = (
                    (exp_avg_sq * beta2) + (1 - beta2) * g * g
                )

                # Deal with stability issues
                mask = exp_avg_sq == 0.
                _maybe_mask(exp_avg_sq, mask)

                if amsgrad:
                    # Maintains the max of all 2nd moment running avg. till now
                    state['max_exp_avg_sq'] = max_exp_avg_sq = _torch.max(
                        max_exp_avg_sq, exp_avg_sq
                    )
                    # Use the max. for normalizing running avg. of gradient
                    denom = _add(
                        max_exp_avg_sq.sqrt() / _math.sqrt(bias_correction2),
                        group['eps']
                    )
                else:
                    denom = _add(
                        exp_avg_sq.sqrt() / _math.sqrt(bias_correction2),
                        group['eps']
                    )

                step_size = group['lr'] / bias_correction1

                group['params'][p_idx] = _addcdiv(
                    p, -step_size, exp_avg, denom
                )


class DifferentiableAdadelta(DifferentiableOptimizer):
    r"""A differentiable version of the Adadelta optimizer.

    This optimizer creates a gradient tape as it updates parameters."""

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:

        zipped = zip(self.param_groups, grouped_grads)
        for group_idx, (group, grads) in enumerate(zipped):
            rho, eps = group['rho'], group['eps']

            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue

                if g.data.is_sparse:
                    raise RuntimeError(
                        'Adadelta does not support sparse gradients'
                    )
                state = self.state[group_idx][p_idx]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = _torch.zeros_like(p.data)
                    state['acc_delta'] = _torch.zeros_like(p.data)

                square_avg, acc_delta = state['square_avg'], state['acc_delta']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    g = _add(g, group['weight_decay'], p)

                square_avg = _addcmul(square_avg.mul(rho), 1 - rho, g, g)
                state['square_avg'] = square_avg
                std = _add(square_avg, eps).sqrt()
                delta = _add(acc_delta, eps).sqrt().div(std).mul(g)
                state['acc_delta'] = _addcmul(
                    acc_delta.mul(rho), 1 - rho, delta, delta
                )
                group['params'][p_idx] = _add(p, -group['lr'], delta)


class DifferentiableAdagrad(DifferentiableOptimizer):
    r"""A differentiable version of the Adagrad optimizer.

    This optimizer creates a gradient tape as it updates parameters."""

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:

        zipped = zip(self.param_groups, grouped_grads)
        for group_idx, (group, grads) in enumerate(zipped):
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue

                state = self.state[group_idx][p_idx]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if g.data.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not compatible with sparse "
                            "gradients"
                        )
                    g = _add(g, group['weight_decay'], p)

                clr = group['lr'] / (
                    1 + (state['step'] - 1) * group['lr_decay']
                )

                if g.is_sparse:
                    # TODO: implement support for sparse gradients.
                    raise NotImplementedError(
                        "sparse gradient support for DifferentiableAdagrad not "
                        "implemented yet."
                    )
                else:
                    state['sum'] = sum_ = _addcmul(state['sum'], 1, g, g)
                    mask = sum_ == 0.
                    _maybe_mask(sum_, mask)
                    std = _add(state['sum'].sqrt(), group['eps'] if 'eps' in group else 1e-10)
                    group['params'][p_idx] = _addcdiv(p, -clr, g, std)


class DifferentiableAdamax(DifferentiableOptimizer):
    r"""A differentiable version of the Adamax optimizer.

    This optimizer creates a gradient tape as it updates parameters."""

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:

        zipped = zip(self.param_groups, grouped_grads)
        for group_idx, (group, grads) in enumerate(zipped):
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue

                if g.is_sparse:
                    raise RuntimeError(
                        'Adamax does not support sparse gradients'
                    )

                state = self.state[group_idx][p_idx]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = _torch.zeros_like(p.data)
                    state['exp_inf'] = _torch.zeros_like(p.data)

                exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
                beta1, beta2 = group['betas']
                eps = group['eps']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    g = _add(g, group['weight_decay'], p)

                # Update biased first moment estimate
                state['exp_avg'] = exp_avg = _add(
                    exp_avg.mul(beta1), 1 - beta1, g
                )
                # Update the exponentially weighted infinity norm.
                state['exp_inf'] = exp_inf = exp_inf.mul(beta2).unsqueeze(0)
                norm_buf = _torch.cat(
                    [exp_inf, _add(g.abs(), eps).unsqueeze(0)], 0
                )
                exp_inf, _ = _torch.max(norm_buf, 0, keepdim=False)
                state['exp_inf'] = exp_inf

                bias_correction = 1 - beta1**state['step']
                clr = group['lr'] / bias_correction

                group['params'][p_idx] = _addcdiv(p, -clr, exp_avg, exp_inf)


class DifferentiableASGD(DifferentiableOptimizer):
    r"""A differentiable version of the ASGD optimizer.

    This optimizer creates a gradient tape as it updates parameters."""

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:

        zipped = zip(self.param_groups, grouped_grads)
        for group_idx, (group, grads) in enumerate(zipped):
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue

                if g.is_sparse:
                    raise RuntimeError('ASGD does not support sparse gradients')
                state = self.state[group_idx][p_idx]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['eta'] = group['lr']
                    state['mu'] = 1
                    state['ax'] = _torch.zeros_like(p.data)

                state['step'] += 1

                if group['weight_decay'] != 0:
                    g = _add(g, group['weight_decay'], p)

                # decay term
                p = p.mul(1 - group['lambd'] * state['eta'])

                # update parameter
                group['params'][p_idx] = _add(p, -state['eta'], g)

                # averaging
                if state['mu'] != 1:
                    state['ax'] = _add(
                        state['ax'],
                        p.sub(state['ax']).mul(state['mu'])
                    )
                else:
                    state['ax'] = p

                # update eta and mu
                state['eta'] = (
                    group['lr'] / _math.pow(
                    (1 + group['lambd'] * group['lr'] * state['step']),
                    group['alpha']
                    )   
                )
                state['mu'] = 1 / max(1, state['step'] - group['t0'])


class DifferentiableRMSprop(DifferentiableOptimizer):
    r"""A differentiable version of the RMSprop optimizer.

    This optimizer creates a gradient tape as it updates parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _warnings.warn(
            "Differentiable RMSprop suffers from gradient correctness issues. "
            "Consider using another optimizer until we fix these..."
        )

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:

        zipped = zip(self.param_groups, grouped_grads)
        for group_idx, (group, grads) in enumerate(zipped):
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue

                if g.is_sparse:
                    raise RuntimeError(
                        'RMSprop does not support sparse gradients'
                    )
                state = self.state[group_idx][p_idx]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = _torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = _torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = _torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    g = _add(g, group['weight_decay'], p)

                square_avg = _addcmul(square_avg.mul(alpha), 1 - alpha, g, g)
                state['square_avg'] = square_avg

                # NB: This prevents nans but is not sufficient to recover
                # correct gradients.
                mask = square_avg == 0.
                _maybe_mask(square_avg, mask)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg = _add(grad_avg.mul(alpha), 1 - alpha, g)
                    state['grad_avg'] = grad_avg
                    eps = group['eps']
                    avg = _add(
                        _addcmul(square_avg, -1, grad_avg, grad_avg).sqrt(), eps
                    )
                else:
                    avg = _add(square_avg.sqrt(), group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf = _addcdiv(buf.mul(group['momentum']), g, avg)
                    state['momentum_buffer'] = buf
                    p = _add(p, -group['lr'], buf)
                else:
                    p = _addcdiv(p, -group['lr'], g, avg)

                group['params'][p_idx] = p


class DifferentiableRprop(DifferentiableOptimizer):
    r"""A differentiable version of the Rprop optimizer.

    This optimizer creates a gradient tape as it updates parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _warnings.warn(
            "Differentiable Rprop (correctly) yields zero second order "
            "gradients, as only the sign of the gradient is used in updates. "
            "Future versions will offer higher order gradients based on a "
            "continuous relaxation of the forward pass."
        )

    def _update(self, grouped_grads: _GroupedGradsType, **kwargs) -> None:

        zipped = zip(self.param_groups, grouped_grads)
        for group_idx, (group, grads) in enumerate(zipped):
            for p_idx, (p, g) in enumerate(zip(group['params'], grads)):
                if g is None:
                    continue

                if g.is_sparse:
                    raise RuntimeError(
                        'Rprop does not support sparse gradients'
                    )

                state = self.state[group_idx][p_idx]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['prev'] = _torch.zeros_like(p.data)
                    state['step_size'] = g.new().resize_as_(g).fill_(
                        group['lr']
                    )

                etaminus, etaplus = group['etas']
                step_size_min, step_size_max = group['step_sizes']
                step_size = state['step_size']

                state['step'] += 1

                sign = g.mul(state['prev']).sign()
                sign[sign.gt(0)] = etaplus
                sign[sign.lt(0)] = etaminus
                sign[sign.eq(0)] = 1

                # update stepsizes with step size updates
                step_size = step_size.mul(sign).clamp(
                    step_size_min, step_size_max
                )
                state['step_size'] = step_size

                # for dir<0, dfdx=0
                # for dir>=0 dfdx=dfdx
                g = _torch.where(sign.eq(etaminus), _torch.zeros_like(g), g)

                # update parameters
                group['params'][p_idx] = _addcmul(p, -1, g.sign(), step_size)

                state['prev'] = g.clone()


_OptMappingType = _typing.Dict[_torch.optim.Optimizer, _typing.Type[DifferentiableOptimizer]]
_opt_mapping: _OptMappingType = {
    _torch.optim.Adadelta: DifferentiableAdadelta,
    _torch.optim.Adagrad: DifferentiableAdagrad,
    _torch.optim.Adam: DifferentiableAdam,
    _torch.optim.AdamW: DifferentiableAdamW,
    _torch.optim.Adamax: DifferentiableAdamax,
    _torch.optim.ASGD: DifferentiableASGD,
    _torch.optim.RMSprop: DifferentiableRMSprop,
    _torch.optim.Rprop: DifferentiableRprop,
    _torch.optim.SGD: DifferentiableSGD,
}


def get_diff_optim(
    opt: _torch.optim.Optimizer,
    reference_params: _typing.Iterable[_torch.Tensor],
    fmodel: _typing.Optional[_patch._MonkeyPatchBase] = None,
    device: _typing.Optional[_torch.device] = None,
    override: _typing.Optional[_OverrideType] = None,
    track_higher_grads: bool = True,
    **kwargs
) -> DifferentiableOptimizer:
    r"""Construct/initialize a differentiable version of an existing optimizer.

    Args:
        opt: an existing optimizer, assumed to be an instance of
            ``torch.optim.Optimizer``, of a supported type which is either defined
            in ``torch.optim``, or a custom implemantation which has been added to
            higher at runtime by using ``higher.register_optim``. We assume this
            optimizer tracks the parameters (or some subset thereof) of a single
            ``torch.nn.Module`` instance, with support for parameter groups.
        reference_params: the parameters of the module tracked by ``opt``, as
            returned by ``module.parameters()``.
        fmodel (optional): a patched version of the ``module`` tracked by ``opt``.
            It is assumed this patched instance has a view on its latest fast
            weights through ``fmodel.parameters()``. If provided, it is not
            necessary to pass the fast weights explicitly to the differentiable
            optimizer's ``step`` function via the keyword arg ``params``. If not
            provided, the fast weights to update must be provided to ``step``.
        device (optional): the device to cast the optimizer state to when
            creating the differentiable optimizer. If not provided, the same
            device as used for the parameters tracked by ``opt`` will be used.
        override (optional): a dictionary mapping optimizer settings (i.e.
            those which would be passed to the optimizer constructor or
            provided within parameter groups) to either singleton lists of
            override values, or to a list of override values of length equal to
            the number of parameter groups. If a single override is provided for
            a keyword, it is used for all parameter groups. If a list is
            provided, the ``i``\ th element of the list overrides the corresponding
            setting in the ``i``\ th parameter group. This permits the passing of
            tensors requiring gradient to differentiable optimizers for use as
            optimizer settings.
        track_higher_grads: if True, during unrolled optimization the graph be
            retained, and the fast weights will bear grad funcs, so as to permit
            backpropagation through the optimization process. Setting this to
            False allows the returned differentiable optimizer to be used in
            "test mode", without potentially tracking higher order gradients.
            This can be useful when running the training loop at test time,
            e.g. in k-shot learning experiments, without incurring a significant
            memory overhead.

    Returns:
        An initialized ``DifferentiableOptimizer`` instance of the right subtype.
  """
    if type(opt) in _opt_mapping:
        return _opt_mapping[type(opt)](
            opt,
            reference_params,
            fmodel=fmodel,
            device=device,
            override=override,
            track_higher_grads=track_higher_grads,
            **kwargs
        )
    else:
        raise ValueError(
            "Optimizer type {} not supported by higher yet.".format(type(opt))
        )


def create_diff_optim(
        opt_type: _typing.Type[_torch.optim.Optimizer],
        opt_kwargs: _typing.Optional[_typing.Dict[str, _typing.Any]] = None,
        params: _typing.Optional[_typing.List[_torch.Tensor]] = None,
        fmodel: _typing.Optional[_patch._MonkeyPatchBase] = None,
        device: _typing.Optional[_torch.device] = None,
        override: _typing.Optional[_OverrideType] = None,
        track_higher_grads: bool = True,
        **kwargs
) -> DifferentiableOptimizer:
    r"""Construct a differentiable version of an new optimizer.

    Args:
        opt_type: the type (constructor) for a torch.optim.Optimizer subtype
            from amongst the types supported by the library, or registered with
            it a runtime.
        opt_kwargs: a dictionary of keywords to be passed to the optimizer
            constructor.
        params (optional): a list of (fast) weights which the differentiable
            optimizer will update. These must be provided if fmodel is not
            provided. If both, these will be used in lieu. These will only
            be used for shape inference when initializing the optimizer.
            This argument can also take the same format as parameter groups,
            i.e. an iterable over dictionaries which contain the 'params' key
            with fast weights as value, and group-specific hyperparameters.
        fmodel (optional): a patched version of the ``module`` tracked by ``opt``.
            It is assumed this patched instance has a view on its latest fast
            weights through ``fmodel.parameters()``. If provided, it is not
            necessary to pass the fast weights explicitly to the differentiable
            optimizer's ``step`` function via the keyword arg ``params``. If not
            provided, the fast weights to update must be provided to ``step``.
        device (optional): the device to cast the optimizer state to when
            creating the differentiable optimizer. If not provided, the same
            device as used for the parameters tracked by ``opt`` will be used.
        override (optional): a dictionary mapping optimizer settings (i.e.
            those which would be passed to the optimizer constructor or
            provided within parameter groups) to either singleton lists of
            override values, or to a list of override values of length equal to
            the number of parameter groups. If a single override is provided for
            a keyword, it is used for all parameter groups. If a list is
            provided, the ``i``\ th element of the list overrides the corresponding
            setting in the ``i``\ th parameter group. This permits the passing of
            tensors requiring gradient to differentiable optimizers for use as
            optimizer settings.
        track_higher_grads: if True, during unrolled optimization the graph be
            retained, and the fast weights will bear grad funcs, so as to permit
            backpropagation through the optimization process. Setting this to
            False allows the returned differentiable optimizer to be used in
            "test mode", without potentially tracking higher order gradients.
            This can be useful when running the training loop at test time,
            e.g. in k-shot learning experiments, without incurring a significant
            memory overhead.

    Returns:
        An initialized ``DifferentiableOptimizer`` instance of the right subtype.
  """

    if opt_type in _opt_mapping:
        if params is not None:
            params = list(params)
            if isinstance(params[0], dict):
                dummy = [
                    {
                    k: _torch.zeros_like(v, requires_grad=True)
                    if k == "params" else v
                    for k, v in group.items()
                    } for group in params
                ]
            else:
                dummy = [
                    _torch.zeros_like(p, requires_grad=True)
                    for p in params
                ]
        elif fmodel is not None:
            dummy = [
                _torch.zeros_like(p, requires_grad=True)
                for p in fmodel.parameters()
            ]
        else:
            raise ValueError("Must specify one of fmodel or params in kwargs.")

        opt_kwargs = {} if opt_kwargs is None else opt_kwargs
        opt = opt_type(dummy, **opt_kwargs)

        return _opt_mapping[opt_type](
            opt,
            dummy,
            fmodel=fmodel,
            device=device,
            override=override,
            track_higher_grads=track_higher_grads,
            **kwargs
        )
    else:
        raise ValueError(
            "Optimizer type {} not supported by higher yet.".format(opt_type)
        )


def register_optim(
        optim_type: _torch.optim.Optimizer,
        diff_optim_type: _typing.Type[DifferentiableOptimizer]
) -> None:
    r"""Registers a new optimizer type for use with higher functions.

    Args:
        optim_type: the type of a new optimizer, assumed to be an instance of
            ``torch.optim.Optimizer``.
        diff_optim_type: the type of a new differentiable optimizer, assumed to
            be an instance of ``higher.optim.DifferentiableOptimizer`` with
            functionally equivalent logic to ``optim_type``.
    """
    _opt_mapping[optim_type] = diff_optim_type


def get_trainable_opt_params(
        opt: _torch.optim.Optimizer, device: _typing.Optional[_torch.device] = None
) -> _OverrideType:
    r"""Get an override dictionary from an optimizer instance.

    Args:
        opt: the optimizer to obtain an override dictionary from.
        device (optional): the device to cast the learnable tensors to.

    Returns:
        A dictionary of the format expected for the override kwarg of
        differentiable optimizers. It is initialized with trainable tensors
        with as values those float and int hyperparameters found in the
        optimizer's parameter groups (or stuctures containing these).
        Heuristically, hyperparameters containing mixtures of differentiable
        and non-differentiable types will be ignored (and must be manually
        specified when constructing an override dict).
    """
    override: _OverrideType = _collections.defaultdict(list)

    def map_fn(x: _typing.Union[_torch.Tensor, int, float]) -> _torch.Tensor:
        if isinstance(x, _torch.Tensor):
            return x.clone().detach().requires_grad_()
        else:
            return _torch.tensor(float(x), device=device, requires_grad=True)

    for group in opt.param_groups:
        for k, v in group.items():
            if k == "params":
                # Ignore actual model parameters tracked by optim
                continue

            # Ignore hyperparameters that aren't structures containing ints
            # or floats
            if all(
                    isinstance(x, int) or isinstance(x, float)
                    for x in _utils.flatten(v)
            ):
                override[k].append(_utils._recursive_map(v, map_fn))

    return override


def apply_trainable_opt_params(
    opt: _torch.optim.Optimizer, override: _OverrideType
) -> None:
    r"""Apply learned hyperparameters back to original optimizer.

    Args:
        opt: the original optimizer. The hyperparameters in its parameter groups
            will be modified in place.
        override: dictionary of the format used for the override kwarg of
            differentiable optimizers.
    """
    for k, v in override.items():
        # Sanity check
        if (len(v) != 1) and (len(v) != len(opt.param_groups)):
            raise ValueError(
                "Mismatch between the number of override tensors for "
                "optimizer parameter {} and the number of "
                "parameter groups.".format(k)
            )
        for group_idx, group in enumerate(opt.param_groups):
            replacement = v[0] if len(v) is 1 else v[group_idx]
            group[k] = _recursive_apply(replacement, group[k])


## Local utility functions
# TODO(egrefen): use funcs below instead of x._add, in diffopt
def _add(
    tensor: _torch.Tensor,
    a1: _typing.Union[float, int, _torch.Tensor],
    a2: _typing.Optional[_torch.Tensor] = None
) -> _torch.Tensor:
    if a2 is None:
        value: _typing.Union[_torch.Tensor, float] = 1.
        other = a1
    else:
        value = a1
        other = a2
    return tensor + (value * other)


def _addcdiv(
    tensor: _torch.Tensor,
    a1: _typing.Union[float, int, _torch.Tensor],
    a2: _torch.Tensor,
    a3: _typing.Optional[_torch.Tensor] = None
) -> _torch.Tensor:
    if a3 is None:
        value: _typing.Union[_torch.Tensor, float] = 1.
        tensor1 = a1
        tensor2 = a2
    else:
        value = a1
        tensor1 = a2
        tensor2 = a3
    return tensor + value * (tensor1 / tensor2)


def _addcmul(
    tensor: _torch.Tensor,
    a1: _typing.Union[float, int, _torch.Tensor],
    a2: _torch.Tensor,
    a3: _typing.Optional[_torch.Tensor] = None
) -> _torch.Tensor:
    if a3 is None:
        value: _typing.Union[_torch.Tensor, float] = 1.
        tensor1 = a1
        tensor2 = a2
    else:
        value = a1
        tensor1 = a2
        tensor2 = a3
    return tensor + (value * tensor1 * tensor2)


# TODO(egrefen): this probably could be refactored into utils
def _recursive_apply(
    replacement: _typing.Union[list, tuple, dict, set, _torch.Tensor],
    target: _typing.Union[_torch.Tensor, int, float]
) -> _typing.Union[_torch.Tensor, int, float]:
    if not isinstance(replacement, type(target)):
        if (
            isinstance(replacement, _torch.Tensor) and
            not _utils._is_container(target)
        ):
            return type(target)(replacement.item())
        raise ValueError(
            "Expected an non-container type for target, but got {} with value "
            "{}".format(type(target), target)
        )
    elif (
        isinstance(replacement, _torch.Tensor) and
        isinstance(target, _torch.Tensor)
    ):
        replacement = replacement.to(target.device)
        target.data = replacement.data
        return target
    if isinstance(target, list):
        return type(target)(
            [_recursive_apply(r, t) for r, t in zip(replacement, target)]
        )
    elif isinstance(target, tuple):
        return type(target)(
            [_recursive_apply(r, t) for r, t in zip(replacement, target)]
        )
    elif isinstance(replacement, dict) and isinstance(target, dict):
        return type(target)(
            {k: _recursive_apply(r, t)
            for (_, r), (k, t) in zip(replacement.items(), target.items())}
        )
    elif isinstance(target, set):
        return type(target)(
            {_recursive_apply(r, t)
             for r, t in zip(replacement, target)}
        )
    else:
        raise ValueError(
            "Couldn't apply replacement of type {} to target of type "
            "{}".format(type(replacement), type(target))
        )
