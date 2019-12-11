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

"""Functions for making ``torch.nn.Module`` subclass instances stateless."""

import abc as _abc
from collections import OrderedDict as _OrderedDict
import typing as _typing
import warnings as _warnings

import torch as _torch

from . import utils as _utils

# ==============================================================================
# Helper functions and attributes for MonkeyPatch modules.
# ==============================================================================

_internal_attrs = {
    '_backend', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks',
    '_forward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks',
    '_modules'
}

_BufferType = _typing.Dict[str, _typing.Optional[_torch.Tensor]]


def _patched_parameters(
    self, recurse: bool = True, time: _typing.Optional[int] = None
) -> _typing.Iterable[_torch.Tensor]:
    r"""Returns an iterator over monkey patched module fast parameters.

    Args:
        recurse (bool): if True, then yields fast parameters of this module
            and all submodules. Otherwise, this *still* yields parameters of
            this module and all submodules, and raises a warning. This keyword
            exists only to satisfy API compatibility with
            ``torch.nn.Module.parameters``.
        time (int or None): if None, the most recent fast parameters are
            provided. The int provided stands for the number of steps since the
            module was created. *Note* that the step counter is incremented
            every time parameters are updated, so this may not align with number
            of training or evaluations steps.

    Yields:
        Parameter: module fast weights.
    """
    if getattr(self, "_fast_params", None) is None:
        raise Exception(
            "Tried to get fast weights of a monkey patched module which does "
            "not encapsulate fast weights."
        )

    if not recurse:
        _warnings.warn(
            "Calling parameters with recurse=False on a monkey patched module "
            "still returns all the fast weights of of nested patched modules."
        )

    time = -1 if time is None else time

    for p in self._fast_params[time]:
        yield p


class _MonkeyPatchBase(_abc.ABC, _torch.nn.Module):
    @_abc.abstractmethod
    def __init__(self) -> None:
        self._param_mapping: _typing.List[int] = []

    def forward(self):
        raise NotImplementedError(
            "The monkey-patching logic has failed to override self.forward "
            "on the new module, or you tried calling forward on a patched "
            "version of a module which doesn't have forward (e.g. ModuleList)."
        )

    def _expand_params(
        self, params: _typing.List[_torch.Tensor]
    ) -> _typing.List[_torch.Tensor]:
        expanded = []
        for index in self._param_mapping:
            expanded.append(params[index])
        return expanded

    @property
    def init_fast_params(self):
        return self._fast_params[0]

    @property
    def fast_params(self):
        return None if self._fast_params is None else self._fast_params[-1]

    @fast_params.setter
    def fast_params(self, value):
        value = list(value)
        if self._fast_params is None:
            self._fast_params = []
        self._fast_params.append(value)


def buffer_sync(
    module: _torch.nn.Module,
    fmodule: _MonkeyPatchBase,
    device: _typing.Optional[_torch.device] = None
) -> None:
    r"""One off sync (copy) of buffers in ``fmodule`` with those from ``module``.
    """
    for key, value in module._buffers.items():
        if not _torch.is_tensor(value):
            fmodule._buffers[key] = value
        elif device is None:
            fmodule._buffers[key] = value.clone().detach()
        else:
            fmodule._buffers[key] = value.clone().detach().to(device)

    for name, child in module._modules.items():
        if name in fmodule._modules:
            buffer_sync(child, fmodule._modules[name], device)
        else:
            raise KeyError(
                "Did not find expected submodule "
                "{} of monkey-patched module {}.".format(name, fmodule)
            )


# ==============================================================================
# Helper class to use instead of actual torch.nn.Parameters when patching.
# ==============================================================================


class _ParameterPlaceholder():
    def __init__(self, name: str) -> None:
        self._param_name = name

    def __repr__(self) -> str:
        return 'Parameter placeholder ("{}")'.format(self._param_name)


_ParameterPlaceholder.__name__ = "ParameterPlaceholder"
_ParameterPlaceholder.__qualname__ = "ParameterPlaceholder"

# ==============================================================================
# Helper function for recursively patching submodules.
# ==============================================================================


def _make_functional(
    module: _torch.nn.Module,
    params_box: _typing.Sequence[_typing.Optional[_typing.List[_torch.Tensor]]],
    params_offset: int
) -> _typing.Tuple[int, _MonkeyPatchBase, _typing.Type[_MonkeyPatchBase]]:

    if isinstance(module, _MonkeyPatchBase):
        raise ValueError(
            "Monkey-patching monkey-patched modules is untested uncharted "
            "territory, so we're going to assume it's done in error. If you "
            "are doing this intentionally and need this to be supported, "
            "contact the developers of this library."
        )

    param_names = list(
        name for name in module._parameters.keys()
        if module._parameters[name] is not None
    )

    _ModuleType: _typing.Type[_torch.nn.Module] = module.__class__

    # type checking of next line disabled as mypy is iffy with dynamic types
    class MonkeyPatched(_ModuleType, _MonkeyPatchBase):  # type: ignore
        _wrapped_name = type(module).__name__

        def __init__(self, original_params) -> None:
            _torch.nn.Module.__init__(self)

            self._fast_params = None
            self._param_names = param_names

            self._original_params = original_params

            # for pretty printing
            self._parameters = _OrderedDict(
                (name, _ParameterPlaceholder(name))
                for name in self._param_names
            )
            self._modules: _typing.Dict[str, _MonkeyPatchBase] = _OrderedDict()

        def __setattr__(self, name, value):
            def remove_from(*dicts):
                for d in dicts:
                    if name in d:
                        del d[name]

            params = self.__dict__.get('_parameters')
            if params is not None and name in params:
                if not isinstance(value, _torch.Tensor):
                    raise TypeError("Require Tensor as fast weights. "
                                    "Got {}".format(_torch.typename(value)))
                # Change parameters in place, usually during boxed_forward pass
                self._parameters[name] = value
            else:
                modules = self.__dict__.get('_modules')
                if isinstance(value, _torch.nn.Module):
                    if modules is None:
                        raise AttributeError(
                            "cannot assign module before Module.__init__() "
                            "call"
                        )
                    remove_from(self.__dict__, self._parameters, self._buffers)
                    modules[name] = value
                elif modules is not None and name in modules:
                    if value is not None:
                        raise TypeError(
                            (
                                "cannot assign '{}' "
                                "as child module '{}'"
                                "(torch.nn.Module or None expected)"
                            ).format(_torch.typename(value), name)
                        )
                    modules[name] = value
                else:
                    buffers = self.__dict__.get('_buffers')
                    if buffers is not None and name in buffers:
                        if value is not None and not isinstance(
                            value, _torch.Tensor
                        ):
                            raise TypeError(
                                "cannot assign '{}' as buffer '{}' "
                                "(torch.Tensor or None expected)".format(
                                    _torch.typename(value), name
                                )
                            )
                        buffers[name] = value
                    else:
                        object.__setattr__(self, name, value)

        def parameters(self) -> _typing.Iterable[_torch.Tensor]:
            r"""This should only be used to check shape/dtype of original params.
            """
            return self._original_params

    MonkeyPatched.__name__ = "InnerFunctional" + type(module).__name__
    MonkeyPatched.__qualname__ = MonkeyPatched.__name__

    fmodule = MonkeyPatched(module.parameters())

    # use 1 as dummy list item since we are only counting
    num_params = len([1 for p in module._parameters.values() if p is not None])

    # Copy over all attributes
    for name, attr in module.__dict__.items():
        if name in _internal_attrs:
            continue
        setattr(fmodule, name, attr)

    # Deal with "None"-style params
    for name, attr in module.__dict__['_parameters'].items():
        if isinstance(attr, _torch.nn.Parameter):
            continue
        else:
            setattr(fmodule, name, attr)

    child_params_offset = params_offset + num_params
    for name, child in module._modules.items():
        child_params_offset, fchild, _ = _make_functional(
            child, params_box, child_params_offset
        )
        fmodule._modules[name] = fchild
        setattr(fmodule, name, fchild)

    true_forward = type(module).forward

    def patched_forward(self, *args, **kwargs):
        for name, param in zip(
            self._param_names,
            params_box[0][params_offset:params_offset + num_params]
        ):
            setattr(self, name, param)
        return true_forward(self, *args, **kwargs)

    setattr(MonkeyPatched, "forward", patched_forward)

    return child_params_offset, fmodule, type(fmodule)


def _update_patched_params(
    fmodule: _MonkeyPatchBase,
    params_box: _typing.Sequence[_typing.List[_torch.Tensor]],
    params_offset: int
) -> int:
    num_params = len([1 for p in fmodule._parameters.values() if p is not None])
    child_params_offset = params_offset + num_params
    for name, child in fmodule._modules.items():
        child_params_offset = _update_patched_params(
            child, params_box, child_params_offset
        )
    for name, param in zip(
        fmodule._param_names,
        params_box[0][params_offset:params_offset + num_params]
    ):
        setattr(fmodule, name, param)
    return child_params_offset


# ==============================================================================
# The main function which does the monkey patching.
# ==============================================================================
_EncapsulatorType = _typing.Optional[
    _typing.Callable[[_MonkeyPatchBase, _torch.nn.Module], None]]


def make_functional(
    module: _torch.nn.Module,
    encapsulator: _EncapsulatorType = None
) -> _MonkeyPatchBase:
    r"""Returns a stateless version of an ``nn.Module`` instance."""
    params_box = [None]
    _, fmodule, MonkeyPatched = _make_functional(module, params_box, 0)
    top_name = "Functional" + MonkeyPatched._wrapped_name
    MonkeyPatched.__name__ = MonkeyPatched.__qualname__ = top_name

    MonkeyPatched.boxed_forward = MonkeyPatched.forward

    param_mapping = _utils._get_param_mapping(module, [], [])
    setattr(fmodule, "_param_mapping", param_mapping)

    def _patched_forward(self, *args, **kwargs):
        if "params" in kwargs:
            params = kwargs.pop('params')
            self.fast_params = params  # update view on latest fast params
        elif self.fast_params is None:
            raise ValueError(
                "params keyword must be provided if patched module not "
                "tracking its own fast parameters"
            )

        params_box[0] = self._expand_params(self.fast_params)
        return self.boxed_forward(*args, **kwargs)

    def _update_params(self, params):
        self.fast_params = params
        params = self._expand_params(params)
        _update_patched_params(self, [params], 0)

    setattr(MonkeyPatched, "forward", _patched_forward)
    setattr(MonkeyPatched, "parameters", _patched_parameters)
    setattr(MonkeyPatched, "update_params", _update_params)

    if encapsulator is not None:
        encapsulator(fmodule, module)

    return fmodule


# ==============================================================================
# Convenience functions and decorators for hiding away a lot of the complexity
# of creating patched modules, taking their parameters, and linking patched
# modules to a differentiable optimizer.
# ==============================================================================


def monkeypatch(
    module: _torch.nn.Module,
    device: _typing.Optional[_torch.device] = None,
    copy_initial_weights: bool = True
) -> _MonkeyPatchBase:
    r"""Create a monkey-patched stateless version of a module.

    This function produces a monkey-patched version of a module, and returns a
    copy of its parameters for use as fast weights. Where the original module
    or any of its submodules have state (e.g. batch norm), this will be copied
    too, but further updates (e.g. during inner loop training) will cause these
    to diverge without changing the state of the original module.

    Args:
        module: a ``torch.nn.Module`` subclass instance.
        device (optional): a device to cast the fast weights and state to.
        copy_initial_weights: if True, the weights of the patched module are
            copied to form the initial weights of the patched module, and thus
            are not part of the gradient tape when unrolling the patched module.
            If this is set to False, the actual module weights will be the
            initial weights of the patched module. This is useful when doing
            MAML, for example.

    Returns:
        ``fmodule``: a "stateless" version of the original module, for which calls
        to forward take the additional kwarg-only parameter ``params``, which
        should be a list of torch tensors requiring gradients, ideally
        provided by this function (see below) or by an update step from one
        of the optimizers in ``higher.optim``.
    """

    def encapsulator(
        fmodule: _MonkeyPatchBase, module: _torch.nn.Module
    ) -> None:
        if copy_initial_weights:
            params = _utils.get_func_params(module, device=device)
        else:
            params = [
                p.clone() if device is None else p.clone().to(device)
                for p in module.parameters()
            ]
        buffer_sync(module, fmodule, device)
        fmodule.update_params(params)

    fmodule = make_functional(module, encapsulator=encapsulator)

    return fmodule
