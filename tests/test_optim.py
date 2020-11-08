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

"""Unit tests for higher.optim."""
import unittest
import copy
from parameterized import parameterized

import torch
from torch import nn, optim

import higher

_test_param_sweep = [
    (
        "simple_model_sgd",
        lambda self: self._model,
        optim.SGD,
    ),
    (
        "simple_model_sgd_momentum",
        lambda self: self._model,
        optim.SGD,
        {
            'momentum': .1
        },
    ),
    (
        "simple_model_sgd_nesterov",
        lambda self: self._model,
        optim.SGD,
        {
            'momentum': .1,
            'nesterov': True
        },
    ),
    (
        "simple_model_sgd_weight_decay",
        lambda self: self._model,
        optim.SGD,
        {
            'weight_decay': .1
        },
    ),
    (
        "share_weight_model_sgd",
        lambda self: self._shared_param_model,
        optim.SGD,
    ),
    (
        "share_weight_seq_model_sgd",
        lambda self: self._shared_param_seq_model,
        optim.SGD,
    ),
    (
        "partially_used_model_sgd",
        lambda self: self._partially_used_model,
        optim.SGD,
    ),
    (
        "simple_model_adam",
        lambda self: self._model,
        optim.Adam,
    ),
    (
        "simple_model_adam_weight_decay",
        lambda self: self._model,
        optim.Adam,
        {
            "weight_decay": 0.1
        },
    ),
    (
        "share_weight_model_adam",
        lambda self: self._shared_param_model,
        optim.Adam,
    ),
    (
        "share_weight_seq_model_adam",
        lambda self: self._shared_param_seq_model,
        optim.Adam,
    ),
    (
        "partially_used_model_adam",
        lambda self: self._partially_used_model,
        optim.Adam,
    ),
    (
        "simple_model_adamw",
        lambda self: self._model,
        optim.AdamW,
    ),
    (
        "simple_model_adamw_weight_decay",
        lambda self: self._model,
        optim.AdamW,
        {
            "weight_decay": 0.1
        },
    ),
    (
        "share_weight_model_adamw",
        lambda self: self._shared_param_model,
        optim.AdamW,
    ),
    (
        "share_weight_seq_model_adamw",
        lambda self: self._shared_param_seq_model,
        optim.AdamW,
    ),
    (
        "partially_used_model_adamw",
        lambda self: self._partially_used_model,
        optim.AdamW,
    ),
    (
        "simple_model_adadelta",
        lambda self: self._model,
        optim.Adadelta,
    ),
    (
        "simple_model_adadelta_weight_decay",
        lambda self: self._model,
        optim.Adadelta,
        {
            "weight_decay": 0.1
        },
    ),
    (
        "share_weight_model_adadelta",
        lambda self: self._shared_param_model,
        optim.Adadelta,
    ),
    (
        "share_weight_seq_model_adadelta",
        lambda self: self._shared_param_seq_model,
        optim.Adadelta,
    ),
    (
        "partially_used_model_adadelta",
        lambda self: self._partially_used_model,
        optim.Adadelta,
    ),
    (
        "simple_model_adagrad",
        lambda self: self._model,
        optim.Adagrad,
    ),
    (
        "simple_model_adagrad_weight_decay",
        lambda self: self._model,
        optim.Adagrad,
        {
            "weight_decay": 0.1
        },
    ),
    (
        "simple_model_adagrad_lr_decay",
        lambda self: self._model,
        optim.Adagrad,
        {
            "lr_decay": 0.1
        },
    ),
    (
        "share_weight_model_adagrad",
        lambda self: self._shared_param_model,
        optim.Adagrad,
    ),
    (
        "share_weight_seq_model_adagrad",
        lambda self: self._shared_param_seq_model,
        optim.Adagrad,
    ),
    (
        "partially_used_model_adagrad",
        lambda self: self._partially_used_model,
        optim.Adagrad,
    ),
    (
        "simple_model_adamax",
        lambda self: self._model,
        optim.Adamax,
    ),
    (
        "simple_model_adamax_weight_decay",
        lambda self: self._model,
        optim.Adamax,
        {
            "weight_decay": 0.1
        },
    ),
    (
        "share_weight_model_adamax",
        lambda self: self._shared_param_model,
        optim.Adamax,
    ),
    (
        "share_weight_seq_model_adamax",
        lambda self: self._shared_param_seq_model,
        optim.Adamax,
    ),
    (
        "partially_used_model_adamax",
        lambda self: self._partially_used_model,
        optim.Adamax,
    ),
    (
        "simple_model_asgd",
        lambda self: self._model,
        optim.ASGD,
    ),
    (
        "simple_model_asgd_weight_decay",
        lambda self: self._model,
        optim.ASGD,
        {
            "weight_decay": 0.1
        },
    ),
    (
        "share_weight_model_asgd",
        lambda self: self._shared_param_model,
        optim.ASGD,
    ),
    (
        "share_weight_seq_model_asgd",
        lambda self: self._shared_param_seq_model,
        optim.ASGD,
    ),
    (
        "partially_used_model_asgd",
        lambda self: self._partially_used_model,
        optim.ASGD,
    ),
    # (
    #     "simple_model_rmsprop",
    #     lambda self: self._model,
    #     optim.RMSprop,
    # ),
    # (
    #     "simple_model_rmsprop_momentum",
    #     lambda self: self._model,
    #     optim.RMSprop,
    #     {
    #         "momentum": 0.1
    #     },
    # ),
    # (
    #     "simple_model_rmsprop_weight_decay",
    #     lambda self: self._model,
    #     optim.RMSprop,
    #     {
    #         "weight_decay": 0.1
    #     },
    # ),
    # (
    #     "simple_model_rmsprop_centered",
    #     lambda self: self._model,
    #     optim.RMSprop,
    #     {
    #         "centered": True
    #     },
    # ),
    # (
    #     "share_weight_model_rmsprop",
    #     lambda self: self._shared_param_model,
    #     optim.RMSprop,
    # ),
    # (
    #     "share_weight_seq_model_rmsprop",
    #     lambda self: self._shared_param_seq_model,
    #     optim.RMSprop,
    # ),
    # (
    #     "partially_used_model_rmsprop",
    #     lambda self: self._partially_used_model,
    #     optim.RMSprop,
    # ),
]


class _NestedEnc(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class _Enc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = _NestedEnc(torch.nn.Linear(4, 2))
        self.e2 = _NestedEnc(self.e1.f)

    def forward(self, x):
        return self.e1(x) + self.e2(x)


class _PartiallyUsed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.rand(4, 3, requires_grad=True))
        self.b = torch.nn.Parameter(torch.rand(4, 3, requires_grad=True))

    def forward(self, x):
        return x @ self.a


def finite_difference(model, closure, eps):
    fd_params = []
    ground = closure()
    for param in model.parameters():
        fd_param = torch.zeros_like(param)
        for p, fdp in zip(param.flatten(), fd_param.flatten()):
            p.data.add_(eps)
            fdp.data.fill_((closure() - ground) / eps)
            p.data.sub_(eps)
        fd_params.append(fd_param)
    return fd_params


class TestOptim(unittest.TestCase):
    """Test case for the optim module."""

    def setUp(self):
        self._model = nn.Sequential(
            nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 4, bias=None),
            nn.Sigmoid(), nn.Linear(4, 2)
        )

        self._shared_param_model = _Enc()

        proj = nn.Linear(4, 3)
        self._shared_param_seq_model = nn.Sequential(
            proj, nn.ReLU(), nn.Linear(3, 4), nn.Sigmoid(), proj, nn.Tanh(),
            nn.Linear(3, 2)
        )

        self._partially_used_model = _PartiallyUsed()

    @parameterized.expand(_test_param_sweep)
    def testDiffOptCorrectness(
        self, _, model_builder, opt_builder, kwargs=None
    ):
        kwargs = {} if kwargs is None else kwargs
        lr = .1
        model = model_builder(self)
        opt = opt_builder(model.parameters(), lr=lr, **kwargs)

        self._run_correctness_test(opt, model)

    @parameterized.expand(_test_param_sweep)
    def testDiffOptGroupedParam(
        self, _, model_builder, opt_builder, kwargs=None
    ):
        kwargs = {} if kwargs is None else kwargs
        lr = .1
        left_lr = .2
        model = model_builder(self)
        full_parameters = list(model.parameters())
        half = len(full_parameters) // 2
        left_parameters = full_parameters[:half]
        right_parameters = full_parameters[half:]
        param_groups = [
            {
                'params': left_parameters,
                'lr': left_lr
            },
            {
                'params': right_parameters
            },
        ]
        opt = opt_builder(param_groups, lr=lr, **kwargs)

        self._run_correctness_test(opt, model)

    def _run_correctness_test(self, opt, model, override=None):
        for i in range(10):
            fmodel = higher.patch.monkeypatch(model)
            diffopt = higher.optim.get_diff_optim(
                opt, model.parameters(), fmodel, override=override
            )
            for j in range(3):
                opt.zero_grad()
                x = torch.rand(10, 4)

                y_model = model(x)
                y_fmodel = fmodel(x)

                loss_model = y_model.pow(2).sum()
                loss_fmodel = y_fmodel.pow(2).sum()

                loss_model.backward()
                diffopt.step(loss_fmodel)
                opt.step()
                self.assertEqual(
                    len(list(model.parameters())),
                    len(list(fmodel.parameters()))
                )

            for p, fp in zip(model.parameters(), fmodel.parameters()):
                torch.testing.assert_allclose(p, fp, atol=1e-5, rtol=1e-1)

    @parameterized.expand(_test_param_sweep)
    def testGradientCorrectness(
        self, _, model_builder, opt_builder, kwargs=None
    ):
        kwargs = {} if kwargs is None else kwargs
        lr = .1
        model = model_builder(self)
        eps = 1e-3

        tests = 10
        count = 0
        threshold = .6  # proportion of tests that should pass

        for i in range(tests):
            xs = [torch.rand(10, 4) for _ in range(2)]

            def closure():
                cmodel = copy.deepcopy(model)
                opt = opt_builder(cmodel.parameters(), lr=lr, **kwargs)
                for x in xs[:-1]:
                    opt.zero_grad()
                    cmodel(x).pow(2).sum().backward()
                    opt.step()
                loss = cmodel(xs[-1]).pow(2).sum()
                return loss

            fd_grads = finite_difference(model, closure, eps)

            opt = opt_builder(model.parameters(), lr=lr, **kwargs)
            with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
                for x in xs[:-1]:
                    loss = fmodel(x).pow(2).sum()
                    diffopt.step(loss)
                loss = fmodel(xs[-1]).pow(2).sum()
                grads = torch.autograd.grad(
                    loss, fmodel.parameters(time=0), allow_unused=True
                )
                close = []
                for g, fg in zip(grads, fd_grads):
                    if g is None:
                        # trusting that the tensor shouldn't have been used...
                        close.append(True)
                    else:
                        self.assertFalse(
                            torch.any(torch.isnan(g)), "NaNs found in gradient."
                        )
                        close.append(torch.allclose(g, fg, 1e-1, 1e-1))
                if all(close):
                    count += 1
        self.assertTrue(
            count / tests >= threshold,
            msg="Proportion of successful finite gradient checks below {:.0f}% "
            "threshold ({:.0f}%).".format(threshold * 100, 100 * count / tests)
        )

    @parameterized.expand([(
        "simple_model_adam",
        lambda self: self._model,
        optim.Adam,
    )])
    def testDiffOptGroupedParamLearn(
        self, _, model_builder, opt_builder, kwargs=None
    ):
        kwargs = {} if kwargs is None else kwargs
        lr = .1
        left_lr = .2
        model = model_builder(self)
        full_parameters = list(model.parameters())
        half = len(full_parameters) // 2
        left_parameters = full_parameters[:half]
        right_parameters = full_parameters[half:]
        param_groups = [
            {
                'params': left_parameters,
                'lr': left_lr
            },
            {
                'params': right_parameters
            },
        ]
        opt = opt_builder(param_groups, lr=lr, **kwargs)

        override = {
            'lr':
                [
                    torch.tensor(.3, requires_grad=True)
                ],
            'betas':
                [
                    (
                        torch.tensor(0.9, requires_grad=True),
                        torch.tensor(0.999, requires_grad=True)
                    ),
                    (
                        torch.tensor(0.8, requires_grad=True),
                        torch.tensor(0.888, requires_grad=True)
                    )
                ]
        }
        meta_params = higher.utils.flatten(override)

        for i in range(1):
            fmodel = higher.patch.monkeypatch(model)
            diffopt = higher.optim.get_diff_optim(
                opt, model.parameters(), fmodel, override=override
            )
            for j in range(3):
                x = torch.rand(10, 4)
                y_fmodel = fmodel(x)
                loss_fmodel = y_fmodel.pow(2).sum()
                diffopt.step(loss_fmodel)
            param_sum = sum(p.sum() for p in fmodel.parameters())
            for g in torch.autograd.grad(param_sum, meta_params):
                self.assertTrue(
                    torch.isfinite(g).all().item(),
                    "Nan or Inf found in hyperparameter gradients."
                )

    @staticmethod
    def _approx_equal_params(params_1, params_2):
        params_1 = list(params_1)
        params_2 = list(params_2)
        if len(params_1) != len(params_2):
            return False
        for p1, p2 in zip(params_1, params_2):
            if not torch.allclose(p1, p2):
                return False
        return True

    @parameterized.expand([(
        "simple_model_adam",
        lambda self: self._model,
        optim.Adam,
    )])
    def testDiffOptCallback(
        self, _, model_builder, opt_builder, kwargs=None
    ):
        kwargs = {} if kwargs is None else kwargs
        lr = .1
        left_lr = .2
        model = model_builder(self)

        full_parameters = list(model.parameters())
        half = len(full_parameters) // 2
        left_parameters = full_parameters[:half]
        right_parameters = full_parameters[half:]
        param_groups = [
            {
                'params': left_parameters,
                'lr': left_lr
            },
            {
                'params': right_parameters
            },
        ]
        opt = opt_builder(param_groups, lr=lr, **kwargs)

        # We should have the following equalities/inequalities for the patched
        # models defined below at the end of training:
        # fmodel_0 != fmodel_1 != fmodel_2
        # fmodel_1 != fmodel_2
        # fmodel_2 == fmodel_3

        callback_1 = lambda all_grad: [g * .1 for g in all_grad]
        callback_2 = lambda all_grad: [g * .2 for g in all_grad]
        callback_3 = callback_2

        for i in range(1):
            fmodel_0 = higher.patch.monkeypatch(model)
            diffopt_0 = higher.optim.get_diff_optim(
                opt, model.parameters(), fmodel_0, grad_callback=None
            )

            fmodel_1 = higher.patch.monkeypatch(model)
            diffopt_1 = higher.optim.get_diff_optim(
                opt, model.parameters(), fmodel_1, grad_callback=callback_1
            )

            fmodel_2 = higher.patch.monkeypatch(model)
            diffopt_2 = higher.optim.get_diff_optim(
                opt, model.parameters(), fmodel_2, grad_callback=callback_2
            )

            fmodel_3 = higher.patch.monkeypatch(model)
            diffopt_3 = higher.optim.get_diff_optim(
                opt, model.parameters(), fmodel_3, grad_callback=None
            )

            for j in range(3):
                x = torch.rand(10, 4)
                diffopt_0.step(fmodel_0(x).pow(2).sum())
                diffopt_1.step(fmodel_1(x).pow(2).sum())
                diffopt_2.step(fmodel_2(x).pow(2).sum())
                diffopt_3.step(
                    fmodel_3(x).pow(2).sum(), grad_callback=callback_3
                )

            # Check that the conditions described at top of loop are satisfied
            self.assertFalse(
                self._approx_equal_params(
                    fmodel_0.parameters(), fmodel_1.parameters()
                )
            )
            self.assertFalse(
                self._approx_equal_params(
                    fmodel_0.parameters(), fmodel_2.parameters()
                )
            )
            self.assertFalse(
                self._approx_equal_params(
                    fmodel_1.parameters(), fmodel_2.parameters()
                )
            )
            self.assertTrue(
                self._approx_equal_params(
                    fmodel_2.parameters(), fmodel_3.parameters()
                )
            )

    @parameterized.expand([(
        "simple_model_adam",
        lambda self: self._model,
        optim.Adam,
    )])
    def testDiffOptGroupedParamLearnStepwise(
        self, _, model_builder, opt_builder, kwargs=None
    ):
        kwargs = {} if kwargs is None else kwargs
        lr = .1
        left_lr = .2
        model = model_builder(self)
        full_parameters = list(model.parameters())
        half = len(full_parameters) // 2
        left_parameters = full_parameters[:half]
        right_parameters = full_parameters[half:]
        param_groups = [
            {
                'params': left_parameters,
                'lr': left_lr
            },
            {
                'params': right_parameters
            },
        ]
        opt = opt_builder(param_groups, lr=lr, **kwargs)

        override = {
            'lr':
                [
                    torch.tensor(.3, requires_grad=True)
                ],
            'betas':
                [
                    (
                        torch.tensor(0.9, requires_grad=True),
                        torch.tensor(0.999, requires_grad=True)
                    ),
                    (
                        torch.tensor(0.8, requires_grad=True),
                        torch.tensor(0.888, requires_grad=True)
                    )
                ]
        }
        meta_params = higher.utils.flatten(override)

        for i in range(1):
            fmodel = higher.patch.monkeypatch(model)
            diffopt = higher.optim.get_diff_optim(
                opt, model.parameters(), fmodel, override=None
            )
            for j in range(3):
                x = torch.rand(10, 4)
                y_fmodel = fmodel(x)
                loss_fmodel = y_fmodel.pow(2).sum()
                diffopt.step(loss_fmodel, override=override)
            param_sum = sum(p.sum() for p in fmodel.parameters())
            for g in torch.autograd.grad(param_sum, meta_params):
                self.assertTrue(
                    torch.isfinite(g).all().item(),
                    "Nan or Inf found in hyperparameter gradients."
                )

    def testFrozenParameters(self):
        """Check if diffopts robuts to frozen parameters.

        Thanks to github user @seanie12 for providing the minimum working
        example for this unit test.
        """
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(30, 50)
                self.fc2 = nn.Linear(50, 1)
                # freeze first FC layer
                for param in self.fc1.parameters():
                    param.requires_grad = False

            def forward(self, x):
                hidden = self.fc1(x)
                logits = self.fc2(hidden).squeeze(1)
                return logits

        # random input and labels for debugging
        inputs = torch.randn(16, 30)
        ones = torch.ones(8)
        zeros = torch.zeros(8)
        labels = torch.cat([ones, zeros], dim=0)

        net = Net()

        param = filter(lambda x: x.requires_grad, net.parameters())
        inner_opt = torch.optim.SGD(param, lr=1e-1)
        loss_func = nn.BCEWithLogitsLoss()

        with higher.innerloop_ctx(net, inner_opt) as (fnet, diffopt):
            logits = fnet(inputs)
            loss = loss_func(logits, labels)
            diffopt.step(loss)
            zipped = list(zip(net.parameters(), fnet.parameters()))
            self.assertTrue(torch.equal(*zipped[0]))
            self.assertTrue(torch.equal(*zipped[1]))
            self.assertFalse(torch.equal(*zipped[2]))
            self.assertFalse(torch.equal(*zipped[3]))

    def testGetApplyRoundTrip(self):
        kwargs = {}
        lr = .1
        left_lr = .2
        model = self._model
        full_parameters = list(model.parameters())
        half = len(full_parameters) // 2
        left_parameters = full_parameters[:half]
        right_parameters = full_parameters[half:]
        param_groups = [
            {
                'params': left_parameters,
                'lr': left_lr
            },
            {
                'params': right_parameters
            },
        ]
        opt = optim.Adam(param_groups, lr=lr, **kwargs)

        override = higher.optim.get_trainable_opt_params(opt)

        def assert_closure(target: torch.Tensor):
            self.assertTrue(torch.is_tensor(target) and target.requires_grad)

        # Check that all items in override are structures containing
        # differentiable tensors requiring gradient
        for hp in override:
            higher.utils._recursive_map(override[hp], assert_closure)

        param_groups = [
            {
                'params': left_parameters,
                'lr': left_lr + 1
            },
            {
                'params': right_parameters
            },
        ]

        # Create new opt with slightly different parameter group hyperparameter
        # values, to simulate divergence between original hyperparameters
        new_opt = optim.Adam(param_groups, lr=lr+5, **kwargs)

        # Overwrite with the "learned" hyperparameters
        higher.optim.apply_trainable_opt_params(new_opt, override)

        # Check that values match
        # TODO(egrefen): would be good to do a structure matching test, or an
        # extrinsic eval whereby we check that both opts are functionally
        # equivalent.
        old_flattened = higher.utils.flatten(opt.param_groups)
        new_flattened = higher.utils.flatten(new_opt.param_groups)
        self.assertEqual(len(old_flattened), len(new_flattened))
        zipped = zip(old_flattened, new_flattened)
        for old, new in zipped:
            self.assertEqual(type(old), type(new))
            if torch.is_tensor(old):
                torch.testing.assert_allclose(old, new)
            elif isinstance(old, float) or isinstance(old, int):
                self.assertAlmostEqual(old, new)
            else:
                self.assertEqual(old, new)



if __name__ == '__main__':
    unittest.main()
