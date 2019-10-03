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

"""Unit tests for higher.patch."""
import unittest
from parameterized import parameterized

import torch
from torch import nn

import higher

_test_sweep = [
    ("simple_model", lambda self: self._model),
    ("share_weight_model", lambda self: self._shared_param_model),
    ("share_weight_seq_model", lambda self: self._shared_param_model),
    ("partially_used_model", lambda self: self._partially_used_model),
    ("batchnorm_mlp", lambda self: self._batchnorm_mlp),
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


class TestPatch(unittest.TestCase):
    """Test case for the patch module."""

    def setUp(self):
        self.lr = .1
        self.linear = torch.nn.Linear(4, 3)
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
        self._batchnorm_mlp = torch.nn.Sequential(
            torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.BatchNorm1d(3),
            torch.nn.Linear(3, 4), torch.nn.Sigmoid(), torch.nn.BatchNorm1d(4),
            torch.nn.Linear(4, 2)
        )

    def testMakeFunctional(self):
        fmodel = higher.patch.make_functional(self.linear)
        self.assertTrue(callable(fmodel))

    @parameterized.expand(_test_sweep)
    def testGet(self, _, model_builder):
        model = model_builder(self)
        params = higher.utils.get_func_params(model)
        fmodel = higher.patch.monkeypatch(model)
        zipped = zip(model.parameters(), params, fmodel.parameters())
        for true_param, got_param, got_param2 in zipped:
            torch.testing.assert_allclose(true_param, got_param)

    @parameterized.expand(_test_sweep)
    def testForward(self, _, model_builder):
        model = model_builder(self)
        fmodel = higher.patch.monkeypatch(model)
        for _ in range(10):
            inputs = torch.rand(8, 4)
            ref_outputs = model(inputs)
            outputs = fmodel(inputs)
            torch.testing.assert_allclose(outputs, ref_outputs)

    @parameterized.expand(_test_sweep)
    def testBackward(self, _, model_builder):
        model = model_builder(self)
        fmodel = higher.patch.monkeypatch(model)
        for _ in range(10):
            inputs = torch.rand(8, 4)
            ref_loss = model(inputs).pow(2).sum()
            loss = fmodel(inputs).pow(2).sum()
            ref_loss.backward()
            loss.backward()
            for true_param, got_param in zip(
                model.parameters(), fmodel.parameters()
            ):
                if true_param.grad is None:
                    self.assertIsNone(got_param.grad)
                else:
                    self.assertIsNotNone(got_param.grad)
                    torch.testing.assert_allclose(
                        true_param.grad, got_param.grad
                    )
        model.zero_grad()

    @parameterized.expand(_test_sweep)
    def testUnrollSGD(self, _, model_builder):
        model = model_builder(self)
        fmodel = higher.patch.monkeypatch(model)
        for _ in range(10):
            inputs = torch.rand(8, 4)
            loss = fmodel(inputs).pow(2).sum()
            grads = torch.autograd.grad(
                loss, fmodel.parameters(), allow_unused=True, create_graph=True
            )
            fmodel.fast_params = [
                p - self.lr * g if g is not None else p
                for p, g in zip(fmodel.parameters(), grads)
            ]
        param_sum = sum(p.sum() for p in fmodel.parameters())
        final_grads = torch.autograd.grad(param_sum, fmodel.parameters(time=0))
        for grad in final_grads:
            self.assertIsNotNone(grad)

    @parameterized.expand(_test_sweep)
    def testCtxManager(self, _, model_builder):
        model = model_builder(self)
        opt = torch.optim.SGD(model.parameters(), lr=self.lr)

        with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
            for _ in range(10):
                inputs = torch.rand(8, 4)
                loss = fmodel(inputs).pow(2).sum()
                diffopt.step(loss)
            param_sum = sum(p.sum() for p in fmodel.parameters())
            final_grads = torch.autograd.grad(
                param_sum, fmodel.parameters(time=0)
            )

        for grad in final_grads:
            self.assertIsNotNone(grad)

    def testRNN(self):
        num_layers = 2
        hidden_size = 20
        num_feats = 10

        batch_size = 3
        seq_length = 5

        for _ in range(5):

            rnn = torch.nn.LSTM(num_feats, hidden_size, num_layers)
            frnn = higher.patch.monkeypatch(rnn)

            for _ in range(10):

                input = torch.randn(seq_length, batch_size, num_feats)
                h0 = torch.randn(num_layers, batch_size, hidden_size)
                c0 = torch.randn(num_layers, batch_size, hidden_size)

                output, (hn, cn) = rnn(input, (h0, c0))
                foutput, (fhn, fcn) = frnn(input, (h0, c0))

                torch.testing.assert_allclose(output, foutput),
                torch.testing.assert_allclose(hn, fhn),
                torch.testing.assert_allclose(cn, fcn)

    @parameterized.expand(_test_sweep)
    def testMiniMAML(self, _, model_builder):
        model = model_builder(self)
        opt = torch.optim.SGD(model.parameters(), lr=self.lr)

        ctx_opts = {"copy_initial_weights": False}
        with higher.innerloop_ctx(model, opt, **ctx_opts) as (fmodel, diffopt):
            for _ in range(10):
                inputs = torch.rand(8, 4)
                loss = fmodel(inputs).pow(2).sum()
                diffopt.step(loss)
            param_sum = sum(p.sum() for p in fmodel.parameters())
            final_grads = torch.autograd.grad(
                param_sum, fmodel.parameters(time=0), retain_graph=True
            )
            param_sum.backward()
            for p, g in zip(model.parameters(), final_grads):
                torch.testing.assert_allclose(p.grad, g)

    @parameterized.expand(_test_sweep)
    def testMiniMAMLTestTime(self, _, model_builder):
        model = model_builder(self)
        opt = torch.optim.SGD(model.parameters(), lr=self.lr)

        ctx_opts = {"copy_initial_weights": False, "track_higher_grads": False}
        with higher.innerloop_ctx(model, opt, **ctx_opts) as (fmodel, diffopt):
            for _ in range(10):
                inputs = torch.rand(8, 4)
                loss = fmodel(inputs).pow(2).sum()
                diffopt.step(loss)
            param_sum = sum(p.sum() for p in fmodel.parameters())
            final_grads = torch.autograd.grad(
                param_sum, fmodel.parameters(time=0), allow_unused=True
            )
            param_sum.backward()
            for p, g in zip(model.parameters(), final_grads):
                self.assertIsNone(p.grad)
                self.assertIsNone(g)


if __name__ == '__main__':
    unittest.main()
