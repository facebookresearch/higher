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

"""Unit tests for higher top level functions."""
import unittest
import copy
from collections import OrderedDict

import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F

import higher


class _ReferenceNet(nn.Module):
    def __init__(self, features, fc):
        super().__init__()
        self.features = features
        self.add_module('fc', fc)

    def batch_norm(
        self,
        inputs,
        weight=None,
        bias=None,
        running_mean=None,
        running_var=None,
        training=True,
        eps=1e-5,
        momentum=0.1
    ):
        running_mean = torch.zeros(np.prod(np.array(inputs.data.size()[1])))
        running_var = torch.ones(np.prod(np.array(inputs.data.size()[1])))
        return F.batch_norm(
            inputs, running_mean, running_var, weight, bias, training, momentum,
            eps
        )

    def maxpool(self, input, kernel_size, stride=None):
        return F.max_pool2d(input, kernel_size, stride)

    def forward(self, x, params=None):
        if params is None:
            x = self.features(x).view(x.size(0), 64)
            x = self.fc(x)
        else:

            x = F.conv2d(
                x, params['features.conv1.weight'],
                params['features.conv1.bias']
            )
            x = self.batch_norm(
                x,
                weight=params['features.bn1.weight'],
                bias=params['features.bn1.bias'],
                momentum=1
            )
            x = F.relu(x)
            x = self.maxpool(x, kernel_size=2, stride=2)
            x = F.conv2d(
                x, params['features.conv2.weight'],
                params['features.conv2.bias']
            )
            x = self.batch_norm(
                x,
                weight=params['features.bn2.weight'],
                bias=params['features.bn2.bias'],
                momentum=1
            )
            x = F.relu(x)
            x = self.maxpool(x, kernel_size=2, stride=2)
            x = F.conv2d(
                x, params['features.conv3.weight'],
                params['features.conv3.bias']
            )
            x = self.batch_norm(
                x,
                weight=params['features.bn3.weight'],
                bias=params['features.bn3.bias'],
                momentum=1
            )
            x = F.relu(x)
            x = self.maxpool(x, kernel_size=2, stride=2)
            x = x.view(x.size(0), 64)
            x = F.linear(x, params['fc.weight'], params['fc.bias'])
        return x

    def get_fast_weights(self):
        fast_weights = OrderedDict(
            (name, param) for (name, param) in self.named_parameters()
        )
        return fast_weights


class _TargetNet(nn.Module):
    def __init__(self, features, fc):
        super().__init__()
        self.features = features
        self.add_module('fc', fc)

    def forward(self, x):
        x = self.features(x).view(x.size(0), 64)
        x = self.fc(x)
        return x


class TestCorrectness(unittest.TestCase):
    """Test case for package-level functions for correctness."""

    def setUp(self):
        self.num_in_channels = num_in_channels = 3
        self.num_classes = num_classes = 5
        self.batch_size = 7
        self.in_h = self.in_w = 28

        features = nn.Sequential(
            OrderedDict(
                [
                    ('conv1', nn.Conv2d(num_in_channels, 64, 3)),
                    ('bn1', nn.BatchNorm2d(64, momentum=1, affine=True)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('pool1', nn.MaxPool2d(2, 2)),
                    ('conv2', nn.Conv2d(64, 64, 3)),
                    ('bn2', nn.BatchNorm2d(64, momentum=1, affine=True)),
                    ('relu2', nn.ReLU(inplace=True)),
                    ('pool2', nn.MaxPool2d(2, 2)),
                    ('conv3', nn.Conv2d(64, 64, 3)),
                    ('bn3', nn.BatchNorm2d(64, momentum=1, affine=True)),
                    ('relu3', nn.ReLU(inplace=True)),
                    ('pool3', nn.MaxPool2d(2, 2))
                ]
            )
        )
        fc = nn.Linear(64, num_classes)
        self.target_net = _TargetNet(features, fc)
        self.reference_net = _ReferenceNet(
            copy.deepcopy(features), copy.deepcopy(fc)
        )
        self.lr = .01
        self.opt = optim.SGD(self.target_net.parameters(), lr=self.lr)

    def testSameInitialWeightsPrePatch(self):
        """Check that reference and unpatched target net have equal weights.

        This is mostly a sanity check for the purpose of the other unit tests.
        """
        ref_params = list(self.reference_net.named_parameters())
        target_params = list(self.target_net.named_parameters())
        self.assertEqual(
            len(ref_params),
            len(target_params),
            msg=(
                "Length mismatched between reference net parameter count "
                "({}) and target ({}).".format(
                    len(ref_params), len(target_params)
                )
            )
        )
        for ref, target in zip(ref_params, target_params):
            ref_name, ref_p = ref
            target_name, target_p = target
            self.assertEqual(
                ref_name,
                target_name,
                msg="Name mismatch or parameter misalignment ('{}' vs '{}')".
                format(ref_name, target_name)
            )
            self.assertTrue(
                torch.equal(ref_p, target_p),
                msg="Parameter value inequality for {}".format(ref_name)
            )

    def testSameInitialWeightsPostPatch(self):
        """Verify fast weight alignment/equality after monkey patching."""
        ref_named_params = list(self.reference_net.get_fast_weights().items())
        ref_params = [p for (_, p) in ref_named_params]
        with higher.innerloop_ctx(self.target_net, self.opt) as (fnet, _):
            target_named_params = list(fnet.named_parameters())
            target_params = fnet.parameters()
            self.assertEqual(
                len(ref_named_params),
                len(target_named_params),
                msg=(
                    "Length mismatched between reference net parameter count "
                    "({}) and target ({}).".format(
                        len(ref_named_params), len(target_named_params)
                    )
                )
            )
            for ref, target in zip(ref_named_params, target_named_params):
                ref_name, ref_p = ref
                target_name, target_p = target
                self.assertEqual(
                    ref_name,
                    target_name,
                    msg="Name mismatch or parameter misalignment ('{}' vs '{}')"
                    .format(ref_name, target_name)
                )
                self.assertTrue(
                    torch.equal(ref_p, target_p),
                    msg="Parameter value inequality for {}".format(ref_name)
                )
            zipped = zip(ref_params, target_params)
            for i, (ref_p, target_p) in enumerate(zipped):
                self.assertTrue(
                    torch.equal(ref_p, target_p),
                    msg="Parameter misalignment in position {}.".format(i)
                )

    def testRandomForwards(self):
        """Test reference and patched net forward equivalence.

        Test if, given rand fast weights, patched net and reference forwards
        match up given random inputs.
        """
        with higher.innerloop_ctx(self.target_net, self.opt) as (fnet, _):
            for i in range(10):
                fast_named_weights = OrderedDict(
                    (name, torch.rand(p.shape, requires_grad=True))
                    for name, p in self.reference_net.named_parameters()
                )
                fast_weights = [p for _, p in fast_named_weights.items()]
                inputs = torch.rand(
                    self.batch_size, self.num_in_channels, self.in_h, self.in_w
                )
                self.assertTrue(
                    torch.equal(
                        self.reference_net(inputs, params=fast_named_weights),
                        fnet(inputs, params=fast_weights)
                    )
                )

    def testUnrollEqualityForward(self):
        """Check if unrolled patched and reference nets produce same meta loss.
        """
        for test_it in range(5):
            with higher.innerloop_ctx(self.target_net,
                                      self.opt) as (fnet, diffopt):
                ref_out, target_out = self._joint_inner_loop(
                    fnet, diffopt=diffopt, num_steps=10
                )
                ref_meta_loss = ref_out[0]
                ref_fast_weights = ref_out[1]
                ref_train_losses = ref_out[2]
                ref_train_grads = ref_out[3]
                target_meta_loss = target_out[0]
                target_fast_weights = target_out[1]
                target_train_losses = target_out[2]
                target_train_grads = target_out[3]

                # Check final losses match
                self.assertTrue(
                    torch.allclose(ref_meta_loss, target_meta_loss),
                    msg=(
                        "Ref ({}) and target ({}) metaloss differed on test_it "
                        "{} (mse {})"
                    ).format(
                        ref_meta_loss.item(), target_meta_loss.item(), test_it,
                        (ref_meta_loss - target_meta_loss).pow(2).item()
                    )
                )

                # Check that training losses align
                for rl, tl in zip(ref_train_losses, target_train_losses):
                    torch.testing.assert_allclose(rl, tl)

                # Check that fast weights align
                for rw, tw in zip(ref_fast_weights, target_fast_weights):
                    torch.testing.assert_allclose(rw, tw)

                # Check that grads align
                for rgs, tgs in zip(ref_train_grads, target_train_grads):
                    for rg, tg in zip(rgs, tgs):
                        torch.testing.assert_allclose(rg, tg)

    def testUnrollEqualityBackward(self):
        """Check if metagrads match for target/ref net."""
        for test_it in range(5):
            with higher.innerloop_ctx(self.target_net,
                                      self.opt) as (fnet, diffopt):
                ref_out, target_out = self._joint_inner_loop(
                    fnet, diffopt=diffopt, num_steps=10
                )
                ref_meta_loss = ref_out[0]
                target_meta_loss = target_out[0]

                ref_metagrads = torch.autograd.grad(
                    ref_meta_loss, self.reference_net.parameters()
                )
                target_metagrads = torch.autograd.grad(
                    target_meta_loss, fnet.parameters(time=0)
                )

                # Check that metagrads align
                for rg, tg in zip(ref_metagrads, target_metagrads):
                    torch.testing.assert_allclose(rg, tg)


    def testUnrollEqualityBackwardManualUnroll(self):
        """Check if metagrads match for target/ref net.

        A differentiable optimizer is not used (manual inner loop SGD).
        """
        for test_it in range(5):
            with higher.innerloop_ctx(self.target_net,
                                      self.opt) as (fnet, diffopt):
                ref_out, target_out = self._joint_inner_loop(
                    fnet, diffopt=None, num_steps=10
                )
                ref_meta_loss = ref_out[0]
                target_meta_loss = target_out[0]

                ref_metagrads = torch.autograd.grad(
                    ref_meta_loss, self.reference_net.parameters()
                )
                target_metagrads = torch.autograd.grad(
                    target_meta_loss, fnet.parameters(time=0)
                )

                # Check that metagrads align
                for rg, tg in zip(ref_metagrads, target_metagrads):
                    torch.testing.assert_allclose(rg, tg)


    def _joint_inner_loop(self, fnet, diffopt=None, num_steps=1):
        ref_fast_weights = self.reference_net.get_fast_weights()
        target_fast_weights = None
        if diffopt is None:
            # If diffopt not provided we manually will update these
            target_fast_weights = list(fnet.parameters())

        # Things we want to track
        ref_train_losses = []
        ref_train_grads = []
        target_train_losses = []
        target_train_grads = []

        for _ in range(num_steps):
            inputs = torch.rand(
                self.batch_size, self.num_in_channels, self.in_h,
                self.in_w
            )
            labels = torch.rand(self.batch_size, self.num_classes)

            # Do inner loop step for reference net
            ref_preds = self.reference_net(inputs, params=ref_fast_weights)
            ref_loss = F.mse_loss(ref_preds, labels)
            ref_grads = torch.autograd.grad(
                ref_loss, ref_fast_weights.values(), create_graph=True
            )

            ref_train_losses.append(ref_loss)
            ref_train_grads.append(ref_grads)

            ref_fast_weights = OrderedDict(
                (name, param - self.lr * grad)
                for ((name, param),
                     grad) in zip(ref_fast_weights.items(), ref_grads)
            )

            # Do inner loop step for target net
            if diffopt is None:
                target_preds = fnet(inputs, params=target_fast_weights)
            else:
                target_preds = fnet(inputs)
            target_loss = F.mse_loss(target_preds, labels)

            if diffopt is None:
                target_grads = torch.autograd.grad(
                    target_loss, target_fast_weights, create_graph=True
                )
                target_fast_weights = [
                    w - (self.lr * g)
                    for w, g in zip(target_fast_weights, target_grads)
                ]
            else:
                target_grads = torch.autograd.grad(
                    target_loss, list(fnet.parameters()), create_graph=True
                )
                diffopt.step(target_loss)

            target_train_losses.append(target_loss)
            target_train_grads.append(target_grads)


        # metaval
        inputs = torch.rand(
            self.batch_size, self.num_in_channels, self.in_h,
            self.in_w
        )
        labels = torch.rand(self.batch_size, self.num_classes)
        ref_preds = self.reference_net(inputs, params=ref_fast_weights)
        ref_meta_loss = F.mse_loss(ref_preds, labels)

        if diffopt is None:
            target_preds = fnet(inputs, params=target_fast_weights)
        else:
            target_preds = fnet(inputs)
        target_meta_loss = F.mse_loss(target_preds, labels)


        ref_fast_weights = ref_fast_weights.values()
        target_fast_weights = fnet.parameters()
        if diffopt is None:
            target_fast_weights = fnet.parameters()

        packed = (
            (
                ref_meta_loss, ref_fast_weights, ref_train_losses,
                ref_train_grads
            ),
            (
                target_meta_loss, target_fast_weights, target_train_losses,
                target_train_grads
            )
        )
        return packed
