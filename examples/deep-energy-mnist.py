#!/usr/bin/env python3
#
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

"""
This example shows how to use higher to implement unrolled energy
networks for MNIST classification.

The relevant papers are:
  + The original Structured Predection Energy Networks (SPEN) paper
    https://arxiv.org/abs/1511.06350
  + End-to-End Learning for SPENs: https://arxiv.org/abs/1703.05667
  + Input Convex Neural Networks: https://arxiv.org/abs/1609.07152

In this example we show the simple case of using an unrolled deep
energy function for single-label MNIST classification that can be used
as a starting point for using SPENs or ICNNs to model output spaces
with more structure.
In the single-label classification setting, given a feature x and
proposed label y, deep energy networks learn a real-valued neural network
energy function E_\theta(x, y) that models the conditional likelihood

    P(y|x) \propto exp(-E_\theta(x, y))

Predictions are made by (approximately) solving the optimization problem

    \hat y = argmin_y E_\theta(x, y)

with a fixed number of gradient descent steps.
In SPENs this is a non-convex optimization problem but can be more expressive,
and in ICNNs this is a convex optimization problem over the output space y,
but still non-convex over the input and parameter space.
Given data (x, y), learning can be done by *unrolling* gradient descent
and optimizing a loss function such as

    L(y, \hat y) = || y - \hat y ||_2^2

with respect to the parameters of the energy function \theta.

This setting requires two levels of optimization:
   1. The outer level of the energy function's parameter optimization.
      We use PyTorch's vanilla Adam optimizer for this.
   2. The inner level of the energy function's input space optimization.
       We use higher's differentiable optimizer for this.

This code is a modified version of the PyTorch MNIST example:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import argparse
import typing

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import higher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', type=str, default='SPEN',
        choices=['SPEN', 'ICNN'])
    args = parser.parse_args()

    # Set up the MNIST data loaders.
    train_loader = DataLoader(
        datasets.MNIST('/tmp/mnist-data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=128, shuffle=True)

    test_loader = DataLoader(
        datasets.MNIST(
            '/tmp/mnist-data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=128, shuffle=True)

    # Initialize an energy model and optimizer for the energy function.
    device = torch.device("cuda")
    Enet = EnergyNet().to(device)

    if args.mode == 'ICNN':
        # In the ICNN, the fce2 weights are required to be non-negative
        # to maintain convexity of the energy function.
        # Here we just project negative weights back onto the
        # non-negative orthant.
        fce2W = Enet.state_dict()['fce2.weight']
        fce2W[fce2W < 0] = 0.

    model = UnrollEnergy(Enet).to(device)
    optimizer = optim.Adam(Enet.parameters(), lr=3e-4)

    for epoch in range(10):
        train(model, device, train_loader, optimizer, args.mode, epoch)
        test(model, device, test_loader)


class EnergyNet(nn.Module):
    """An energy function E(x, y) for visual single-label classification.

    An energy function takes an image x and label y
    as the input and outputs a real number.
    We use a LeNet-style architecture to extract an embedding from x
    that is that concatenated with y and passed through a single hidden
    layer fully-connected network.

    Args:
        n_fc_hidden (int): The number of hidden units the
          fully-connected layers have.
        n_cls (int): The number of classes.
    """
    def __init__(self, n_fc_hidden: int = 500, n_cls: int = 10):
        super(EnergyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, n_fc_hidden)
        self.fc2 = nn.Linear(n_fc_hidden, n_fc_hidden)

        self.fce1 = nn.Linear(n_fc_hidden+n_cls, n_fc_hidden)
        self.fce2 = nn.Linear(n_fc_hidden, 1)


    def forward(self, x, y):
        # First extract an embedding z from the visual input x.
        #
        # We use softplus activations so our model has
        # (generally) non-zero second-order derivatives.
        z = F.softplus(self.conv1(x))
        z = F.max_pool2d(z, 2, 2)
        z = F.softplus(self.conv2(z))
        z = F.max_pool2d(z, 2, 2)
        z = z.view(-1, 4*4*50)
        z = F.softplus(self.fc1(z))
        z = self.fc2(z)

        # Next combine that embedding with the proposed label y
        # and pass that through a single hidden-layer to predict
        # the energy function value.
        v = torch.cat((z, y), dim=1)
        v = F.softplus(self.fce1(v))
        E = self.fce2(v).squeeze()
        return E


class UnrollEnergy(nn.Module):
    """A deep energy module that unrolls an optimizer over the energy function.

    This module takes a grayscale 28x28 image x as the input and
    outputs a class prediction by (approximately) solving the
    optimization problem

        \hat y = argmin_y E_\theta(x, y)

    with a fixed number of gradient steps.

    Args:
        Enet: The energy network.
        n_cls (int): The number of classes.
        n_inner_iter (int): The number of optimization steps to take.
    """
    def __init__(self, Enet: EnergyNet, n_cls: int = 10, n_inner_iter: int = 5):
        super(UnrollEnergy, self).__init__()
        self.Enet = Enet
        self.n_cls = n_cls
        self.n_inner_iter = n_inner_iter

    def forward(self, x):
        assert x.ndimension() == 4
        nbatch = x.size(0)

        # Make an initial guess of the labels.
        # For more sophisticated tasks this could also be learned.
        y = torch.zeros(nbatch, self.n_cls, device=x.device, requires_grad=True)

        # Define a differentiable optimizer to update the label with.
        inner_opt = higher.get_diff_optim(
            torch.optim.SGD([y], lr=1e-1),
            [y], device=x.device
        )

        # Take a few gradient steps to find the labels that
        # optimize the energy function.
        for _ in range(self.n_inner_iter):
            E = self.Enet(x, y)
            y, = inner_opt.step(E.sum(), params=[y])

        return y


def train(model, device, train_loader, optimizer, epoch, mode, log_interval=10):
    """The training loop that optimizes the likelihood of a differentable model.

    Our model in this example internally unrolls gradient descent
    over an energy function in a differentiable way using higher
    and we can use the outputs of this model just as we use the
    outputs of any other differentiable model to optimize
    a loss function by taking gradient steps.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if mode == 'ICNN':
            # In the ICNN, the fce2 weights are required to be non-negative
            # to maintain convexity of the energy function.
            # Here we just project negative weights back onto the
            # non-negative orthant.
            fce2W = model.Enet.state_dict()['fce2.weight']
            fce2W[fce2W < 0] = 0.

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    """Evaluate the performance on the test dataset."""
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()
