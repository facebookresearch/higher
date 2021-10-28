# Code for "i-RevNet: Deep Invertible Networks", ICLR 2018
# Author: Joern-Henrik Jacobsen, 2018
#
# Modified from Pytorch examples code.
# Original license shown below.
# =============================================================================
# BSD 3-Clause License
#
# Copyright (c) 2017, 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse

from models.utils_cifar import train, test, std, mean, get_hms
from models.iRevNet import iRevNet


parser = argparse.ArgumentParser(description='Train i-RevNet/RevNet on Cifar')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='i-revnet', type=str, help='model type')
parser.add_argument('--batch', default=128, type=int, help='batch size')
parser.add_argument('--init_ds', default=0, type=int, help='initial downsampling')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--nBlocks', nargs='+', type=int)
parser.add_argument('--nStrides', nargs='+', type=int)
parser.add_argument('--nChannels', nargs='+', type=int)
parser.add_argument('--bottleneck_mult', default=4, type=int, help='bottleneck multiplier')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')


def main():
    args = parser.parse_args()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[args.dataset], std[args.dataset]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean[args.dataset], std[args.dataset]),
    ])

    if(args.dataset == 'cifar10'):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        nClasses = 10
        in_shape = [3, 32, 32]
    elif(args.dataset == 'cifar100'):
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
        nClasses = 100
        in_shape = [3, 32, 32]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    def get_model(args):
        if (args.model == 'i-revnet'):
            model = iRevNet(nBlocks=args.nBlocks, nStrides=args.nStrides,
                            nChannels=args.nChannels, nClasses=nClasses,
                            init_ds=args.init_ds, dropout_rate=0.1, affineBN=True,
                            in_shape=in_shape, mult=args.bottleneck_mult)
            fname = 'i-revnet-'+str(sum(args.nBlocks)+1)
        elif (args.model == 'revnet'):
            raise NotImplementedError
        else:
            print('Choose i-revnet or revnet')
            sys.exit(0)
        return model, fname

    model, fname = get_model(args)

    use_cuda = True
    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=(0,))  # range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
            model = checkpoint['model']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        test(model, testloader, testset, start_epoch, use_cuda, best_acc, args.dataset, fname)
        return

    print('|  Train Epochs: ' + str(args.epochs))
    print('|  Initial Learning Rate: ' + str(args.lr))

    elapsed_time = 0
    best_acc = 0.
    for epoch in range(1, 1+args.epochs):
        start_time = time.time()

        train(model, trainloader, trainset, epoch, args.epochs, args.batch, args.lr, use_cuda, in_shape)
        best_acc = test(model, testloader, testset, epoch, use_cuda, best_acc, args.dataset, fname)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    print('Testing model')
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))


if __name__ == '__main__':
    main()
