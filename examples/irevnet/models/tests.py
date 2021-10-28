import unittest

import torch
from torch import nn

from .model_utils import psi


class psi_legacy(nn.Module):  # Original implementation of class model_utils.psi
    def __init__(self, block_size):
        super(psi_legacy, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def inverse(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.contiguous().view(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.contiguous().view(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).contiguous().view(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()


class TPsi(unittest.TestCase):
    def test_forward(self):
        block_size = 3
        t = torch.randn(64, 5, 192, 192, dtype=torch.float32)
        psi_new_result = psi(block_size).forward(t.clone())
        psi_old_result = psi_legacy(block_size).forward(t.clone())
        self.assertTrue((psi_new_result == psi_old_result).all().item())
        self.assertTrue(psi_new_result.is_contiguous())

    def test_inverse(self):
        block_size = 3
        t = torch.randn(64, 45, 64, 64, dtype=torch.float32)
        psi_new_result = psi(block_size).inverse(t.clone())
        psi_old_result = psi_legacy(block_size).inverse(t.clone())
        self.assertTrue((psi_new_result == psi_old_result).all().item())
        self.assertTrue(psi_new_result.is_contiguous())
