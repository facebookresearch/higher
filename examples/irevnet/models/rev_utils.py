'''
Works similar to torch.utils.checkpoint, but saves more video memory than it does, but is slower and more restricted
工作原理
forward: Only save output
backward:  Read saved output, restore each layer input, enable grad and run forward again, then backward.

哪里耗时？checkpoint 需要在反传的时候额外计算一次正向传播，而我的则需要在额外计算二次正向传播。
'''

import torch
import torch.nn as nn
from torch.utils.checkpoint import get_device_states, set_device_states, check_backward_validity


class RevSequentialBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rev_block_stack, preserve_rng_state, *inputs):
        '''
        :param ctx:                 context, like self
        :param rev_block_stack:     Module with multiple reversible blocks stacked, such as RevSequential
        :param preserve_rng_state:  Whether to save the random number state, can be used to reproduce the random number
                                    only torch random numbers, numpy does not include
        :param inputs:              Multiple tensor, requires that at least one tensor requires_grad is True,
                                    otherwise this function will not calculate backpropagation,
                                     which is a limitation from torch.autograd.Function
        :return:
        '''
        # Warn when requires_grad of all inputs tensor is not True
        check_backward_validity(inputs)
        # Make sure the input is a list of modules and supports invert operations
        assert isinstance(rev_block_stack, nn.ModuleList)
        assert hasattr(rev_block_stack, 'inverse') and callable(rev_block_stack.inverse)

        ctx.rev_block_stack = rev_block_stack
        ctx.preserve_rng_state = preserve_rng_state

        # rng state save
        # Note that the state should be saved and restored layer by layer
        if preserve_rng_state:
            ctx.status_stack = []
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.cuda_status_stack = []

            # Since the execution order of the modules is reversed when the back pass is required,
            # each sub-module needs to save the random number state separately.
            outputs = inputs
            for m in ctx.rev_block_stack:
                fwd_cpu_state = torch.get_rng_state()
                ctx.status_stack.append(fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    fwd_gpu_devices, fwd_gpu_states = get_device_states(*outputs)
                    ctx.cuda_status_stack.append([fwd_gpu_devices, fwd_gpu_states])
                # Set torch.no_grad because don't save intermediate variables
                with torch.no_grad():
                    outputs = m(*outputs)

        else:
            # If you don't need to save the random number state, you can run the entire module directly to get the output
            with torch.no_grad():
                outputs = rev_block_stack(*inputs)

        # Save output for backward function
        ctx.save_for_backward(*outputs)
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        '''
        :param ctx: context, like self
        :param grad_output: the last module backward output
        :return: grad output, require number of outputs is the number of forward parameters -1, because ctx is not included
        '''

        # Get output that saved by forward function
        bak_outputs = ctx.saved_tensors
        with torch.no_grad():

            # Start from the last module
            for m in list(ctx.rev_block_stack)[::-1]:

                if ctx.preserve_rng_state:
                    # Restore rng state
                    rng_devices = []
                    if ctx.had_cuda_in_fwd:
                        fwd_gpu_devices, fwd_gpu_states = ctx.cuda_status_stack.pop(-1)
                        rng_devices = fwd_gpu_devices

                    fwd_cpu_state = ctx.status_stack.pop(-1)

                    with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
                        torch.set_rng_state(fwd_cpu_state)
                        if ctx.had_cuda_in_fwd:
                            set_device_states(fwd_gpu_devices, fwd_gpu_states)
                        # Restore input from output
                        inputs = m.inverse(*bak_outputs)
                    # Detach variables from graph
                    # Fix some problem in pytorch1.6
                    inputs = [t.detach().clone() for t in inputs]

                    # You need to set requires_grad to True to differentiate the input.
                    # The derivative is the input of the next backpass function.
                    # This is how grad_output comes.
                    for inp in inputs:
                        inp.requires_grad = True
                    # run backward for each sub-module
                    with torch.enable_grad():
                        # Restore rng state again
                        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
                            torch.set_rng_state(fwd_cpu_state)
                            if ctx.had_cuda_in_fwd:
                                set_device_states(fwd_gpu_devices, fwd_gpu_states)
                            outputs = m(*inputs)

                        if isinstance(outputs, torch.Tensor):
                            outputs = (outputs,)
                        torch.autograd.backward(outputs, grad_output)

                        grad_output = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                                      for inp in inputs)
                        bak_outputs = inputs

                else:
                    # Don't save rng state
                    # Restore input from output
                    inputs = m.inverse(*bak_outputs)
                    # Detach variables from graph
                    # Fix some problem in pytorch1.6
                    inputs = [t.detach().clone() for t in inputs]
                    for inp in inputs:
                        inp.requires_grad = True
                    # backward for each local and small graph
                    with torch.enable_grad():
                        outputs = m(*inputs)
                    if isinstance(outputs, torch.Tensor):
                        outputs = (outputs,)
                    torch.autograd.backward(outputs, grad_output)
                    grad_output = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                                        for inp in inputs)
                    bak_outputs = inputs
        return (None, None) + grad_output


def rev_sequential_backward_wrapper(m, *args, preserve_rng_state=True):
    return RevSequentialBackwardFunction.apply(m, preserve_rng_state, *args)
