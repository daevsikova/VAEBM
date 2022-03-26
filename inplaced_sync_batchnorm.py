from __future__ import division

import torch
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from torch.autograd.function import Function


class SyncBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        input = input.contiguous()

        size = input.numel() // input.size(1)
        if size == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
        count = torch.Tensor([size]).to(input.device)

        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)

        count_all = torch.empty(world_size, 1, dtype=count.dtype, device=count.device)
        mean_all = torch.empty(world_size, mean.size(0), dtype=mean.dtype, device=mean.device)
        invstd_all = torch.empty(world_size, invstd.size(0), dtype=invstd.dtype, device=invstd.device)

        count_l = list(count_all.unbind(0))
        mean_l = list(mean_all.unbind(0))
        invstd_l = list(invstd_all.unbind(0))

        # using all_gather instead of all reduce so we can calculate count/mean/var in one go
        count_all_reduce = torch.distributed.all_gather(count_l, count, process_group, async_op=True)
        mean_all_reduce = torch.distributed.all_gather(mean_l, mean, process_group, async_op=True)
        invstd_all_reduce = torch.distributed.all_gather(invstd_l, invstd, process_group, async_op=True)

        # wait on the async communication to finish
        count_all_reduce.wait()
        mean_all_reduce.wait()
        invstd_all_reduce.wait()

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all.view(-1).long().tolist()
        )

        self.save_for_backward(input, weight, mean, invstd, bias)
        self.process_group = process_group
        self.world_size = world_size

        # apply element-wise normalization
        out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)

        # arash: apply swish
        assert eps == 1e-5, "I assumed below that eps is 1e-5"
        out = out * torch.sigmoid(out)
        # end arash

        return out

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, bias = self.saved_tensors

        # arash: re-compute batch normalized out
        eps = 1e-5
        out = torch.batch_norm_elemt(saved_input, weight, bias, mean, invstd, eps)
        sigmoid_out = torch.sigmoid(out)
        grad_output *= (sigmoid_out * (1 + out * (1 - sigmoid_out)))
        # end arash

        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group
        world_size = self.world_size

        # calculate local stats as well as grad_weight / grad_bias
        mean_dy, mean_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            self.needs_input_grad[0],
            self.needs_input_grad[1],
            self.needs_input_grad[2]
        )

        if self.needs_input_grad[0]:
            # synchronizing stats used to calculate input gradient.
            # TODO: move div_ into batch_norm_backward_elemt kernel
            mean_dy_all_reduce = torch.distributed.all_reduce(
                mean_dy, torch.distributed.ReduceOp.SUM, process_group, async_op=True)
            mean_dy_xmu_all_reduce = torch.distributed.all_reduce(
                mean_dy_xmu, torch.distributed.ReduceOp.SUM, process_group, async_op=True)

            # wait on the async communication to finish
            mean_dy_all_reduce.wait()
            mean_dy_xmu_all_reduce.wait()

            mean_dy.div_(world_size)
            mean_dy_xmu.div_(world_size)
            # backward pass for gradient calculation
            grad_input = torch.batch_norm_backward_elemt(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                mean_dy,
                mean_dy_xmu
            )

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None

        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class SyncMeanBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        input = input.contiguous()

        size = input.numel() // input.size(1)
        if size == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
        count = torch.Tensor([size]).to(input.device)

        # calculate mean/invstd for input.
        mean, invstd = torch.batch_norm_stats(input, eps)
        # mean = torch.mean(input, dim=[0, 2, 3])
        invstd = torch.ones_like(mean)

        count_all = torch.empty(world_size, 1, dtype=count.dtype, device=count.device)
        mean_all = torch.empty(world_size, mean.size(0), dtype=mean.dtype, device=mean.device)
        invstd_all = torch.empty(world_size, invstd.size(0), dtype=invstd.dtype, device=invstd.device)

        count_l = list(count_all.unbind(0))
        mean_l = list(mean_all.unbind(0))
        invstd_l = list(invstd_all.unbind(0))

        # using all_gather instead of all reduce so we can calculate count/mean/var in one go
        count_all_reduce = torch.distributed.all_gather(count_l, count, process_group, async_op=True)
        mean_all_reduce = torch.distributed.all_gather(mean_l, mean, process_group, async_op=True)
        invstd_all_reduce = torch.distributed.all_gather(invstd_l, invstd, process_group, async_op=True)

        # wait on the async communication to finish
        count_all_reduce.wait()
        mean_all_reduce.wait()
        invstd_all_reduce.wait()

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all.view(-1).long().tolist()
        )

        # apply element-wise normalization
        # out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        b = mean.view(1, -1, 1, 1) - bias.view(1, -1, 1, 1)
        input -= b

        self.save_for_backward(input, weight, mean, invstd, bias)
        self.process_group = process_group
        self.world_size = world_size

        out = input * torch.sigmoid(input)

        return out

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, bias = self.saved_tensors

        sigmoid_out = torch.sigmoid(saved_input)
        grad_output *= (sigmoid_out * (1 + saved_input * (1 - sigmoid_out)))

        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group
        world_size = self.world_size

        grad_bias = torch.sum(grad_output, dim=[0, 2, 3])
        mean_dy = grad_bias / (grad_output.size(0) * grad_output.size(2) * grad_output.size(3))

        if self.needs_input_grad[0]:
            # synchronizing stats used to calculate input gradient.
            # TODO: move div_ into batch_norm_backward_elemt kernel
            mean_dy_all_reduce = torch.distributed.all_reduce(
                mean_dy, torch.distributed.ReduceOp.SUM, process_group, async_op=True)

            # wait on the async communication to finish
            mean_dy_all_reduce.wait()

            mean_dy.div_(world_size)
            grad_input = grad_output - mean_dy.view(1, -1, 1, 1)

        # synchronizing of grad_weight / grad_bias is not needed as distributed
        # training would handle all reduce.
        if weight is None or not self.needs_input_grad[1]:
            grad_weight = None

        if weight is None or not self.needs_input_grad[2]:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class SyncBatchNormSwish(_BatchNorm):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, process_group=None):
        super(SyncBatchNormSwish, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.process_group = process_group
        self.ddp_gpu_size = None

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError('expected at least 2D input (got {}D input)'
                             .format(input.dim()))

    def _specify_ddp_gpu_num(self, gpu_size):
        if gpu_size > 1:
            raise ValueError('SyncBatchNorm is only supported for DDP with single GPU per process')
        self.ddp_gpu_size = gpu_size

    def forward(self, input):
        # currently only GPU input is supported
        if not input.is_cuda:
            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        need_sync = self.training or not self.track_running_stats
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            out = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
            return Swish.apply(out)
        else:
            # Arash: I only use it in this setting.
            if not self.ddp_gpu_size and False:
                raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')

            return SyncBatchNorm.apply(
                input, self.weight, self.bias, self.running_mean, self.running_var,
                self.eps, exponential_average_factor, process_group, world_size)