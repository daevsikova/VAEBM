import matplotlib
matplotlib.use('agg')

import logging
import time
import sys

import torch
import numpy as np
import torch.distributed as dist
from tensorboardX import SummaryWriter


import os

def cleanup():
    dist.destroy_process_group()

def init_processes(rank, size, fn, bs, path):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(0)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(bs, path)
    cleanup()

def count_parameters_in_M(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def logsumexp(logits, dim):
    mx = torch.max(logits, dim, keepdim=True)[0]
    return torch.log(torch.sum(torch.exp(logits - mx), dim=dim, keepdim=True)) + mx


class Logger(object):
    def __init__(self, rank, save):
        self.rank = rank
        if self.rank == 0:
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(save, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
            self.start_time = time.time()

    def info(self, string, *args):
        if self.rank == 0:
            elapsed_time = time.time() - self.start_time
            elapsed_time = time.strftime(
                '(Elapsed: %H:%M:%S) ', time.gmtime(elapsed_time))
            if isinstance(string, str):
                string = elapsed_time + string
            else:
                logging.info(elapsed_time)
            logging.info(string, *args)


class Writer(object):
    def __init__(self, rank, save):
        self.rank = rank
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=save, flush_secs=20)

    def add_scalar(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_scalar(*args, **kwargs)

    def add_figure(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_figure(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_image(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.add_histogram(*args, **kwargs)

    def add_histogram_if(self, write, *args, **kwargs):
        if write and False:   # Used for debugging.
            self.add_histogram(*args, **kwargs)

    def close(self, *args, **kwargs):
        if self.rank == 0:
            self.writer.close()


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def get_stride_for_cell_type(cell_type):
    if cell_type.startswith('normal') or cell_type.startswith('combiner'):
        stride = 1
    elif cell_type.startswith('down'):
        stride = 2
    elif cell_type.startswith('up'):
        stride = -1
    else:
        raise NotImplementedError(cell_type)

    return stride

def average_gradients(params, is_distributed):
    """ Gradient averaging. """
    if is_distributed:
        size = float(dist.get_world_size())
        for param in params:
            if param.requires_grad:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= size


def average_params(params):
    """ parameter averaging. """
    size = float(dist.get_world_size())
    for param in params:
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size


def average_tensor(t, is_distributed):
    if is_distributed:
        size = float(dist.get_world_size())
        dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
        t.data /= size


def one_hot(indices, depth, dim):
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size).cuda()
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)

    return y_onehot


def num_output(dataset):
    if dataset == 'mnist':
        return 28 * 28
    elif dataset == 'stacked_mnist':
        return 28 * 28
    elif dataset == 'cifar10':
        return 3 * 32 * 32
    elif dataset.startswith('celeba') or dataset.startswith('imagenet') or dataset.startswith('lsun'):
        size = int(dataset.split('_')[-1])
        return 3 * size * size
    elif dataset == 'ffhq':
        return 3 * 256 * 256
    else:
        raise NotImplementedError


def get_input_size(dataset):
    if dataset == 'cifar10':
        return 32
    else:
        raise NotImplementedError


def pre_process(x, num_bits):
    if num_bits != 8:
        x = torch.floor(x * 255 / 2 ** (8 - num_bits))
        x /= (2 ** num_bits - 1)
    return x


def sample_data(loader):
    loader_iter = iter(loader)

    while True:
        try:
            yield next(loader_iter)

        except StopIteration:
            loader_iter = iter(loader)

            yield next(loader_iter)


def clip_grad(parameters, optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))



def get_arch_cells(arch_type):
    if arch_type == 'res_elu':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_elu', 'res_elu']
        arch_cells['down_enc'] = ['res_elu', 'res_elu']
        arch_cells['normal_dec'] = ['res_elu', 'res_elu']
        arch_cells['up_dec'] = ['res_elu', 'res_elu']
        arch_cells['normal_pre'] = ['res_elu', 'res_elu']
        arch_cells['down_pre'] = ['res_elu', 'res_elu']
        arch_cells['normal_post'] = ['res_elu', 'res_elu']
        arch_cells['up_post'] = ['res_elu', 'res_elu']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_bnelu':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnelu', 'res_bnelu']
        arch_cells['down_enc'] = ['res_bnelu', 'res_bnelu']
        arch_cells['normal_dec'] = ['res_bnelu', 'res_bnelu']
        arch_cells['up_dec'] = ['res_bnelu', 'res_bnelu']
        arch_cells['normal_pre'] = ['res_bnelu', 'res_bnelu']
        arch_cells['down_pre'] = ['res_bnelu', 'res_bnelu']
        arch_cells['normal_post'] = ['res_bnelu', 'res_bnelu']
        arch_cells['up_post'] = ['res_bnelu', 'res_bnelu']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_bnswish':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_sep':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k5g0']
        arch_cells['down_enc'] = ['mconv_e6k5g0']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['mconv_e3k5g0']
        arch_cells['down_pre'] = ['mconv_e3k5g0']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_sep11':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k11g0']
        arch_cells['down_enc'] = ['mconv_e6k11g0']
        arch_cells['normal_dec'] = ['mconv_e6k11g0']
        arch_cells['up_dec'] = ['mconv_e6k11g0']
        arch_cells['normal_pre'] = ['mconv_e3k5g0']
        arch_cells['down_pre'] = ['mconv_e3k5g0']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res53_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish5', 'res_bnswish']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res35_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish5']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res55_mbconv':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['down_enc'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['down_pre'] = ['res_bnswish5', 'res_bnswish5']
        arch_cells['normal_post'] = ['mconv_e3k5g0']
        arch_cells['up_post'] = ['mconv_e3k5g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'res_mbconv9':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_dec'] = ['mconv_e6k9g0']
        arch_cells['up_dec'] = ['mconv_e6k9g0']
        arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_post'] = ['mconv_e3k9g0']
        arch_cells['up_post'] = ['mconv_e3k9g0']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_res':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k5g0']
        arch_cells['down_enc'] = ['mconv_e6k5g0']
        arch_cells['normal_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_dec'] = ['res_bnswish', 'res_bnswish']
        arch_cells['normal_pre'] = ['mconv_e3k5g0']
        arch_cells['down_pre'] = ['mconv_e3k5g0']
        arch_cells['normal_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['up_post'] = ['res_bnswish', 'res_bnswish']
        arch_cells['ar_nn'] = ['']
    elif arch_type == 'mbconv_den':
        arch_cells = dict()
        arch_cells['normal_enc'] = ['mconv_e6k5g0']
        arch_cells['down_enc'] = ['mconv_e6k5g0']
        arch_cells['normal_dec'] = ['mconv_e6k5g0']
        arch_cells['up_dec'] = ['mconv_e6k5g0']
        arch_cells['normal_pre'] = ['mconv_e3k5g8']
        arch_cells['down_pre'] = ['mconv_e3k5g8']
        arch_cells['normal_post'] = ['mconv_e3k5g8']
        arch_cells['up_post'] = ['mconv_e3k5g8']
        arch_cells['ar_nn'] = ['']
    else:
        raise NotImplementedError

    return arch_cells


def groups_per_scale(num_scales, num_groups_per_scale, is_adaptive, divider=2, minimum_groups=1):
    g = []
    n = num_groups_per_scale
    for s in range(num_scales):
        assert n >= 1
        g.append(n)
        if is_adaptive:
            n = n // divider
            n = max(minimum_groups, n)
    return g


