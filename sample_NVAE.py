from nvae_model import AutoEncoder
import torchvision
import argparse
import utils
import torch
from utils import init_processes


def sample_from_VAE(VAE, t, batch_size):

    for p in VAE.parameters():
        p.requires_grad = False

    with torch.no_grad():
        logits = VAE.sample(batch_size, t)[0]
        sample = VAE.decoder_output(logits)
        final_sample = sample.sample()

    return final_sample


def main(eval_args):
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']

    if not hasattr(args, 'ada_groups'):
        args.ada_groups = False

    if not hasattr(args, 'min_groups_per_scale'):
        args.min_groups_per_scale = 1

    arch_instance = utils.get_arch_cells(args.arch_instance)
    model = AutoEncoder(args, None, arch_instance)
    model = model.cuda()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.cuda()

    iter_needed = eval_args.num_samples // eval_args.batch_size
    model.eval()
    for i in range(iter_needed):
        sample = sample_from_VAE(model, 1, eval_args.batch_size)

        for j in range(sample.size(0)):
            torchvision.utils.save_image(sample[j],
                                         (eval_args.savedir + f'/{j + i * eval_args.batch_size}.png'),
                                         normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sample from VAE')
    parser.add_argument('--checkpoint', type=str, default='/tmp/nvae/checkpoint.pth', help='location of the nvae checkpoint')
    parser.add_argument('--local_rank', type=int, default=0, help='rank of process')
    parser.add_argument('--savedir', default='./samples/', type=str, help='path to save samples for eval')
    parser.add_argument('--num_samples', type=int, default=10000, help='number of samples to generate')
    parser.add_argument('--batch_size', type=int, default = 40, help='batch size for generating samples from EBM')
    parser.add_argument('--master_address', type=str, default='127.0.0.1', help='address for master')
    args = parser.parse_args()

    args.distributed = False
    init_processes(0, 1, main, args)
