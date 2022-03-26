import torch
import numpy as np
from torch.autograd import Variable
from nvae_model import AutoEncoder
import utils
import torchvision
from tqdm import tqdm
from ebm_model import EBM
from utils import init_processes


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def sample_from_EBM(model, VAE, t, batch_size=16):
    parameters = model.parameters()
    requires_grad(VAE.parameters(), False)
    requires_grad(parameters, False)

    with torch.no_grad():
        _, z_list, _ = VAE.sample(batch_size, t)
        image = torch.zeros(batch_size, 3, 32, 32)

    model.eval()
    VAE.eval()

    noise_x = torch.randn(image.size()).cuda()
    noise_list = [torch.randn(zi.size()).cuda() for zi in z_list]

    eps_z = [Variable(torch.Tensor(zi.size()).normal_(0, 1.).cuda(), requires_grad=True) for zi in z_list]

    eps_x = torch.Tensor(image.size()).normal_(0, 1.).cuda()
    eps_x = Variable(eps_x, requires_grad=True)

    step_size = 8e-5
    sample_step = 20

    # динамика Ланжевена
    for k in tqdm(range(sample_step)):

        logits, _, log_p_total = VAE.sample(batch_size, t, eps_z)
        output = VAE.decoder_output(logits)
        neg_x = output.sample_given_eps(eps=eps_x)
        neg_x_renorm = 2. * neg_x - 1.

        log_pxgz = output.log_prob(neg_x_renorm).sum(dim=[1, 2])
        dvalue = model(neg_x_renorm) - log_p_total - log_pxgz
        dvalue = dvalue.mean()
        dvalue.backward()

        for i in range(len(eps_z)):
            noise_list[i].normal_(0, 1)

            eps_z[i].data.add_(-0.5*step_size, eps_z[i].grad.data * batch_size)
            eps_z[i].data.add_(np.sqrt(step_size), noise_list[i].data)
            eps_z[i].grad.detach_()
            eps_z[i].grad.zero_()

        noise_x.normal_(0, 1)
        eps_x.data.add_(-0.5*step_size, eps_x.grad.data * batch_size)
        eps_x.data.add_(np.sqrt(step_size), noise_x.data)
        eps_x.grad.detach_()
        eps_x.grad.zero_()


    eps_z = [eps_zi.detach() for eps_zi in eps_z]
    eps_x = eps_x.detach()

    logits, _, _ = VAE.sample(batch_size, t, eps_z)
    output = VAE.decoder_output(logits)
    final_sample = output.sample_given_eps(eps_x)  # [bs, 3, 32, 32]
    return final_sample


def main(num_samples=1000, batch_size=16, vae_checkpoint_path='./saved_models/cifar_10/NVAE.pt',
         ebm_checkpoint_path='./saved_models/cifar_10/EBM.pt'):
    checkpoint = torch.load(vae_checkpoint_path, map_location='cpu')
    args = checkpoint['args']
    arch_instance = utils.get_arch_cells(args.arch_instance)
    model = AutoEncoder(args, None, arch_instance)
    model = model.cuda()
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.cuda()

    t = 1.
    EBM_model = EBM(3, 64).cuda()

    with torch.no_grad():
        EBM_model(torch.rand(10, 3, 32, 32).cuda())

    state_EBM = torch.load(ebm_checkpoint_path)
    EBM_model.load_state_dict(state_EBM['model'])

    iter_needed = num_samples // batch_size
    model.eval()
    for i in range(iter_needed):
        sample = sample_from_EBM(EBM_model, model, t, batch_size)
        for j in range(sample.size(0)):
            torchvision.utils.save_image(sample[j], './samples/' + f'/{j + i * batch_size}.png', normalize=True)


if __name__ == '__main__':
    init_processes(0, 1, main, 1000, 16, './saved_models/cifar_10/NVAE.pt', './saved_models/cifar_10/EBM.pt')