import os
from datetime import datetime
import numpy as np
import torch
import torchvision
import wandb
from torch import autograd, optim
from torch.autograd import Variable
from tqdm import tqdm
import torchvision.datasets as dset
import torchvision.transforms as transforms

import utils
from ebm_model import EBM
from nvae_model import AutoEncoder
from utils import sample_data
from utils import init_processes


def get_loaders(data, batch_size):
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    train_data = dset.CIFAR10(root=data, train=True, download=True, transform=train_transform)

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                              sampler=None, pin_memory=True, num_workers=8, drop_last=True)

    return train_queue


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def train(model, VAE, t, loader, model_path, total_iter=30000, batch_size=16):

    requires_grad(VAE.parameters(), False)
    loader = tqdm(enumerate(sample_data(loader)))

    if not os.environ.get("WANDB_API_KEY", None):
        os.environ["WANDB_API_KEY"] = 'e891f26c3ad7fd5a7e215dc4e344acc89c8861da'

    name = 'EBM' + datetime.strftime(datetime.now(), "_%h%d_%H%M%S")
    project = "project_gans"
    wandb.init(project=project, entity="daevsikova", name=name)

    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=4e-5, betas=(0.99, 0.999), weight_decay=3e-5)

    d_s_t = []

    with torch.no_grad():
        _, z_list, _ = VAE.sample(batch_size, t)

    noise_list = [torch.randn(zi.size()).cuda() for zi in z_list]

    for idx, (image) in loader:
        image = image[0]
        image = image.cuda()

        noise_x = torch.randn(image.size()).cuda()

        eps_z = [Variable(torch.Tensor(zi.size()).normal_(0, 1.0).cuda(), requires_grad=True) for zi in z_list]

        eps_x = torch.Tensor(image.size()).normal_(0, 1.0).cuda()
        eps_x = Variable(eps_x, requires_grad=True)

        requires_grad(parameters, False)
        model.eval()
        VAE.eval()

        step_size = 8e-5
        sample_step = 10
        # двойная динамика Ланжевена
        for k in range(sample_step):

            logits, _, log_p_total = VAE.sample(batch_size, t, eps_z)
            output = VAE.decoder_output(logits)
            neg_x = output.sample_given_eps(eps_x)
            log_pxgz = output.log_prob(neg_x).sum(dim=[1, 2])

            # energy
            dvalue = model(neg_x) - log_p_total - log_pxgz
            dvalue = dvalue.mean()
            dvalue.backward()

            for i in range(len(eps_z)):
                noise_list[i].normal_(0, 1)
                eps_z[i].data.add_(-0.5 * step_size, eps_z[i].grad.data * batch_size)
                eps_z[i].data.add_(np.sqrt(step_size), noise_list[i].data)
                eps_z[i].grad.detach_()
                eps_z[i].grad.zero_()

            # update x
            noise_x.normal_(0, 1)
            eps_x.data.add_(-0.5 * step_size, eps_x.grad.data * batch_size)
            eps_x.data.add_(np.sqrt(step_size), noise_x.data)
            eps_x.grad.detach_()
            eps_x.grad.zero_()

        eps_z = [eps_zi.detach() for eps_zi in eps_z]
        eps_x = eps_x.detach()

        requires_grad(parameters, True)
        model.train()

        model.zero_grad()
        logits, _, _ = VAE.sample(batch_size, t, eps_z)
        output = VAE.decoder_output(logits)

        neg_x = output.sample_given_eps(eps_x)

        pos_out = model(image)
        neg_out = model(neg_x)

        loss = pos_out.mean() - neg_out.mean()
        loss.backward()
        optimizer.step()

        loader.set_description(f"loss: {loss.mean().item():.5f}")
        loss_print = pos_out.mean() - neg_out.mean()
        d_s_t.append(loss_print.item())

        wandb.log({"EMB loss": loss.mean().item(), "EMB total loss": loss.mean().item()})

        if idx % 20 == 0:
            torchvision.utils.save_image(neg_x, model_path + "/images/sample.png", nrow=16, normalize=True)
            wandb.log({"Sample": wandb.Image((model_path + "/images/sample.png".format(idx)))})

            torch.save(d_s_t, model_path + "d_s_t")

        if idx % 500 == 0:
            state_dict = {}
            state_dict["model"] = model.state_dict()
            state_dict["optimizer"] = optimizer.state_dict()
            model_save_path = model_path + "EBM_{}.pth".format(idx)
            torch.save(state_dict, model_save_path)
            wandb.save(model_save_path)

        if idx == total_iter:
            break


def main(batch_size=16, checkpoint_path="./checkpoints/NVAE_checkpoint.pt"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint["args"]
    arch_instance = utils.get_arch_cells(args.arch_instance)

    # define and load pre-trained VAE
    model = AutoEncoder(args, None, arch_instance)
    model = model.cuda()
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.cuda()

    t = 1
    loader = get_loaders('./data/cifar10', batch_size)

    EBM_model = EBM(3, 64).cuda()

    model_path = "./saved_models/cifar10/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        os.makedirs(model_path + "/images/")

    # use 5 batch of training images to initialize the data dependent init for weight norm
    init_image = []
    for idx, (image) in enumerate(loader):
        img = image[0]
        init_image.append(img)
        if idx == 4:
            break
    init_image = torch.stack(init_image).cuda()
    init_image = init_image.view(-1, 3, 32, 32)

    EBM_model(init_image)
    train(EBM_model, model, t, loader, model_path)


if __name__ == "__main__":
    init_processes(0, 1, main, 16, "./checkpoints/NVAE_checkpoint.pt")
