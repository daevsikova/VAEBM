from utils.vqvae_utils import *

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb
import os
import torchvision
from datetime import datetime

from vqvae_model import PixelCNN, VQVAE

DEVICE = 'cuda'


def test(model, testloader, renorm=True):
    with torch.no_grad():
        total_loss = 0
        tc = 0
        for batch in testloader:
            batch = batch.to(DEVICE).float()
            if renorm:
                batch = batch * 2 - 1
            total_loss += model.loss(batch) * batch.shape[0]
            tc += batch.shape[0]
        return total_loss / tc


def train(model, train_data, test_data, num_epochs=20, lr=1e-3, batch_size=128, renorm=True, exp_name='vqvae'):
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    if exp_name == 'vqvae':
        dl = DataLoader(test_data[:50], batch_size=50, shuffle=False)
        

    train_loss = []
    test_loss = []

    loss_test = test(model, testloader, renorm).item()
    test_loss.append(loss_test)

    model_path = f'./saved_models/{exp_name}/'
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(os.path.join(model_path, 'images'), exist_ok=True)
    
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        model.train()
        for batch in trainloader:
            batch = batch.to(DEVICE).float()
            if renorm:
                batch = batch * 2 - 1
            loss = model.loss(batch)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss.append(loss.item())
            wandb.log({f"Train loss {exp_name}": loss.item()})
        pbar.set_postfix({'train_loss': loss.item()})

        model.eval()
        loss_test = test(model, testloader, renorm).item()
        test_loss.append(loss_test)
        wandb.log({f"Test loss {exp_name}": loss_test})

        state_dict = {}
        state_dict["model"] = model.state_dict()
        model_save_path = model_path + f"{exp_name}_{epoch}.pth"
        torch.save(state_dict, model_save_path)
        wandb.save(model_save_path)

        if exp_name == 'vqvae':
            for batch in dl:
                recs = model.get_reconstructions(batch.to(DEVICE))
                torchvision.utils.save_image(recs, model_path + f"/images/rec_{epoch}.png", nrow=10)
                wandb.log({"Reconstructions": wandb.Image((model_path + f"/images/rec_{epoch}.png"))})
    
    return np.array(train_loss), np.array(test_loss)


def main(train_data, test_data, dset_id):
    model = VQVAE().to(DEVICE)

    train_data = np.transpose(train_data.astype(np.float32), (0, 3, 1, 2)) / 255.
    test_data = np.transpose(test_data.astype(np.float32), (0, 3, 1, 2)) / 255.

    print('train shape:', train_data.shape)
    print('test shape:', test_data.shape)

    if not os.environ.get("WANDB_API_KEY", None):
        os.environ["WANDB_API_KEY"] = 'e891f26c3ad7fd5a7e215dc4e344acc89c8861da'

        name = 'VQVAE' + datetime.strftime(datetime.now(), "_%h%d_%H%M%S")
        project = "project_gans"
        wandb.init(project=project, entity="daevsikova", name=name)

    train_loss_vae, test_loss_vae = train(model, 
                                train_data, 
                                test_data, 
                                num_epochs=1000, 
                                lr=1e-3, 
                                batch_size=256, 
                                exp_name='vqvae')
    
    print('PixelCNN training')
    model_prior = PixelCNN(input_shape=(8, 8, 1), dim=256, res_blocks=10 if dset_id == 1 else 15).to(DEVICE)
    train_dl = DataLoader(train_data, batch_size=128)
    test_dl = DataLoader(test_data, batch_size=128)
    
    with torch.no_grad():
        train_embs = []
        for batch in iter(train_dl):
            z = model.encoder(batch.to(DEVICE).mul(2).sub(1))
            train_embs.append(model.find_nearest(z))

        test_embs = []
        for batch in iter(test_dl):
            z = model.encoder(batch.to(DEVICE).mul(2).sub(1))
            test_embs.append(model.find_nearest(z))

    train_embs = torch.cat(train_embs, dim=0)
    test_embs = torch.cat(test_embs, dim=0)
    print('train_embs:', train_embs.shape)
    print('test_embs:', test_embs.shape)

    train_loss_pcnn, test_loss_pcnn = train(
        model_prior,
        train_embs,
        test_embs,
        num_epochs=20,
        lr=1e-3,
        batch_size=40, 
        renorm = False, 
        exp_name='pixelCNN'
    )
    
    # sampling
    model_path = f'./saved_models/'
    os.makedirs(os.path.join(model_path, 'images'), exist_ok=True)
    
    z_samples = model_prior.sample(100).long()

    with torch.no_grad():
        samples = model.decoder(model.embeddings(z_samples).permute(0, 3, 1, 2)).clamp(-1, 1).add(1).div(2) 
        
        torchvision.utils.save_image(samples, model_path + "/images/sample.png", nrow=10)
        wandb.log({"Sample": wandb.Image((model_path + "/images/sample.png"))})

    return train_loss_vae, test_loss_vae, train_loss_pcnn, test_loss_pcnn, samples, samples


if __name__ == '__main__':
    save_results(2, main)
