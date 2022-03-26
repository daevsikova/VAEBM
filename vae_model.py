import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch import distributions as dist
import numpy as np
from torch.utils import data
from tqdm import tqdm
from distributions import NormalDecoder

device = 'cuda'


class Encoder(nn.Module):
    def __init__(self, in_ch=3, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(in_ch, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)  # 16 x 16
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)  # 8 x 8
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)  # 4 x 4
        self.linear = nn.Linear(4 * 4 * 256, 2 * latent_dim)

    def forward(self, x):
        bs = x.shape[0]
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.contiguous().view(bs, -1)
        out = self.linear(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_ch=3, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.linear = nn.Linear(latent_dim, 4 * 4 * 128)
        self.t_conv1 = nn.ConvTranspose2d(128, 128, 4, 2, 1)  # 8 x 8
        self.t_conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 16 x 16
        self.t_conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 32 x 32
        self.conv = nn.Conv2d(32, 2 * in_ch, 3, 1, 1)

    def forward(self, x):
        bs = x.shape[0]
        out = F.relu(self.linear(x))
        out = out.view(bs, 4, 4, 128)
        out = out.permute(0, 3, 1, 2)
        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))
        out = F.relu(self.t_conv3(out))
        out = self.conv(out)
        return out


class VAE(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.lat_dist = dist.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))

    def forward(self, x):

        mu_z, log_sig_z = self.encoder(x).chunk(2, dim=1)
        z_sample = self.lat_dist.sample(mu_z.shape).squeeze(2)
        z_sample = z_sample * log_sig_z.exp() + mu_z
        z_sample = z_sample.to(device)
        mu_x = self.decoder(z_sample)
        return mu_x

    def loss(self, x):
        bs = x.shape[0]

        mu_z, log_sig_z = self.encoder(x).chunk(2, dim=1)
        z = self.lat_dist.sample(mu_z.shape).squeeze(2)
        z = z * log_sig_z.exp() + mu_z
        z = z.to(device)  # [128, 16]

        logits = self.decoder(z)
        output = self.decoder_output(logits)
        mu_x = output.mu
        rec_loss = (x - mu_x) ** 2  # = -c * log_prob
        rec_loss = rec_loss.view(bs, -1).sum(1).mean()

        log_qz_x = -0.5 * np.log(2 * np.pi) - log_sig_z - 0.5 * (z - mu_z) ** 2 * torch.exp(-2 * log_sig_z)

        log_pz = self.lat_dist.log_prob(z)
        kl = (log_qz_x - log_pz).sum(1).mean()
        return rec_loss, kl

    def decoder_output(self, logits):
        return NormalDecoder(logits)

    def test(self, testloader):
        loss = 0
        tl = 0
        kl = 0
        log_prob = 0
        with torch.no_grad():
            for batch in tqdm(testloader):
                batch = batch.to(device)
                lp, kl_b = self.loss(batch)
                loss += (lp + kl_b) * batch.shape[0]
                kl += kl_b * batch.shape[0]
                log_prob += lp * batch.shape[0]
                tl += batch.shape[0]
        return np.array([(loss / tl).item(), (log_prob / tl).item(), (kl / tl).item()])

    def fit(self, train_data, test_data, lr=1e-3, num_epochs=100, bs=128):
        trainloader = data.DataLoader(train_data,
                                      batch_size=bs,
                                      shuffle=True)

        testloader = torch.utils.data.DataLoader(test_data,
                                                 batch_size=bs)

        optim = opt.Adam(self.parameters(), lr=lr)

        train_loss = None
        test_loss = self.test(testloader)

        for epoch in tqdm(range(num_epochs)):
            for batch in tqdm(trainloader):
                batch = batch.to(device)
                lp, kl = self.loss(batch)
                loss = lp + kl
                optim.zero_grad()
                loss.backward()
                optim.step()

                if train_loss is None:
                    train_loss = np.array([loss.cpu().detach().numpy(),
                                           kl.cpu().detach().numpy(),
                                           lp.cpu().detach().numpy()])
                else:
                    train_loss = np.vstack((train_loss,
                                            np.array([loss.cpu().detach().numpy(),
                                                      kl.cpu().detach().numpy(),
                                                      lp.cpu().detach().numpy()])))

            cur_test_loss = self.test(testloader)
            print(
                f'Epoch {epoch + 1}: -ELBO= {cur_test_loss[0]:.4f}, Recon Loss= {cur_test_loss[1]:.4f}, KL Loss= {cur_test_loss[2]:.4f}')
            test_loss = np.vstack((test_loss, cur_test_loss))

            PATH = f'/vae_{epoch + 1}.trch'
            torch.save(self.state_dict(), PATH)

        return train_loss, test_loss

    def sample(self, n=100):
        with torch.no_grad():
            z = self.lat_dist.sample((n, self.latent_dim)).squeeze(2)
            logits = self.decoder(z)
            output = self.decoder_output(logits)
            mu_x = output.mu
            x = torch.clip(mu_x, -1, 1)
        return x.cpu().detach().numpy()
