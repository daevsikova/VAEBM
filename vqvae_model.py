import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(dim), 
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.layer(x) + x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, dim, 4, 2, 1), 
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim)
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=3, dim=256):
        super().__init__()
        self.model = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(), 
            nn.ConvTranspose2d(dim, out_channels, 4, 2, 1)
        )

    def forward(self, x):
        return self.model(x)


class MaskedConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.set_mask()
    
    def forward(self, x):
        # self.weight.data *= mask
        return F.conv2d(x, self.mask * self.weight, bias=self.bias, stride=self.stride, padding=self.padding)

    def set_mask(self):
        raise NotImplementedError()


class ConvA(MaskedConv):
    def set_mask(self):
        h, w = self.kernel_size
        self.mask[:, :, h // 2, : w // 2] = 1
        self.mask[:, :, :h // 2] = 1


class ConvB(MaskedConv):
    def set_mask(self):
        h, w = self.kernel_size
        self.mask[:, :, h // 2, : w // 2 + 1] = 1
        self.mask[:, :, :h // 2] = 1


class ResBlockCNN(nn.Module):
    def __init__(self, in_c, use_norm=True):
        super().__init__()
        layers = [nn.ReLU(),
                  ConvB(in_c, in_c // 2, kernel_size=(1, 1), stride=1, padding=0),
                  nn.ReLU(), 
                  ConvB(in_c // 2, in_c // 2, kernel_size=(7, 7), stride=1, padding=3),
                  nn.ReLU(), 
                  ConvB(in_c // 2, in_c, kernel_size=(1, 1), stride=1, padding=0)
        ]
        self.use_norm = use_norm
        self.resblock = nn.Sequential(*layers)
        if self.use_norm:
            self.norm = nn.LayerNorm(in_c)

    def forward(self, x):
        output = self.resblock(x)
        res = output + x
        if self.use_norm:
            res = res.permute(0, 2, 3, 1)
            res = self.norm(res)
            return res.permute(0, 3, 1, 2)
        return res


class PixelCNN(nn.Module):
    def __init__(self, input_shape, dim=256, nf=120, res_blocks=12):
        super().__init__()
        self.H, self.W, self.C = input_shape
        
        layers = [ConvA(dim, nf, kernel_size=(7, 7), stride=1, padding=3)]
        
        if res_blocks > 0:
            for _ in range(res_blocks):
                layers.extend([ResBlockCNN(nf, use_norm=False)])
        
        layers.extend([nn.ReLU()])
        layers.extend([
              ConvB(nf, nf, kernel_size=(1, 1)), nn.ReLU(), 
              ConvB(nf, 128, kernel_size=(1, 1))
        ])
        
        self.model = nn.Sequential(*layers)
        self.embs = nn.Embedding(128, dim)
        
    def forward(self, x):
        inp = self.embs.forward(x.long()).permute(0, 3, 1, 2)
        return self.model.forward(inp).reshape(-1, 128, self.H, self.W).contiguous()

    def sample(self, n=100):
        with torch.no_grad():
            samples = torch.zeros(n, self.H, self.W).to('cuda')

            for i in range(self.H):
                for j in range(self.W):
                    for c in range(self.C):
                        output = self.forward(samples)[:, :, i, j]
                        samples[:, i, j] = torch.distributions.categorical.Categorical(logits=output).sample()
            
            return samples
            
    def loss(self, x):
        return F.cross_entropy(self.forward(x), x.long())


class VQVAE(nn.Module):
    def __init__(self, emb_dim=256, k=128):
        super().__init__()

        self.encoder = Encoder(dim=256)
        self.decoder = Decoder(dim=256)

        self.k = k
        self.emb_dim = emb_dim
        self.beta = 0.5

        self.embeddings = nn.Embedding(num_embeddings=self.k, embedding_dim=self.emb_dim)
        nn.init.uniform_(self.embeddings.weight, -1 / self.k, 1 / self.k)
        
    def find_nearest(self, z):
        distances = torch.cdist(self.embeddings.weight, z.permute(0, 2, 3, 1).reshape(-1, self.emb_dim))
        B, C, H, W = z.shape
        nearest_idx = torch.argmin(distances, dim=0)
        
        return nearest_idx.reshape(B, H, W)

    def forward(self, x):
        z = self.encoder(x)
        embs_idx = self.find_nearest(z)
        embs = self.embeddings(embs_idx).permute(0, 3, 1, 2).contiguous()
        x_pred = self.decoder((embs - z).detach() + z)

        loss = self.beta * ((z - embs.detach()) ** 2).mean() + ((embs - z.detach()) ** 2).mean()
        
        return x_pred, loss

    def loss(self, x):
        x_pred, loss = self.forward(x)
        mse = F.mse_loss(x_pred, x)
        loss += mse
        return loss

    def get_reconstructions(self, x):
        with torch.no_grad():
            z = self.encoder(x.mul(2).sub(1))
            embs_idx = self.find_nearest(z)
            embs = self.embeddings(embs_idx).permute(0, 3, 1, 2).contiguous()
            
            x_pred = torch.stack((x, self.decoder(embs).clamp(-1, 1).add(1).div(2)), dim=1).view(-1, 3, 32, 32)
            # x_pred = x_pred.permute(0, 2, 3, 1).cpu().numpy()
        return x_pred
