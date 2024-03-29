import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from collections import OrderedDict
from torch import optim, Tensor
from typing import List, Dict, Any

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
EMBED_DIM = 768
HIDDEN_N = int(os.environ['VAE_HIDDEN_N']
               ) if 'VAE_HIDDEN_N' in os.environ else 4
VOCAB_SIZE = 560
LATENT_DIM = int(os.environ['VAE_LATENT_DIM']
                 ) if 'VAE_LATENT_DIM' in os.environ else 512
COMPUTE_LOGITS = False
DROPOUT = 0.05
LR = 1e-5


def layer_sizes(num=HIDDEN_N, latent_dim=LATENT_DIM):
    sizes = [latent_dim * 2 ** i for i in range(num)]

    sizes.reverse()

    return sizes


class Encoder(nn.Module):
    def __init__(self, hidden_n=HIDDEN_N, latent_dim=LATENT_DIM, embed_dim=EMBED_DIM):
        super(Encoder, self).__init__()

        hidden_dims = layer_sizes(hidden_n, latent_dim)
        hidden = OrderedDict()
        hidden['input'] = nn.Linear(embed_dim, hidden_dims[0], bias=False)

        for i in range(0, len(hidden_dims)):
            dim = hidden_dims[i]
            next_dim = hidden_dims[i+1] if (i+1) <= len(hidden_dims) - 1 else dim

            hidden[f'hidden_{i}'] = nn.Sequential(
                nn.Linear(
                    dim,
                    next_dim,
                    bias=False
                ),
                nn.LeakyReLU(0.2)
            )

        self.dropout = nn.Dropout(DROPOUT)
        self.module = nn.Sequential(hidden)
        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        hidden = self.dropout(self.module(input))
        mean = self.fc_mean(hidden)
        logvar = self.fc_logvar(hidden)

        return hidden, mean, logvar


class Decoder(nn.Module):
    def __init__(self, hidden_n=HIDDEN_N, latent_dim=LATENT_DIM, embed_dim=EMBED_DIM):
        super(Decoder, self).__init__()

        hidden_dims = layer_sizes(hidden_n, latent_dim)
        hidden_dims.reverse()

        hidden = OrderedDict()
        hidden['input'] = nn.Linear(latent_dim, hidden_dims[0], bias=False)

        for i in range(0, len(hidden_dims)):
            dim = hidden_dims[i]
            next_dim = hidden_dims[i+1] if (i+1) <= len(hidden_dims) - 1 else dim

            hidden[f'hidden_{i}'] = nn.Sequential(
                nn.Linear(
                    dim,
                    next_dim,
                    bias=False
                ),
                nn.LeakyReLU(0.2)
            )

        hidden['out'] = nn.Linear(hidden_dims[-1], embed_dim, bias=False)
        self.module = nn.Sequential(hidden)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        result = self.module(input)

        return result


class VAE(pl.LightningModule):

    def __init__(self,
                 embed_dim: int = EMBED_DIM,
                 latent_dim: int = LATENT_DIM,
                 hidden_n: int = HIDDEN_N,
                 vocab_size: int = VOCAB_SIZE,
                 ) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = Encoder(hidden_n, latent_dim, embed_dim)
        self.decoder = Decoder(hidden_n, latent_dim, embed_dim)
        self.z_emb = nn.Linear(latent_dim, embed_dim, bias=False)

        if COMPUTE_LOGITS:
            self.proj = nn.Linear(embed_dim, vocab_size, bias=True)
            self.ln_out = nn.LayerNorm(vocab_size)

            self.proj.bias.data.zero_()
            self.proj.weight.data.uniform_(-0.1, 0.1)

        self.emb.weight.data.uniform_(-0.1, 0.1)

    def encode(self, input: Tensor) -> List[Tensor]:
        emb = self.emb(input)
        hidden, mean, logvar = self.encoder(emb)

        return [emb, hidden, mean, logvar]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder(z) + self.z_emb(z)

        return result

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mean, var) from
        N(0,1).
        :param mean: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mean

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        emb, hidden, mean, log_var = self.encode(input)
        z = self.reparameterize(mean, log_var)
        emb_hat = self.decode(z)

        if COMPUTE_LOGITS:
            output = self.proj(emb_hat)
            output = self.ln_out(output)
            output = torch.softmax(output, dim=-1)
        else:
            output = None

        return [output, emb_hat, emb, hidden, mean, log_var]

    def kl_loss(self, mean: Tensor, logvar: Tensor) -> Tensor:
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        return kl_loss

    def reconstruction_loss(self, input_emb: Tensor, output_emb: Tensor, input: Tensor, output: Tensor, padding_index=0) -> Tensor:
        loss_emb = F.mse_loss(output_emb, input_emb)

        if not COMPUTE_LOGITS:
            return loss_emb

        output = torch.argmax(output, dim=1)

        # Create a mask to exclude the padding_idx from loss computation
        mask = input != padding_index

        # Apply the mask to the input and output tensors
        input_masked = input.to(dtype=torch.float).view(-1)[mask.view(-1)]
        output_masked = output.to(
            dtype=torch.float).view(-1)[mask.view(-1)]

        # Compute the cross-entropy loss only for non-padding positions
        loss = F.cross_entropy(output_masked, input_masked)

        recon_loss = loss_emb + loss

        return recon_loss

    def loss_function(self, input_emb: Tensor, output_emb: Tensor, input: Tensor, output: Tensor, mean: Tensor, logvar: Tensor, padding_index=0) -> Tensor:
        recon_loss = self.reconstruction_loss(
            input_emb, output_emb, input, output, padding_index)
        kl_loss = self.kl_loss(mean, logvar)
        total_loss = recon_loss + kl_loss

        return total_loss

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)

        return samples

    def training_step(self, batch, batch_idx):
        # handle x,y case or proceed unmodified
        input = torch.cat((batch[0], batch[1]), 1) \
            if type(batch) == list and len(batch) == 2 else batch
        output, emb_hat, emb, hidden, mean, log_var = self(input)
        loss = self.loss_function(
            emb,
            emb_hat,
            input,
            output,
            mean, log_var, padding_index=0
        )

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=LR, betas=(0.5, 0.999))
        self.trainer._scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.7)

        return [optimizer], [self.trainer._scheduler]

    def from_pretrained(pth_path: str, embed_dim=EMBED_DIM, latent_dim=LATENT_DIM, hidden_n=HIDDEN_N, vocab_size=VOCAB_SIZE):
        model = VAE(
            embed_dim,
            latent_dim,
            hidden_n,
            vocab_size
        )

        load_dict = torch.load(pth_path, map_location=DEVICE)
        load_keys = load_dict.keys()

        for k in model.state_dict():
            if k not in load_keys:
                load_dict[k] = model.state_dict()[k]

        model.load_state_dict(load_dict, strict=True)

        return model
