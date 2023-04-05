import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange

from spatial_genomics_autoencoders.models.unet import Unet, DecoderBlock


def construct_tile_expression(padded_exp, masks, n_voxels, normalize=True):
    tile = torch.zeros((masks.shape[0], masks.shape[-2], masks.shape[-1], padded_exp.shape[-1]),
                       device=padded_exp.device)
    for b in range(tile.shape[0]):
        for exp, m in zip(padded_exp[b], masks[b]):
            tile[b, :, :][m==1] = exp.to(torch.float32)
            
    tile = rearrange(tile, 'b h w c -> b c h w')
    tile = tile.detach().cpu().numpy()
    
    tile /= np.expand_dims(tile.max(axis=(0, -2, -1)), (0, -2, -1))

    return rearrange(tile, 'b c h w -> b h w c')


def reduce_to_voxel_level(x, masks):
    """
    x - (b, c, h, w)
    masks - (b, v, h, w)
    
    out - (b, v, c)
    """
    masks = masks.unsqueeze(dim=2) # (b, v, c, h, w)
    x = x.unsqueeze(dim=1).repeat(1, masks.shape[1], 1, 1, 1) # (b, v, c, h, w)
    x *= masks 
    return x.sum(dim=(-1, -2)) # (b, v, m)


def mask_nb_params(r, voxel_idxs):
    mask = torch.zeros_like(voxel_idxs, dtype=torch.bool)
    if r.is_cuda:
        mask = mask.cuda()
        
    mask[voxel_idxs == 0] = 1

    mask = mask.unsqueeze(dim=-1)
    masked_r = r.masked_fill(mask, 0.)
    
    return masked_r


class LoggingCallback(pl.Callback):
    def __init__(self, log_every=10, log_n_samples=8, plot_genes=['EPCAM', 'IL7R', 'ACTA2']):
        self.log_every = log_every
        self.log_n_samples = log_n_samples
        self.plot_genes = plot_genes

    def log_epoch(self, key, trainer, pl_module, outputs, batch, batch_idx):
            img = batch['he'][:self.log_n_samples].clone().detach().cpu()
            img -= img.min()
            img /= img.max()
            trainer.logger.log_image(
                key=f"{key}/he",
                images=[i[0] if i.shape[0] not in [1, 3] else i for i in img],
                caption=[i for i in range(img.shape[0])]
            )
            
            img = batch['context_he'][:self.log_n_samples].clone().detach().cpu()
            img -= img.min()
            img /= img.max()
            trainer.logger.log_image(
                key=f"{key}/context_he",
                images=[i[0] if i.shape[0] not in [1, 3] else i for i in img],
                caption=[i for i in range(img.shape[0])]
            )
            
            img = batch['masks'][:self.log_n_samples].clone().detach().cpu()
            img = img.sum(1) > 0
            new = torch.zeros_like(img, dtype=torch.float32)
            new[img] = 1.
            trainer.logger.log_image(
                key=f"{key}/voxels",
                images=[i for i in new],
                caption=[i for i in range(new.shape[0])]
            )

            gene_idxs = [pl_module.model.genes.index(g) for g in self.plot_genes]
            img = construct_tile_expression(batch['exp'][:, :, gene_idxs], batch['masks'], batch['n_voxels'])
            img = img[:self.log_n_samples]
            img = torch.tensor(rearrange(img, 'b h w c -> c b 1 h w'))
            trainer.logger.log_image(
                key=f"{key}/exp_groundtruth",
                images=[i for i in img],
                caption=[g for g in self.plot_genes]
            )

            gene_idxs = [pl_module.model.genes.index(g) for g in self.plot_genes]
            img = construct_tile_expression(outputs['result']['exp'][:, :, gene_idxs], batch['masks'], batch['n_voxels'])
            img = img[:self.log_n_samples]
            img = torch.tensor(rearrange(img, 'b h w c -> c b 1 h w'))
            trainer.logger.log_image(
                key=f"{key}/exp_prediction",
                images=[i for i in img],
                caption=[g for g in self.plot_genes]
            )  
  
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.log_every == 0 and batch_idx==0:
            self.log_epoch('train', trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.log_every == 0 and batch_idx==0:
            self.log_epoch('val', trainer, pl_module, outputs, batch, batch_idx)


class MetageneAutoencoder(nn.Module):
    def __init__(
        self,
        genes,
        n_metagenes = 10,
        in_channels = 3,
        out_channels = 64,
        decoder_channels = (128, 64, 32, 16, 8),
        context_decoder_channels = (128, 64, 32, 16, 8),
        he_scaler = .1,
        kl_scaler = .001,
        exp_scaler = 1.
    ):
        super().__init__()
        
        self.genes = genes
        self.n_genes = len(genes)

        
        self.he_scaler = he_scaler
        self.kl_scaler = kl_scaler
        self.exp_scaler = exp_scaler
        
        self.unet = Unet(backbone='resnet34',
                         decoder_channels=decoder_channels,
                         in_chans=in_channels,
                         num_classes=out_channels)
        
        self.context_unet = Unet(backbone='resnet34',
                         decoder_channels=context_decoder_channels,
                         in_chans=in_channels,
                         num_classes=out_channels)
        
        self.post_unet_conv = nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels,
                                        kernel_size=1)

        # latent mu and var
        self.latent_mu = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                   kernel_size=1)
        self.latent_var = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=1)
        self.latent_norm = nn.BatchNorm2d(out_channels) # try changing momentum
        
        self.n_metagenes = n_metagenes
        self.metagenes = torch.nn.Parameter(torch.rand(self.n_metagenes, self.n_genes))
        self.scale_factors = torch.nn.Parameter(torch.rand(self.n_genes))
        self.p = torch.nn.Parameter(torch.rand(self.n_genes))
        
        self.post_decode_he = torch.nn.Conv2d(out_channels, 3, 1)
        self.post_decode_he = torch.nn.Sequential( # we need to go up 3x
            DecoderBlock(out_channels, out_channels, scale_factor=2.0),
            DecoderBlock(out_channels, out_channels // 2, scale_factor=2.0),
            DecoderBlock(out_channels // 2, out_channels // 2, scale_factor=2.0),
            torch.nn.Conv2d(out_channels // 2, 3, kernel_size=(1, 1))
        )
        self.post_decode_exp = torch.nn.Conv2d(out_channels, self.n_metagenes, 1)
        
        self.he_loss = torch.nn.MSELoss()

    def _kl_divergence(self, z, mu, std):
        # lightning imp.
        # Monte carlo KL divergence
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)

        return kl

    def encode(self, x, x_context, use_means=False):
        x_encoded = self.unet(x)
        x_context_encoded = self.context_unet(x)
        
        x_encoded = torch.concat((x_encoded, x_context_encoded), dim=1)
        x_encoded = self.post_unet_conv(x_encoded)
        
        x_encoded = self.latent_norm(x_encoded)
        
        mu, log_var = self.latent_mu(x_encoded), self.latent_var(x_encoded)
        
        # sample z from parameterized distributions
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        # get our latent
        if use_means:
            z = mu
        else:
            z = q.rsample()

        return z, mu, std
    
    def calculate_loss(self, he_true, exp_true, result):
        exp_loss = torch.mean(-result['nb'].log_prob(exp_true))
        
        kl_loss = torch.mean(self._kl_divergence(result['z'], result['z_mu'], result['z_std']))
        
        he_loss = torch.mean(self.he_loss(he_true, result['he']))
        
        return {
            'overall_loss': exp_loss * self.exp_scaler + kl_loss * self.kl_scaler + he_loss * self.he_scaler,
            'exp_loss': exp_loss,
            'kl_loss': kl_loss,
            'he_loss': he_loss
        }

    def reconstruct_expression(self, dec, masks=None, voxel_idxs=None, reduce_to_voxel=True,
                               gene_idxs=None):
        x = self.post_decode_exp(dec) # (b c h w)
        
        if reduce_to_voxel:
            x = reduce_to_voxel_level(x, masks) # (b, v, m)
        else:  
            x = rearrange(x, 'b c h w -> b h w c')
            
        if gene_idxs is not None:
            metagenes = self.metagenes[:, gene_idxs]
            scale_factors = self.scale_factors[gene_idxs]
            p = self.p[gene_idxs]
        else:
            metagenes = self.metagenes
            scale_factors = self.scale_factors
            p = self.p
        
        r = x @ metagenes
        r = r * scale_factors
        r = F.softplus(r)
        
        p = torch.sigmoid(p)
        
        if reduce_to_voxel:
            p = rearrange(p, 'c -> 1 1 c')
            r = mask_nb_params(r, voxel_idxs)
        else:
            p = rearrange(p, 'c -> 1 1 1 c')
            
        r += .00000001
            
        nb = torch.distributions.NegativeBinomial(r, p)
        
        return {
            'r': r,
            'p': p,
            'exp': nb.mean,
            'nb': nb,
            'metagene_activity': x # (b v m)
        }
    
    def reconstruct_he(self, dec):
        he = self.post_decode_he(dec)
        return he

    def forward(self, x, x_context, masks=None, voxel_idxs=None, reduce_to_voxel=True,
                use_means=False, gene_idxs=None):
        z, z_mu, z_std = self.encode(x, x_context, use_means=use_means)

        he = self.reconstruct_he(z)
        
        exp_result = self.reconstruct_expression(
            z, masks=masks, voxel_idxs=voxel_idxs, reduce_to_voxel=reduce_to_voxel,
            gene_idxs=gene_idxs)
        
        result = {
            'z': z,
            'z_mu': z_mu,
            'z_std': z_std,
            'he': he,
        }
        result.update(exp_result)

        return result


class LitMetageneAutoencoder(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        
        self.model = model
        self.lr = config['training']['lr']
        self.config = config # saving config so we can load from checkpoint

        self.set_prediction_genes(config['genes'])
        
        self.save_hyperparameters(ignore=['model'])

    @staticmethod
    def load_from_checkpoint(checkpoint_path):
        """Need to overwrite default method due to model pickling issue"""
        checkpoint = torch.load(checkpoint_path)
        config = checkpoint['hyper_parameters']['config']
        m = MetageneAutoencoder(
            config['genes'],
            n_metagenes=config['n_metagenes'],
            out_channels=config['latent_dim'],
            decoder_channels=config['decoder_channels'],
            context_decoder_channels=config['decoder_channels'],
            he_scaler=config['he_scaler'],
            kl_scaler=config['kl_scaler'],
            exp_scaler=config['exp_scaler'],
        )
        d = {re.sub(r'^model.(.*)$', r'\1', k):v for k, v in checkpoint['state_dict'].items()}
        m.load_state_dict(d)

        return LitMetageneAutoencoder(m, config)

    def set_prediction_genes(self, genes):
        a = set(genes)
        b = set(self.model.genes)
        missing = a - b
        if len(missing) != 0:
            raise RuntimeError(f'The following genes were not in the training dataset and cannot be predicted by the model: {missing}')

        gene_idxs = [self.model.genes.index(g) for g in genes]
        self.prediction_genes = genes
        self.prediction_gene_idxs = gene_idxs

    def training_step(self, batch, batch_idx):
        x, x_context, masks, voxel_idxs, exp = (
            batch['he'], batch['context_he'], batch['masks'], batch['voxel_idxs'], batch['exp'])
        result = self.model(x, x_context, masks=masks, voxel_idxs=voxel_idxs)
        losses = self.model.calculate_loss(x, exp, result)
        losses = {f'train/{k}':v for k, v in losses.items()}
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        losses['loss'] = losses['train/overall_loss']
        losses['result'] = result
        
        return losses
    
    def validation_step(self, batch, batch_idx):
        x, x_context, masks, voxel_idxs, exp = (
            batch['he'], batch['context_he'], batch['masks'], batch['voxel_idxs'], batch['exp'])
        result = self.model(x, x_context, masks=masks, voxel_idxs=voxel_idxs)
        losses = self.model.calculate_loss(x, exp, result)
        losses = {f'val/{k}':v for k, v in losses.items()}
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        losses['result'] = result 
        
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, batch):
        x, x_context = batch['he'], batch['context_he']
        result = self.model(
            x, x_context,
            reduce_to_voxel=False, use_means=True, gene_idxs=self.prediction_gene_idxs)
        
        return result