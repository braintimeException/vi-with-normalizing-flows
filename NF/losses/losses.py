import torch
from torch.nn import functional as F

def reconstruction_loss(recon_x, x):
    """Loss based on binary cross entropy between original and reconstructed input

    Args
    recon_x  -- reconstructed input
    x        -- original input

    Returns binary cross entropy between x and x_recon
    """
    return F.binary_cross_entropy(recon_x, x.view(-1, recon_x.shape[-1]), reduction='sum')

def kld_loss(mu, logvar):
    """Loss based on KL-divergence"""
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def VAE_loss(recon_x, x, mu, logvar):
    """Variational Autoencoder loss
    Args:
    recon_x   -- reconstruction of x
    x         -- original x
    mu        -- amortized mean
    logvar    -- amortized log variance
    """
    return reconstruction_loss(recon_x, x) + kld_loss(mu, logvar) / x.size(0)

def VAENF_loss(recon_x, x, mu, logvar, sum_log_det):
    """Variational Autoencoder with Normalizing Flow loss
    Args:
    recon_x      -- reconstruction of x
    x            -- original x
    mu           -- amortized mean
    logvar       -- amortized log variance
    sum_log_det  -- sum of log jacobians
    """
    return VAE_loss(recon_x, x, mu, logvar) - sum_log_det.mean()
