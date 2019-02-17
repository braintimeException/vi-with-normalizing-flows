import torch

from NF.autoencoders import VariationalAutoencoderNormalizingFlow
from NF.losses import VAE_loss, VAENF_loss
from NF.utils import plot_image

def train(epoch, model, optimizer, train_loader, device):
    """Train VariationalAutoencoder for one epoch
    Args
    epoch  -- the current epoch
    model  -- isinstance of VariationalAutoencoder subclass
    optimizer  -- pytorch optimizer
    train_loader -- pytorch Dataloader for training
    device -- 'cpu' of 'cuda'
    """
    model.train()
    is_model_with_nf = isinstance(model, VariationalAutoencoderNormalizingFlow)
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        if len(data) == 2:
            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        if is_model_with_nf:
            sum_log_det = model.flow.get_sum_log_det()
            loss = VAENF_loss(recon_batch, data, mu, logvar, sum_log_det)
        else:
            loss = VAE_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch, model, test_loader, device):
    """Test the performance of the VariationalAutoencoder
    Args
    epoch  -- the current epoch
    model  -- isinstance of VariationalAutoencoder subclass
    train_loader -- pytorch Dataloader for training
    device -- 'cpu' of 'cuda'
    """
    model.eval()
    is_model_with_nf = isinstance(model, VariationalAutoencoderNormalizingFlow)
    plot_title = 'VAE{} reconstruction K={} epoch={}'.format(
        'NF' if is_model_with_nf else '',
        model.flow.K if is_model_with_nf else 0,
        epoch)
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if len(data) == 2:
                data = data[0]
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            if is_model_with_nf:
                sum_log_det = model.flow.get_sum_log_det()
                loss = VAENF_loss(recon_batch, data, mu, logvar, sum_log_det)
            else:
                loss = VAE_loss(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(*(data.shape))[:n]])
                plot_image(comparison.cpu(),
                           plot_title, figsize=[6, 6], padding=0, nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
