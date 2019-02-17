from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def plot_image(tensor, title, figsize=None, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Plot tensor as an image"""
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.permute(1, 2, 0).cpu().numpy()
    _, fig_width, _ = ndarr.shape
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(ndarr)
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_visible(False)
    cur_axes.axes.get_yaxis().set_visible(False)
    plt.title(title)
    plt.show()
    plt.close()
