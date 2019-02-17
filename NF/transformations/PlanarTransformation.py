import torch
from torch import nn
from torch.nn import functional as F

from .Transformation import Transformation

class PlanarTransformation(Transformation):

    """Planar Transformation"""

    def __init__(self, dim:int, training:bool=True):

        self.dim = dim
        self.h = nn.Tanh()
        self.training = training

    def get_num_params(self):
        """Get the number of parameters this Transformation requires"""
        return self.dim * 2 + 1

    def transform(self, zi, params):
        """Transform the latent variables using this transformation

        Args
        zi     -- variable which will be transfomed
        params -- parameters for this Transformation

        Returns transformed variable zi' of the same shape as the input zi
        """
        w, u, b = self.unwrap_params(params)
        return zi + u * self.h(F.linear(zi, w, b))

    def unwrap_params(self, params):
        """Convert an array with params into vectors of this Transformation
        Args
        params -- array with parameters
        z      -- latent variable

        Returns w, u, b
        """
        w = params[:self.dim]
        u = params[self.dim:-1]
        b = params[-1]
        if torch.dot(w, u) < -1:
            dotwu = torch.dot(w, u)
            u = u + (-1 + torch.log(1 + torch.exp(dotwu)) - dotwu) \
                            * w / torch.norm(w)
        return w.unsqueeze(0), u.unsqueeze(0), b

    def h_deriv(self, x):
        """Derivative of the activation function"""
        ff = self.h(x)
        return 1 - ff * ff

    def psi(self, z, w, u, b):
        return self.h_deriv(F.linear(z, w, b)) * w

    def det(self, z, params):
        """Compute the jacobian of this transformation given zi and parameters"""
        w, u, b = self.unwrap_params(params)
        return (1 + torch.mm(self.psi(z, w, u, b), u.t())).abs()
