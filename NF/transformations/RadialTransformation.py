import torch

from .Transformation import Transformation


class RadialTransformation(Transformation):

    """Radial Transformation"""

    def __init__(self, dim: int, training: bool=True):

        self.dim = dim
        self.training = training

    def get_num_params(self):
        """Get the number of parameters this Transformation requires"""
        return self.dim + 2

    def unwrap_params(self, params):
        """Convert an array with params into vectors of this Transformation
        Args
        params -- array with parameters
        z      -- latent variable

        Returns z0, alpha, beta
        """
        z0 = params[:self.dim]
        alpha = params[-2]
        beta = params[-1]
        if beta < -alpha:
            beta = -alpha + torch.log( 1 + torch.exp( beta ) )
        return z0.unsqueeze(0), alpha, beta

    def transform(self, zi, params):
        """Transform the latent variables using this transformation

        Args
        zi     -- variable which will be transfomed
        params -- parameters for this Transformation

        Returns transformed variable zi' of the same shape as the input zi
        """
        z0, alpha, beta = self.unwrap_params(params)
        r = torch.norm((zi - z0), p=2, dim=1, keepdim=True)
        return zi + beta * (self.h(r, alpha) * (zi - z0))

    def h(self, r, alpha):
        """Radial function"""
        return 1 / (alpha + r)

    def h_deriv(self, r, alpha):
        """Derivative of the radial function"""
        ff = self.h(r, alpha)
        return - ff * ff

    def det(self, zi, params):
        """Compute the jacobian of this transformation given zi and parameters"""
        z0, alpha, beta = self.unwrap_params(params)
        r = torch.norm((zi - z0), p=2, dim=1, keepdim=True)
        tmp = 1 + beta * self.h(r, alpha)
        return torch.clamp(tmp.pow(self.dim - 1) *
                           (tmp + beta * self.h_deriv(r, alpha) * r),
                           min=1e-7)
