import torch
from torch import nn
from torch.nn import functional as F

from .Transformation import Transformation

class SylvesterTransformation(Transformation):
    
    """Base class for all Sylvester tranformations"""

    def __init__(self, dim:int, num_hidden:int=1, device:str='cuda', training:bool=True):
        
        self.dim = dim
        self.h = nn.Tanh()
        self.training = training
        self.num_hidden = num_hidden
        self.device = device
        self.eye_M = torch.eye(num_hidden, device=device)
        
    def get_num_params(self):
        raise NotImplementedError()
    
    def transform(self, z, params):
        """Transform the latent variables using this transformation
        
        Args
        zi     -- variable which will be transfomed
        params -- parameters for this Transformation

        Returns transformed variable zi' of the same shape as the input zi
        """
        Q, R, R_hat, b = self.unwrap_params( params, z )
        
        return z + Q.mm(R).mm(self.h( F.linear(z, R_hat.mm(Q.t()), b) ).t()).t()
    
    def unwrap_params(self, params, z):
        raise NotImplementedError()
    
    def h_deriv(self, x):
        """Derivative of the activation function"""
        ff = self.h( x )
        return 1 - ff * ff
    
    def det_triag(self, mat):
        """Determinant of an upper or lower triangular matix"""
        return torch.cumprod(torch.diagonal(mat, dim1=-2, dim2=-1), -1)
    
    # det ( I_M + diag ( h′( R_hat Q^T z + b ) ) R_hat R )
    def det(self, z, params):
        """Compute the jacobian of this transformation given zi and parameters"""
        Q, R, R_hat, b = self.unwrap_params( params, z )
        psi = self.h_deriv(F.linear(z, R_hat.mm(Q.t()), b))
        
        # a workaround for batched matrices
        psi_mat = torch.zeros((psi.shape[0], psi.shape[1], psi.shape[1]),device=self.device)
        psi_mat.as_strided(psi.size(), [psi_mat.stride(0), psi_mat.size(2) + 1]).copy_(psi)
        
        psi_mat_RR = torch.matmul(psi_mat, R_hat.mm(R))
        return self.det_triag( self.eye_M.repeat(psi_mat.shape[0], 1, 1) - psi_mat_RR ).abs()

class OrthoSylvesterTransformation(SylvesterTransformation):

    """SylvesterTransformation where column-wise orthogonality of Q is ensured
        by an iterative procedure.
    """
    
    def __init__(self, dim:int, num_hidden:int=1, device:str='cuda', training:bool=True):
        
        super().__init__(dim, num_hidden, device, training)
        
    def get_num_params(self):
        """Get the number of parameters this Transformation requires"""
        # Q, R, R_hat, b
        return self.dim * self.num_hidden + self.num_hidden * self.num_hidden + self.num_hidden + self.num_hidden
    
    def unwrap_params(self, params, z):
        """Convert an array with params into vectors and matrices of this Transformation
        Args
        params -- array with parameters
        z      -- latent variable
        
        Returns Q, R and R_hat matrices and b bias vector
        """
        Q = params[:self.dim * self.num_hidden].reshape((self.dim, self.num_hidden))
        RR = params[self.dim * self.num_hidden : (self.dim + self.num_hidden) * self.num_hidden].reshape((self.num_hidden, self.num_hidden))
        R = torch.triu(RR)
        
        R_hat = torch.tril(RR).t()
        v = params[-2 * self.num_hidden : -self.num_hidden]
        mask = torch.diag(torch.ones_like(v))
        R_hat = mask * torch.diag(v) + (1. - mask) * R_hat

        b = params[-self.num_hidden:]
        Q = self.make_ortho( Q, 1e-1 )
        return Q, R, R_hat, b
    
    def make_ortho(self, Q, eps):
        """Iteratively convert Q into column-wise orthogonal matrix"""
        # TODO: how to make sure that the convergence condition is fulfilled?
        QQ = Q.t().mm(Q)

        # check convergence condition
        _, s, _ = torch.svd(QQ - self.eye_M)
        if s[0] > 1:
            print( "[WARN] Q will not converge to orthogonal" )
            return Q
        
        # while not converged
        while torch.norm(QQ - self.eye_M) > eps:
            Q = Q.mm( self.eye_M + (self.eye_M - QQ ) / 2 )
        return Q

class HouseholderSylvesterTransformation(SylvesterTransformation):

    """SylvesterTransformation where column-wise orthogonality of Q is ensured
        by a Householder operation.
    """
    
    def __init__(self, dim:int, device:str='cuda', training:bool=True):
        
        super().__init__(dim, dim, device, training)
        
    def get_num_params(self):
        """Get the number of parameters this Transformation requires"""
        # Q, R, R_hat, b
        return self.dim + self.num_hidden * self.num_hidden + self.num_hidden + self.num_hidden
    
    def unwrap_params(self, params, z):
        """Convert an array with params into vectors and matrices of this Transformation
        Args
        params -- array with parameters
        z      -- latent variable
        
        Returns Q, R and R_hat matrices and b bias vector
        """
        v = params[:self.dim]
        Q = self.make_Q_Householder(v, z)
        
        RR = params[self.dim * self.num_hidden : (self.dim + self.num_hidden) * self.num_hidden].reshape((self.num_hidden, self.num_hidden))
        R = torch.triu(RR)
        
        R_hat = torch.tril(RR).t()
        v = params[-2 * self.num_hidden : -self.num_hidden]
        mask = torch.diag(torch.ones_like(v))
        R_hat = mask * torch.diag(v) + (1. - mask) * R_hat

        b = params[-self.num_hidden:]
        return Q, R, R_hat, b
    
    def make_Q_Householder(self, v, z):
        """Create column-wise orthogonal matrix Q using Householder operation"""
        # TODO: implement
        raise NotImplementedError()

class TriagSylvesterTransformation(SylvesterTransformation):
    
    """SylvesterTransformation where Q is either an identity or a reverse
        permutation matrix. In this implementation Q is always identity,
        and the reverse permutation is achieved by alternating R and R_hat
        between upper and lower triangular forms
    """

    def __init__(self, dim:int, num_hidden:int=1, permute:bool=False, device:str='cuda', training:bool=True):
        
        super().__init__(dim, num_hidden, device, training)
        self.permute = permute
        self.Q = torch.eye(self.dim, self.num_hidden, device=device)
        
    def get_num_params(self):
        """Get the number of parameters this Transformation requires"""
        # Q, R, R_hat, b
        return self.num_hidden * self.num_hidden + self.num_hidden + self.num_hidden
    
    def unwrap_params(self, params, z):
        """Convert an array with params into vectors and matrices of this Transformation
        Args
        params -- array with parameters
        z      -- latent variable
        
        Returns Q, R and R_hat matrices and b bias vector
        """
        RR = params[:self.num_hidden * self.num_hidden].reshape((self.num_hidden, self.num_hidden))
        R = torch.triu(RR)
        
        R_hat = torch.tril(RR)
        if self.permute:
            R = R.t()
        else:
            R_hat = R_hat.t()
            
        v = params[-2 * self.num_hidden : -self.num_hidden]
        mask = torch.diag(torch.ones_like(v))
        R_hat = mask * torch.diag(v) + (1. - mask) * R_hat

        b = params[-self.num_hidden:]
        return self.Q, R, R_hat, b
