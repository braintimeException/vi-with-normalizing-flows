import torch

class Transformation:

    """Base class of all normalizing flow transformations"""
    
    def __init__(self):
        self.training = None
        self.log_det = None
        
    @property
    def training(self):
        return self._training
    
    @training.setter
    def training(self, enable:bool):
        """When training is enabled, the jacobians are recorded"""
        if not enable:
            self.log_det = None
        self._training = enable

    def transform(self, zi, params):
        """Transform the latent variables using this transformation
        
        Args
        zi     -- variable which will be transfomed
        params -- parameters for this Transformation

        Returns transformed variable zi' of the same shape as the input zi
        """
        raise NotImplementedError()
    
    def det(self, zi, params):
        """Compute the jacobian of this transformation given zi and parameters"""
        raise NotImplementedError()
    
    def forward(self, zi, params):
        """Forward pass applies this Transformation with parameters on zi"""
        if self.training:
            self.log_det = torch.log( self.det( zi, params ).squeeze() + 1e-7 )
        return self.transform( zi, params )
    
    def get_num_params(self):
        """Get the number of parameters this Transformation requires"""
        return 0
