import torch
from torch import nn
from torch.nn import functional as F

from NF.flows.NormalizingFlow import NormalizingFlow

class VAE(nn.Module):


    def __init__(self, input_dim, num_latent):
        super().__init__()

        self.input_dim = input_dim
        self.num_latent = num_latent
        
        ############
        #  Encoder
        ############

        # first fully connected layer
        self.fc1 = nn.Linear(input_dim, 400)

        # parallel layers
        # encode mean
        self.fc21_mean = nn.Linear(400, num_latent)
        # encode variance
        self.fc22_var = nn.Linear(400, num_latent)

        ############
        #  Decoder
        ############

        # two fully connected layer, i.e. simplified reverse of the encoder
        self.fc3 = nn.Linear(num_latent, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        """Convert input into parameters
        
        Args
        x -- input tensor
        """
        raise NotImplementedError()

    def reparameterize(self, mu, logvar):
        """Use mean and variance to generate latent variables z
        
        Args
        mu      -- mean from encode()
        logvar  -- log variance from encode()

        Returns latent variables z
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        """Transform latent variables back to original space
        Reconstructs the input from latent variables

        Args
        z -- latent variables
        Returns reconstructed input of the same shape as original input
        """
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        """Forward pass
        Transforms the input into latent variables and reconstructs the input

        Args
        x  -- input tensor

        Returns recontructed input along with mean and variance of the latent variables
        """
        raise NotImplementedError()


class VariationalAutoencoderOriginal(VAE):


    def __init__(self, input_dim, num_latent):
        super().__init__(input_dim, num_latent)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc21_mean(h), self.fc22_var(h)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VariationalAutoencoderNormalizingFlow(VAE):


    def __init__(self, input_dim, num_latent, flow_transform, flow_latent, flow_len):
        super().__init__(input_dim, num_latent)

        # normalizing flow
        self.flow = NormalizingFlow(flow_transform, flow_latent, flow_len)

        # encode flow parameters ( parallel to mean and var )
        self.fc23_flow = nn.Linear(400, self.flow.nParams * flow_len)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        # returns mean, logvar and flow params
        return (self.fc21_mean(h), self.fc22_var(h),
                self.fc23_flow(h).mean(dim=0).chunk(self.flow.K, dim=0))

    def forward(self, x):
        mu, logvar, params = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        z = self.flow.forward(z, params)
        return self.decode(z), mu, logvar
