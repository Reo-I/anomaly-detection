import torch
from torch import nn
import torchvision

class VAE(nn.Module):
    def __init__(self, latent_dim, device):
        super(VAE, self).__init__()
        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32, 3, stride=2, padding=1),  # b, 64, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # b, 128, 16, 16
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        #self.linear1 = nn.Linear(128*self.H*self.W, 64)

        self.mean = nn.Sequential(
            nn.Linear(13312,latent_dim),# b, 64 ==> b, latent_dim
            )
        
        self.var = nn.Sequential(
            nn.Linear(13312,latent_dim),# b, 64 ==> b, latent_dim
            )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,13312),# b, latent_dim ==> b, 13312
            nn.BatchNorm1d(13312),
            nn.ReLU(),
            )

        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1,padding = 0),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.UpsamplingNearest2d(size = [25, 16]),
            )
        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=1,padding = 0),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.UpsamplingNearest2d(size = [50, 32]),
            )
        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 3, stride=1,padding = 0),  # b, 3, 32, 32
            nn.BatchNorm2d(1),
            nn.UpsamplingNearest2d(size = [100, 64]),
            nn.Sigmoid()
            )
        
    def _sample_z(self, mean, var):

      epsilon = torch.randn(mean.shape, device=self.device)
      return mean + epsilon*torch.exp(0.5 * var)
    
    #Encoder
    def _encoder(self, x):
        x = self.conv1(x)     
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        mean = self.mean(x)
        var = self.var(x)
        return mean,var
    #Decoder
    def _decoder(self, z):
        z = self.decoder(z)
        z = z.view(-1,128,13,8)
        x = self.convTrans1(z)
        x = self.convTrans2(x)
        x = self.convTrans3(x)
        return x
    
    def forward(self, x):
        mean,var = self._encoder(x) #mean and log_variance
        z = self._sample_z(mean, var) 
        x = self._decoder(z)
        return x,mean,var,z
    
    def loss(self, x):
      mean, var = self._encoder(x) #\sigma^2
      
      KL = 0.5 * torch.mean(1 + var- mean**2 - torch.exp(var)) #KLDivvergence
      
      z = self._sample_z(mean, var) 
      y = self._decoder(z) 
      delta = 1e-7 

      reconstruction = torch.mean(x * torch.log(y+delta) + (1 - x) * torch.log(1 - y +delta)) #reconstruction loss
      lower_bound = [KL, reconstruction]   

      return -sum(lower_bound) ,y,mean,var,z