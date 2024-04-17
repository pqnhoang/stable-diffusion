import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init_(self):
        super.__init__(
            # (batch_size, channels, H, W) -> (batch_size, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # (batch_size, 128, H, W) -> (batch_size, 128, H, W)
            VAE_ResidualBlock(128, 128),

            # (batch_size, 128, H, W) -> (batch_size, 128, H, W)
            VAE_ResidualBlock(128, 128),

            # (batch_size, 128, H, W) -> (batch_size, 128, H/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (batch_size, 128, H/2, W/2) -> (batch_size, 256, H/2, W/2)
            VAE_ResidualBlock(128, 256),

            # (batch_size, 256, H/2, W/2) -> (batch_size, 256, H/2, W/2)
            VAE_ResidualBlock(256, 256),

            # (batch_size, 256, H/2, W/2) -> (batch_size, 256, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (batch_size, 256, H/4, W/4) -> (batch_size, 512, H/4, W/4)
            VAE_ResidualBlock(256, 512),

            # (batch_size, 512, H/4, W/4) -> (batch_size, 512, H/4, W/4)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, H/4, W/4) -> (batch_size, 512, H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, H/8, W/8) -> (batch_size, 512, H/8, W/8)
            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, H/8, W/8) -> (batch_size, 512, H/8, W/8)
            nn.GroupNorm(32, 512),

            nn.SiLU(),

            # (batch_size, 512, H/8, W/8) -> (batch_size, 8, H/8, W/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (batch_size, 8, H/8, W/8) -> (batch_size, 8, H/8, W/8)
            nn.Conv2d(8, 8, kernel_size=3, padding=1)
        )
    
    def forward(self, x : torch.Tensor, noise : torch.Tensor) -> torch.Tensor:
        # x = (batch_size, channels, H, W)
        # noise = (batch_size, out_channels, H/8, W/8)
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # (batch_size, 8, H/8, W/8) -> (batch_size, 4, H/8, W/8) x 2
        mean, log_variance = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_variance, -30, 20)

        # (batch_size, 4, H/8, W/8) -> (batch_size, 4, H/8, W/8)
        variance = log_variance.exp()

        # (batch_size, 4, H/8, W/8) -> (batch_size, 4, H/8, W/8)
        stdev = variance.sqrt()

        #Z=N(0,1) -> N(mean, variance) =X?
        x = mean + noise*stdev
        #scale by const

        x *= 0.18215

        return x
