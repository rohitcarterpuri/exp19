"""Generator and Discriminator models for GAN"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generator network for creating fake MNIST digits"""
    
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28), features=256):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        
        # Calculate output size after each transpose conv
        self.model = nn.Sequential(
            # Input: latent_dim x 1 x 1
            self._block(latent_dim, features * 4, 4, 1, 0),  # 4x4
            self._block(features * 4, features * 2, 4, 2, 1),  # 8x8
            self._block(features * 2, features, 4, 2, 1),  # 14x14
            nn.ConvTranspose2d(features, img_shape[0], 4, 2, 1),  # 28x28
            nn.Tanh()
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, z):
        # Reshape latent vector to (batch, latent_dim, 1, 1)
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    """Discriminator network to distinguish real from fake digits"""
    
    def __init__(self, img_shape=(1, 28, 28), features=128):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0], features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            self._block(features, features * 2, 4, 2, 1),
            self._block(features * 2, features * 4, 4, 2, 1),
            self._block(features * 4, features * 8, 4, 2, 1),
            
            nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1)


def initialize_weights(model):
    """Initialize weights for better training"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
