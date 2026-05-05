"""Utility functions for data loading, visualization, and logging"""

import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def get_dataloader(batch_size=64, data_path='./data'):
    """Load MNIST dataset and return dataloader"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    dataset = datasets.MNIST(
        root=data_path, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader


def save_generated_images(generator, epoch, batch_idx, latent_dim, device, 
                          sample_dir='./samples', n_samples=64, figsize=(8, 8)):
    """Generate and save a grid of sample images"""
    
    generator.eval()
    os.makedirs(sample_dir, exist_ok=True)
    
    with torch.no_grad():
        # Generate random noise
        z = torch.randn(n_samples, latent_dim).to(device)
        samples = generator(z).cpu()
        
        # Denormalize images from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2.0
        
        # Create grid
        grid = torchvision.utils.make_grid(samples, nrow=8, padding=2, normalize=False)
        
        # Convert to numpy and plot
        plt.figure(figsize=figsize)
        plt.imshow(np.transpose(grid, (1, 2, 0)), cmap='gray')
        plt.axis('off')
        plt.title(f'Epoch {epoch}, Batch {batch_idx}')
        
        # Save figure
        filename = f'{sample_dir}/epoch_{epoch}_batch_{batch_idx}.png'
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    generator.train()


def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, 
                   epoch, loss_history, checkpoint_dir='./checkpoints'):
    """Save model checkpoint"""
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'generator_optimizer': g_optimizer.state_dict(),
        'discriminator_optimizer': d_optimizer.state_dict(),
        'loss_history': loss_history,
        'latent_dim': generator.latent_dim
    }
    
    checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')


def load_checkpoint(checkpoint_path, generator, discriminator, g_optimizer, d_optimizer, device):
    """Load model checkpoint"""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['generator_optimizer'])
    d_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
    
    epoch = checkpoint['epoch']
    loss_history = checkpoint['loss_history']
    
    print(f'Checkpoint loaded from epoch {epoch}')
    
    return epoch, loss_history


def plot_loss_curves(loss_history, save_path='./loss_curves.png'):
    """Plot generator and discriminator loss curves"""
    
    plt.figure(figsize=(10, 5))
    
    epochs = range(1, len(loss_history['g_losses']) + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history['g_losses'], label='Generator Loss', color='blue')
    plt.plot(epochs, loss_history['d_losses'], label='Discriminator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss_history['g_losses'], label='Generator Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss (Zoomed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f'Loss curves saved to {save_path}')


def set_seed(seed):
    """Set random seeds for reproducibility"""
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seek_all(seed)
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
