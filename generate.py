"""Generate new handwritten digits using trained generator"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image

from config import Config
from models import Generator


def generate_samples(generator_path, num_samples=64, latent_dim=100, 
                     save_grid=True, save_individual=False, device='cpu'):
    """Generate new digit images using trained generator"""
    
    # Load generator
    generator = Generator(latent_dim=latent_dim, img_shape=(1, 28, 28))
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.to(device)
    generator.eval()
    
    print(f'Generator loaded from {generator_path}')
    
    # Generate samples
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        generated_imgs = generator(z).cpu()
        
        # Denormalize from [-1, 1] to [0, 1]
        generated_imgs = (generated_imgs + 1) / 2.0
    
    # Create directory for outputs
    os.makedirs('generated_samples', exist_ok=True)
    
    # Save grid of images
    if save_grid:
        save_image(generated_imgs, 'generated_samples/grid.png', nrow=8, padding=2)
        print(f'Grid saved to generated_samples/grid.png')
    
    # Save individual images
    if save_individual:
        for i, img in enumerate(generated_imgs):
            save_image(img, f'generated_samples/sample_{i}.png')
        print(f'{num_samples} individual samples saved to generated_samples/')
    
    # Display samples
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(generated_imgs[i].squeeze(), cmap='gray')
            ax.axis('off')
    
    plt.suptitle('Generated Handwritten Digits', fontsize=16)
    plt.tight_layout()
    plt.savefig('generated_samples/display.png', dpi=150)
    plt.show()
    
    print('\nGeneration completed!')
    return generated_imgs


def generate_interpolation(generator_path, num_steps=10, latent_dim=100, device='cpu'):
    """Generate interpolations between two random noise vectors"""
    
    generator = Generator(latent_dim=latent_dim, img_shape=(1, 28, 28))
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.to(device)
    generator.eval()
    
    with torch.no_grad():
        # Generate two random latent vectors
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)
        
        # Create interpolations
        alphas = np.linspace(0, 1, num_steps)
        interpolated_imgs = []
        
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            img = generator(z).cpu()
            img = (img + 1) / 2.0
            interpolated_imgs.append(img)
        
        # Display interpolation
        fig, axes = plt.subplots(1, num_steps, figsize=(15, 2))
        for i, img in enumerate(interpolated_imgs):
            axes[i].imshow(img.squeeze(), cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'{alphas[i]:.1f}')
        
        plt.suptitle('Latent Space Interpolation', fontsize=14)
        plt.tight_layout()
        plt.savefig('generated_samples/interpolation.png', dpi=150)
        plt.show()
        print('Interpolation saved to generated_samples/interpolation.png')


if __name__ == '__main__':
    # Configuration
    CHECKPOINT_PATH = './checkpoints/generator_final.pth'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LATENT_DIM = 100
    
    # Generate samples
    generate_samples(
        generator_path=CHECKPOINT_PATH,
        num_samples=64,
        latent_dim=LATENT_DIM,
        save_grid=True,
        save_individual=False,
        device=DEVICE
    )
    
    # Generate interpolation (optional)
    generate_interpolation(
        generator_path=CHECKPOINT_PATH,
        num_steps=10,
        latent_dim=LATENT_DIM,
        device=DEVICE
    )
