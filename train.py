"""Main training script for GAN on MNIST"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import Config
from models import Generator, Discriminator, initialize_weights
from utils import (
    get_dataloader, save_generated_images, save_checkpoint, 
    plot_loss_curves, set_seed
)


def train():
    # Load configuration
    config = Config()
    
    # Set device
    if config.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    print(f'Using device: {device}')
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)
    
    # Load data
    print('Loading MNIST dataset...')
    dataloader = get_dataloader(config.batch_size, config.data_path)
    print(f'Dataset loaded: {len(dataloader.dataset)} images')
    
    # Initialize models
    generator = Generator(
        latent_dim=config.latent_dim,
        img_shape=config.img_shape,
        features=config.generator_features
    ).to(device)
    
    discriminator = Discriminator(
        img_shape=config.img_shape,
        features=config.discriminator_features
    ).to(device)
    
    # Initialize weights
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    
    print(generator)
    print(discriminator)
    
    # Loss function
    adversarial_loss = nn.BCELoss()
    
    # Optimizers
    g_optimizer = optim.Adam(
        generator.parameters(), 
        lr=config.lr, 
        betas=(config.beta1, config.beta2)
    )
    
    d_optimizer = optim.Adam(
        discriminator.parameters(), 
        lr=config.lr, 
        betas=(config.beta1, config.beta2)
    )
    
    # Training history
    loss_history = {
        'g_losses': [],
        'd_losses': []
    }
    
    print('\nStarting training...')
    print('=' * 60)
    
    for epoch in range(1, config.epochs + 1):
        g_loss_epoch = 0
        d_loss_epoch = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.epochs}')
        
        for batch_idx, (imgs, _) in enumerate(progress_bar):
            # Prepare batch
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)
            
            # Labels for real and fake images
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ========== Train Discriminator ==========
            d_optimizer.zero_grad()
            
            # Loss on real images
            real_pred = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, real_labels)
            
            # Loss on fake images
            z = torch.randn(batch_size, config.latent_dim).to(device)
            fake_imgs = generator(z)
            fake_pred = discriminator(fake_imgs.detach())
            d_fake_loss = adversarial_loss(fake_pred, fake_labels)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
            
            # ========== Train Generator ==========
            g_optimizer.zero_grad()
            
            # Generate new fake images
            z = torch.randn(batch_size, config.latent_dim).to(device)
            fake_imgs = generator(z)
            fake_pred = discriminator(fake_imgs)
            
            # Generator loss - try to fool discriminator
            g_loss = adversarial_loss(fake_pred, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            # Track losses
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}'
            })
            
            # Save sample images
            if batch_idx % config.sample_interval == 0:
                save_generated_images(
                    generator, epoch, batch_idx, config.latent_dim, 
                    device, config.sample_dir
                )
        
        # Calculate average losses for the epoch
        avg_g_loss = g_loss_epoch / num_batches
        avg_d_loss = d_loss_epoch / num_batches
        
        loss_history['g_losses'].append(avg_g_loss)
        loss_history['d_losses'].append(avg_d_loss)
        
        print(f'Epoch {epoch}/{config.epochs} - '
              f'D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}')
        
        # Save checkpoint
        if epoch % config.checkpoint_interval == 0 or epoch == config.epochs:
            save_checkpoint(
                generator, discriminator, g_optimizer, d_optimizer,
                epoch, loss_history, config.checkpoint_dir
            )
    
    print('\nTraining completed!')
    
    # Plot loss curves
    plot_loss_curves(loss_history, save_path='./loss_curves.png')
    
    # Save final model
    torch.save(generator.state_dict(), f'{config.checkpoint_dir}/generator_final.pth')
    torch.save(discriminator.state_dict(), f'{config.checkpoint_dir}/discriminator_final.pth')
    print(f'Final models saved to {config.checkpoint_dir}/')


if __name__ == '__main__':
    train()
