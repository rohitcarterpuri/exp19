"""Configuration file for GAN training"""

class Config:
    # Dataset
    img_size = 28
    channels = 1
    img_shape = (channels, img_size, img_size)
    
    # Model architecture
    latent_dim = 100
    generator_features = 256
    discriminator_features = 128
    
    # Training
    batch_size = 64
    epochs = 200
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    
    # Device
    device = 'cuda'  # Will be auto-detected if 'auto'
    
    # Paths
    data_path = './data'
    checkpoint_dir = './checkpoints'
    sample_dir = './samples'
    
    # Logging
    sample_interval = 50  # Save sample images every N batches
    checkpoint_interval = 5  # Save checkpoint every N epochs
    
    # Random seed
    seed = 42
