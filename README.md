# GAN for Handwritten Digit Generation

A deep learning project implementing a Generative Adversarial Network (GAN) to generate realistic handwritten digits using the MNIST dataset.

## 🎯 Overview

This project implements a standard DCGAN (Deep Convolutional GAN) architecture to generate synthetic handwritten digits that resemble the MNIST dataset. The generator learns to create realistic digit images, while the discriminator learns to distinguish between real and fake digits.

## ✨ Features

- **DCGAN Architecture**: Uses transposed convolutions for generation and strided convolutions for discrimination
- **Training Monitoring**: Real-time loss tracking and image sample generation
- **Checkpoint System**: Save and resume training from checkpoints
- **Visualization Tools**: Generate sample grids and latent space interpolations
- **Reproducibility**: Fixed random seeds for consistent results

## 🏗️ Architecture

### Generator
- Input: Random noise vector (100-dim)
- 4 transposed convolutional layers with batch normalization
- Output: 28×28 grayscale image

### Discriminator
- Input: 28×28 grayscale image
- 4 convolutional layers with batch normalization
- Output: Probability of image being real (0-1)

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
