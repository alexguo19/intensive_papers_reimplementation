# GAN PyTorch

A PyTorch implementation of a vanilla Generative Adversarial Network (GAN) for image generation on MNIST and CIFAR-10 datasets.

## Overview

This project implements a GAN with fully-connected architecture for generating synthetic images. The model consists of:

- **Generator**: Transforms random noise from the latent space into synthetic images
- **Discriminator**: Classifies images as real or fake

The two networks are trained in an adversarial manner where the generator learns to produce increasingly realistic images while the discriminator learns to better distinguish real from fake.

## Project Structure

```
gan_pytorch/
├── model.py              # Generator and Discriminator model definitions
├── train.py              # Training script with CLI arguments
├── ckpt/                 # Saved model checkpoints (.pth files)
├── data/                 # Dataset directories (auto-downloaded)
│   ├── mnist/
│   └── cifar10/
└── images/               # Generated sample images during training
    └── {dataset}/        # 5x5 grid samples saved periodically
```

## Installation

### Requirements

- Python 3.6+
- PyTorch
- torchvision
- NumPy

```bash
pip install torch torchvision numpy
```

## Usage

### Training

Train on CIFAR-10 (default):
```bash
python train.py
```

Train on MNIST:
```bash
python train.py --dataset mnist --channels 1
```

Train with custom hyperparameters:
```bash
python train.py --dataset cifar10 --n_epochs 300 --batch_size 128 --lr 0.0001
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `cifar10` | Dataset: `mnist` or `cifar10` |
| `--n_epochs` | int | `200` | Number of training epochs |
| `--batch_size` | int | `64` | Batch size |
| `--lr` | float | `0.0002` | Learning rate |
| `--b1` | float | `0.5` | Adam beta1 parameter |
| `--b2` | float | `0.999` | Adam beta2 parameter |
| `--latent_dim` | int | `100` | Latent space dimensionality |
| `--img_size` | int | `28` | Image size (images resized to this) |
| `--channels` | int | `3` | Image channels (1 for MNIST, 3 for CIFAR-10) |
| `--sample_interval` | int | `400` | Batches between saving samples |

## Model Architecture

### Generator

```
Input: Random noise (latent_dim) ~ N(0, 1)
    ↓
Linear(latent_dim → 128) + BatchNorm + LeakyReLU(0.2)
    ↓
Linear(128 → 256) + BatchNorm + LeakyReLU(0.2)
    ↓
Linear(256 → 512) + BatchNorm + LeakyReLU(0.2)
    ↓
Linear(512 → 1024) + BatchNorm + LeakyReLU(0.2)
    ↓
Linear(1024 → channels × img_size²) + Tanh
    ↓
Output: Image (channels, img_size, img_size) in [-1, 1]
```

### Discriminator

```
Input: Image (channels × img_size²)
    ↓
Linear(channels × img_size² → 512) + LeakyReLU(0.2)
    ↓
Linear(512 → 256) + LeakyReLU(0.2)
    ↓
Linear(256 → 1) + Sigmoid
    ↓
Output: Probability [0, 1] (real vs fake)
```

## Training Details

- **Loss Function**: Binary Cross-Entropy (BCE)
- **Optimizer**: Adam with configurable betas
- **Data Preprocessing**: Normalized to [-1, 1] range
- **Hardware**: Automatic CUDA/GPU detection

Training alternates between:
1. **Discriminator update**: Maximize log(D(x)) + log(1 - D(G(z)))
2. **Generator update**: Maximize log(D(G(z)))

## Datasets

| Dataset | Size | Channels | Classes |
|---------|------|----------|---------|
| MNIST | 70,000 images (28×28) | 1 (grayscale) | 10 digits |
| CIFAR-10 | 50,000 images (32×32) | 3 (RGB) | 10 objects |

Datasets are automatically downloaded on first run.

## Output

### Checkpoints
- Location: `ckpt/{dataset}_generator.pth` and `ckpt/{dataset}_discriminator.pth`
- Saved every `sample_interval` batches

### Generated Images
- Location: `images/{dataset}/{batch_number}.png`
- Format: 5×5 grid of 25 generated samples
- Useful for monitoring training progress

### Console Logging
```
[Epoch X/Y] [Batch Z/N] [D loss: 0.XXXX] [G loss: 0.XXXX]
```

## Tips

- **GPU**: Use CUDA-enabled GPU for faster training
- **Batch Size**: Larger batches (128-256) improve stability but need more memory
- **Learning Rate**: Reduce if training is unstable, increase if too slow
- **Epochs**: 200-300 epochs typically produces reasonable results

## License

MIT
