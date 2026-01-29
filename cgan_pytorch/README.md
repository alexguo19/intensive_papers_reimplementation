# Conditional GAN (CGAN) - PyTorch Implementation

A PyTorch implementation of Conditional Generative Adversarial Networks (CGAN) for conditional image generation on MNIST and CIFAR10 datasets.

## Overview

Conditional GANs extend the standard GAN framework by conditioning both the Generator and Discriminator on class labels. This enables controlled image generation where you can specify which class of image to generate.

## Project Structure

```
cgan_pytorch/
├── train.py          # Main training script
├── model.py          # Generator and Discriminator architectures
├── README.md         # Documentation
├── ckpt/             # Model checkpoints
├── data/             # Datasets (auto-downloaded)
│   ├── mnist/
│   └── cifar10/
└── images/           # Generated sample images
    ├── mnist/
    └── cifar10/
```

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- TorchVision >= 0.10.0
- NumPy >= 1.19.0

Install dependencies:

```bash
pip install torch torchvision numpy
```

## Model Architecture

### Generator

- Fully connected network with label embedding
- Architecture: Input → 128 → 256 → 512 → 1024 → Output
- Batch Normalization on all layers except first
- LeakyReLU(0.2) activation, Tanh output

### Discriminator

- Fully connected classifier with label embedding
- Architecture: Input → 512 → 512 → 512 → 1
- Dropout(0.4) for regularization
- LeakyReLU(0.2) activation

## Usage

### Train on MNIST (default)

```bash
python train.py
```

### Train on CIFAR10

```bash
python train.py --dataset cifar10 --channels 3
```

### Custom Training Configuration

```bash
python train.py --dataset cifar10 \
                --n_epochs 300 \
                --batch_size 128 \
                --lr 0.0001 \
                --latent_dim 100
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | mnist | Dataset to use: `mnist` or `cifar10` |
| `--n_epochs` | 200 | Number of training epochs |
| `--batch_size` | 64 | Training batch size |
| `--lr` | 0.0002 | Learning rate for Adam optimizer |
| `--b1` | 0.5 | Adam beta1 parameter |
| `--b2` | 0.999 | Adam beta2 parameter |
| `--latent_dim` | 100 | Dimensionality of latent space |
| `--n_classes` | 10 | Number of classes in dataset |
| `--img_size` | 32 | Size of generated images |
| `--channels` | 1 | Number of image channels (1 for MNIST, 3 for CIFAR10) |
| `--sample_interval` | 400 | Interval for saving sample images |

## Outputs

- **Generated Images**: Saved to `images/{dataset}/` as PNG grids showing all 10 classes
- **Model Checkpoints**: Saved to `ckpt/{dataset}_generator.pth` and `ckpt/{dataset}_discriminator.pth`

## How It Works

1. **Generator** receives random noise + class label and generates a fake image
2. **Discriminator** receives an image + class label and predicts if it's real or fake
3. Both networks are trained adversarially with MSE loss
4. The conditioning allows generation of specific digit/object classes

## References

- [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) - Mirza & Osindero, 2014
