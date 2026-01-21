# PyTorch AlexNet

[English](#english) | [中文](README_ZH.md)

---

# PyTorch AlexNet

A PyTorch implementation of the classic AlexNet for image classification on the Imagenette dataset.

## Introduction

This repository is a PyTorch implementation of the AlexNet architecture, trained on the Imagenette dataset (a subset of 10 classes from ImageNet).

## Features

- **Framework**: PyTorch
- **Dataset**: Imagenette (10-class ImageNet subset)
- **Monitoring**: TensorBoard visualization of loss and accuracy
- **Metrics**: Support for top-1 and top-5 accuracy tracking
- **Model**: Classic AlexNet with ReLU activation and Dropout regularization

## Project Structure

```
.
├── main.py              # Training script
├── model.py             # AlexNet model definition
├── utils.py             # Utility functions and metrics
├── datasets/
│   └── imagenette2/     # Imagenette dataset (train/val folders)
├── checkpoints/         # Saved model weights
└── runs/                # TensorBoard logs
    └── alexnet/
```

## Requirements

```
python>=3.7
torch
torchvision
tqdm
```

CUDA support is recommended but not required. CPU training is also supported.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd alexnet_pytorch
```

2. Install dependencies:
```bash
pip install torch torchvision tqdm
```

## Dataset Preparation

1. Download the Imagenette dataset from [fastai/imagenette](https://github.com/fastai/imagenette)

2. Extract to the datasets folder:
```bash
tar -xzf imagenette2.tgz -C datasets/
```

The directory structure should look like:
```
datasets/
└── imagenette2/
    ├── train/
    │   ├── n01440764/
    │   ├── n02102040/
    │   └── ... (8 more classes)
    └── val/
        ├── n01440764/
        ├── n02102040/
        └── ... (8 more classes)
```

## Training

### Basic Usage

Run the training script:
```bash
python main.py
```

### Command Line Arguments

- `--dataset_root_dir`: Path to the dataset directory (default: `datasets/imagenette2`)
- `--epochs`: Number of training epochs (default: `50`)
- `--batch-size`: Batch size for training (default: `64`)
- `--lr`: Learning rate (default: `0.001`)
- `--num_classes`: Number of classes (default: `1000` for full ImageNet, use `10` for Imagenette)

### Example

```bash
python main.py --epochs 100 --batch-size 128 --lr 0.001 --num_classes 10
```

## Monitoring Training with TensorBoard

The training script automatically logs metrics to TensorBoard. To visualize:

1. Start TensorBoard:
```bash
tensorboard --logdir=runs
```

2. Open your browser and navigate to: `http://localhost:6006`

You can track:
- Training loss and accuracy
- Validation loss and accuracy
- Best model performance

## Model Architecture

The AlexNet model consists of:

**Feature Extraction**:
- 5 convolutional layers with ReLU activation
- 3 max pooling layers
- Input: 224×224 RGB images
- Output: 256 feature maps (6×6)

**Classification**:
- 3 fully connected layers
- 2 dropout layers for regularization
- Output: num_classes predictions

## Key Features

✅ Automatic best model saving based on validation accuracy  
✅ TensorBoard integration for real-time monitoring  
✅ Support for both CPU and GPU training  
✅ Top-1 and top-5 accuracy metrics  
✅ Cross-entropy loss tracking  
✅ Clean, modular code structure  

## Output

After training:
- **Model checkpoints** saved in `checkpoints/` directory
- **Best model** automatically saved as the validation accuracy improves
- **TensorBoard logs** saved in `runs/alexnet/` for visualization

## Notes

- For Imagenette dataset, typically use `--num_classes 10`
- Default learning rate (0.001) works well with Adam optimizer
- Training progress and metrics are logged to both console and TensorBoard
- Best model is saved whenever validation accuracy improves

## License

This project is provided as-is for educational purposes.

## References

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NIPS.
- [Imagenette Dataset](https://github.com/fastai/imagenette)
