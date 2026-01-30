# GoogLeNet PyTorch Implementation

A PyTorch implementation of GoogLeNet (Inception-v1) for image classification tasks. This project trains a GoogLeNet model on a custom image dataset with 11 classes.

## Project Overview

This repository contains a complete implementation of GoogLeNet, a deep convolutional neural network architecture introduced by Google. The model features the innovative Inception modules that allow for multi-scale feature extraction within a single layer.

### Key Features

- **GoogLeNet Architecture**: Full implementation of the GoogLeNet architecture with Inception modules
- **Auxiliary Classifiers**: Two auxiliary classifiers during training to improve gradient flow
- **Custom Dataset Support**: Configurable for various image classification tasks
- **PyTorch Framework**: Built using PyTorch for flexibility and performance
- **GPU Support**: Automatic GPU acceleration when available

## Dataset

The dataset consists of 11 classes organized in the `data/` directory:
- adder
- ball
- beer_barrel
- cob
- entrepreneur
- finger
- geyser
- professional_tennis
- schnauzer
- sister
- stemma

**Total samples**: 5,495 images (4,396 training, 1,099 testing)

## Project Structure

```
├── model.py          # GoogLeNet model architecture with Inception blocks
├── train.py          # Training script
├── utils.py          # Utility functions (loss functions)
├── README.md         # This file
└── data/             # Dataset directory with class subdirectories
```

## Model Architecture

### GoogLeNet Components

The GoogLeNet model includes:

1. **Initial Convolution Layer**: 7×7 convolution for feature extraction
2. **Inception Modules**: Multi-scale parallel convolution paths
   - 1×1 convolution (dimension reduction)
   - 3×3 convolution (medium receptive field)
   - 5×5 convolution (large receptive field)
   - Max pooling path (feature preservation)
3. **Auxiliary Classifiers**: Intermediate classification paths during training for improved gradient flow
4. **Final Classification Layer**: Global average pooling followed by fully connected layers

### Inception Block Structure

Each Inception module concatenates outputs from multiple parallel convolutional paths with different kernel sizes, allowing the network to capture features at multiple scales simultaneously.

## Requirements

- Python 3.x
- PyTorch
- TorchVision
- NumPy

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install torch torchvision numpy
```

## Usage

### Training

Run the training script with optional arguments:

```bash
python train.py --batch_size 8 --epochs 100 --lr 0.0001 --n_class 11
```

#### Training Arguments

- `--batch_size`: Batch size for training (default: 8)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate for the optimizer (default: 0.0001)
- `--n_class`: Number of classes (default: 11)

### Training Output

The training script will:
- Load the dataset from the `data/` directory
- Split data into training (80%) and testing (20%) sets
- Train the model using Adam optimizer and CrossEntropyLoss
- Print epoch-wise statistics including training loss, test loss, and accuracy
- Save the trained model as `model.pkl`

## Model Details

### Training Configuration

- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss (with auxiliary classifiers weighted at 0.3 each during training)
- **Device**: GPU (CUDA) if available, otherwise CPU
- **Data Normalization**: ImageNet-style normalization (mean=0.5, std=0.5)

### Auxiliary Classifiers

During training, the model uses auxiliary classifiers at two intermediate points to:
- Provide additional supervision signals to deeper layers
- Combat vanishing gradient problems
- Improve convergence speed

These auxiliary classifiers are removed during inference for standard prediction.

## Output

After training completes, the model is saved to `model.pkl` using pickle. The training script also prints:

```
Epoch: 1/100, Train Loss: 2.3456, Test Loss 2.1234, Accuracy: 15.50
...
```

## File Descriptions

### [model.py](model.py)
Contains the `GoogLeNet` and `Inception` class implementations. The GoogLeNet model includes 9 Inception modules and auxiliary classifiers for intermediate supervision.

### [train.py](train.py)
Main training script that:
- Loads and preprocesses the dataset
- Initializes the model and optimizer
- Implements the training loop
- Evaluates on test set
- Saves the trained model

### [utils.py](utils.py)
Utility functions including `weightedCrossEntropyLoss` for combining losses from the main and auxiliary classifiers.

## Notes

- The current training loop primarily uses the main classifier's loss (auxiliary classifiers' losses are commented out). You can uncomment to use the weighted combination.
- Data split is fixed with a seed (42) for reproducibility
- The model expects 3-channel RGB images
- Input images should be appropriately sized (typically 224×224 for GoogLeNet)

## References

- Szegedy, C., et al. (2015). "Going Deeper with Convolutions" (GoogLeNet/Inception-v1 Paper)
