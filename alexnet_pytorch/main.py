import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tqdm
import logging
from torchvision.datasets import ImageFolder
from torchvision import transforms
import argparse
from torch.utils.tensorboard import SummaryWriter
from model import AlexNet
from utils import save_model, load_model, AccuracyMeter, Metric, CrossEntropyLossMetric, AccuracyMetric

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train AlexNet on ImageFolder dataset')
    parser.add_argument('--dataset_root_dir', type=str, default="datasets/imagenette2", help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=1000, help='Number of classes in the dataset')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset_root_dir = args.dataset_root_dir
    dataset_train_dir = os.path.join(dataset_root_dir, "train")
    dataset_valid_dir = os.path.join(dataset_root_dir, "val")

    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalization,
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalization,
    ])

    train_dataset = ImageFolder(root=dataset_train_dir, transform=train_transform)
    valid_dataset = ImageFolder(root=dataset_valid_dir, transform=valid_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = AlexNet(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    meter = AccuracyMeter((1, 5))
    best_acc = 0.0
    
    writer = SummaryWriter('runs/alexnet')

    for epoch in range(args.epochs):
        logging.info(f"training on epoch {epoch}")
        model.train()
        train_acc_metric = AccuracyMetric()
        train_cel_metric = CrossEntropyLossMetric()
        meter.reset()
        for input, target in tqdm.tqdm(train_loader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)
            
            meter.update(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc_metric.update(output, target)
            train_cel_metric.update(output, target)
            meter.update(output, target)

        train_loss = train_cel_metric.average()
        train_acc = train_acc_metric.average()
        logging.info(f"training cel: {train_loss}, acc: {train_acc}, {str(meter)}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        
        logging.info(f"testing on epoch {epoch}...")
        model.eval()
        valid_acc_metric = AccuracyMetric()
        valid_cel_metric = CrossEntropyLossMetric()
        meter.reset()
        with torch.no_grad():
            for x, y in tqdm.tqdm(valid_loader):
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                valid_acc_metric.update(pred, y)
                valid_cel_metric.update(pred, y)
                meter.update(pred, y)
        
        valid_acc = valid_acc_metric.average()
        valid_loss = valid_cel_metric.average()
        logging.info(f"testing loss: {valid_loss}, acc: {valid_acc}, {str(meter)}")
        
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch)
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            save_model(model, f"best_model_epoch{epoch}")
            logging.info(f"saved best model with accuracy: {best_acc}")
    
    writer.close()
