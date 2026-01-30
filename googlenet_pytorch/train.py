import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import numpy as np
import pickle
from model import GoogLeNet
from utils import weightedCrossEntropyLoss
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GoogLeNet on CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.01)')
    parser.add_argument('--n_class', type=int, default=11, help='number of classes (default: 10)')

    args = parser.parse_args()
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = datasets.ImageFolder(root='./data', transform=transform)
    test_data, train_data = random_split(dataset,(4396,1099), generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GoogLeNet(num_classes=args.n_class).to(device)

    loss_function = weightedCrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss = [0 for i in range(args.epochs)]
    test_loss = [0 for i in range(args.epochs)]
    test_loss_func = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(args.epochs):
        model.train()
        model.training = True
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            y1, y2, y3 = model(xb)
            loss = test_loss_func(y3, yb) #+ 0.3*test_loss_func(y2,yb) + test_loss_func(y3,yb)
            train_loss[epoch] = loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            model.training = False
            losses, nums = zip(*[(test_loss_func(model(xb.to(device)),yb.to(device)).item(),len(xb.to(device))) for xb, yb in test_loader])
            test_loss[epoch] = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            correct = 0
            total = 0
            
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Epoch: {epoch+1}/{args.epochs}, Train Loss: {train_loss[epoch]}, Test Loss {test_loss[epoch]}, Accuracy: {100*correct/total}')

    with open('model.pkl','wb') as f:
        pickle.dump(model,f)

