import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch
from model import Generator, Discriminator

def sample_image(args, device, n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = torch.randn(n_row ** 2, args.latent_dim, device=device)
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = torch.LongTensor(labels).to(device)
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, os.path.join("images", args.dataset, "%d.png" % batches_done), nrow=n_row, normalize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Conditional GAN Example")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"], help="dataset to use")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels, cifar10 is 3 and mnist is 1")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")

    args = parser.parse_args()

    if not os.path.exists(os.path.join("images", args.dataset)):
        os.makedirs(os.path.join("images", args.dataset))
    
    if not os.path.exists("ckpt"):
        os.makedirs("ckpt")

    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loss functions
    adversarial_loss = torch.nn.MSELoss().to(device)

    # Initialize generator and discriminator
    generator = Generator(args)
    discriminator = Discriminator(args)
    generator.to(device)
    discriminator.to(device)

    if not os.path.exists(os.path.join("data",args.dataset)):
        os.makedirs(os.path.join("data",args.dataset))

    if args.dataset == "mnist": 
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                os.path.join("data", "mnist"),
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
        )
    elif args.dataset == "cifar10":
        dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                os.path.join("data", "cifar10"),
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=True,
        )
    else:
        raise ValueError("Dataset not recognized. Please use 'mnist' or 'cifar10'.")

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    for epoch in range(args.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = torch.ones(batch_size, 1, device=device, dtype=torch.float)
            fake = torch.zeros(batch_size, 1, device=device, dtype=torch.float)

            # Configure input
            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_labels = torch.LongTensor(np.random.randint(0, args.n_classes, batch_size)).to(device)

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % args.sample_interval == 0:
                torch.save(generator.state_dict(), os.path.join("ckpt", args.dataset + "_generator.pth"))
                torch.save(discriminator.state_dict(), os.path.join("ckpt", args.dataset + "_discriminator.pth"))
                sample_image(args, device, n_row=10, batches_done=batches_done)
