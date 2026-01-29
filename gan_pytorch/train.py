import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import Generator, Discriminator  # Assume model.py contains the GAN model definitions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a vanilla GAN on MNIST")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"], help="dataset to use")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels, cifar10 is 3 and mnist is 1")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    args = parser.parse_args()

    if not os.path.exists(os.path.join("images", args.dataset)):
        os.makedirs(os.path.join("images", args.dataset))
    
    if not os.path.exists("ckpt"):
        os.makedirs("ckpt")

    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adversarial_loss = nn.BCELoss().to(device)
    generator = Generator(args)
    generator.to(device)

    discriminator = Discriminator(args)
    discriminator.to(device)

    if not os.path.exists(os.path.join("data",args.dataset)):
        os.makedirs(os.path.join("data",args.dataset))

    if args.dataset == "mnist": 
        dataloader = DataLoader(
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
        dataloader = DataLoader(
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

    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            valid = torch.ones(imgs.size(0), 1, device=device, dtype=torch.float)
            fake = torch.zeros(imgs.size(0), 1, device=device, dtype=torch.float)

            real_imgs = imgs.to(device)
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), args.latent_dim, device=device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]"
                  % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.sample_interval == 0:
                torch.save(generator.state_dict(), os.path.join("ckpt", args.dataset + "_generator.pth"))
                torch.save(discriminator.state_dict(), os.path.join("ckpt", args.dataset + "_discriminator.pth"))
                save_image(gen_imgs.data[:25], os.path.join("images", args.dataset, "%d.png" % batches_done), nrow=5, normalize=True)

