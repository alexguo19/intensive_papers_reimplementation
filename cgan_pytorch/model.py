import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.label_emb = nn.Embedding(self.args.n_classes, self.args.n_classes)

        self.model = nn.Sequential(
            *self.block(self.args.latent_dim + self.args.n_classes, 128, normalize=False),
            *self.block(128, 256),
            *self.block(256, 512),
            *self.block(512, 1024),
            nn.Linear(1024, int(np.prod((self.args.channels, self.args.img_size, self.args.img_size)))),
            nn.Tanh()
        )

    def block(self, in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers


    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), self.args.channels, self.args.img_size, self.args.img_size)
        return img


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        self.label_embedding = nn.Embedding(self.args.n_classes, self.args.n_classes)
        self.model = nn.Sequential(
            nn.Linear(self.args.n_classes + int(np.prod((self.args.channels, self.args.img_size, self.args.img_size))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

