import os
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from pytorch_fid.fid_score import calculate_fid_given_paths
from PIL import Image

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))

class Discriminator(nn.Module):
    def __init__(self, ndf=64, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )
    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)

def init_population(pop_size, latent_dim, device):
    return torch.randn(pop_size, latent_dim, device=device)

def tournament_selection(pop, fitness, k=3):
    idxs = np.random.choice(len(pop), k, replace=False)
    best = idxs[0]
    for i in idxs:
        if fitness[i] > fitness[best]:
            best = i
    return best

def crossover(a, b, p=0.5):
    mask = (torch.rand_like(a) < p).float()
    return mask * a + (1 - mask) * b

def mutate(z, rate=0.1, scale=0.1):
    mask = (torch.rand_like(z) < rate).float()
    noise = torch.randn_like(z) * scale
    return z + mask * noise

def compute_diversity_score(pop):
    with torch.no_grad():
        diffs = pop.unsqueeze(1) - pop.unsqueeze(0)
        dists = torch.sqrt((diffs ** 2).sum(dim=2) + 1e-8)
        return dists.mean(dim=1)

def save_image_grid(tensor, path, nrow=8, normalize=True):
    grid = utils.make_grid(tensor, nrow=nrow, normalize=normalize)
    utils.save_image(grid, path)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*args.channels, [0.5]*args.channels),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    G = Generator(latent_dim=args.latent_dim, ngf=args.ngf, channels=args.channels).to(device)
    D = Discriminator(ndf=args.ndf, channels=args.channels).to(device)
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)
    G.apply(weights_init)
    D.apply(weights_init)
    optimD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce_loss = nn.BCEWithLogitsLoss()
    population = init_population(args.pop_size, args.latent_dim, device)
    fixed_noise = torch.randn(64, args.latent_dim, device=device)
    g_losses, d_losses, fid_scores = [], [], []
    global_step = 0
    for epoch in range(args.epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            D.zero_grad()
            real_logits = D(real_images)
            labels_real = torch.ones(batch_size, device=device)
            loss_D_real = bce_loss(real_logits, labels_real)
            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = G(noise)
            fake_logits = D(fake_images.detach())
            labels_fake = torch.zeros(batch_size, device=device)
            loss_D_fake = bce_loss(fake_logits, labels_fake)
            lossD = loss_D_real + loss_D_fake
            lossD.backward()
            optimD.step()
            G.zero_grad()
            fake_logits_forG = D(fake_images)
            lossG = bce_loss(fake_logits_forG, labels_real)
            lossG.backward()
            optimG.step()
            if global_step % args.ga_every == 0:
                with torch.no_grad():
                    gen_imgs = G(population)
                    prob = torch.sigmoid(D(gen_imgs))
                diversity_scores = compute_diversity_score(population)
                div_norm = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min() + 1e-8)
                fitness = prob.cpu().numpy() * (1.0 - args.lambda_div) + div_norm.cpu().numpy() * args.lambda_div
                new_population = population.clone()
                num_offspring = int(len(population) * args.ga_offspring_fraction)
                for _ in range(num_offspring):
                    a_idx = tournament_selection(population, fitness, k=args.tournament_k)
                    b_idx = tournament_selection(population, fitness, k=args.tournament_k)
                    child = crossover(population[a_idx], population[b_idx])
                    child = mutate(child, mutation_rate=args.mutation_rate, mutation_scale=args.mutation_scale)
                    worst_idx = int(np.argmin(fitness))
                    new_population[worst_idx] = child.detach()
                population = new_population.detach()
            if global_step % args.log_every == 0:
                g_losses.append(lossG.item())
                d_losses.append(lossD.item())
            global_step += 1
    torch.save({"g_losses": g_losses, "d_losses": d_losses, "fid_scores": fid_scores}, Path(args.save_dir)/"metrics.pt")
    return g_losses, d_losses, fid_scores
