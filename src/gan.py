import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from pathlib import Path
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths

class Generator(nn.Module):
    def __init__(self, latent_dim=100, ngf=64, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1),1,1))

class Discriminator(nn.Module):
    def __init__(self, ndf=64, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8,1,4,1,0,bias=False)
        )
    def forward(self, x):
        return self.net(x).view(-1,1).squeeze(1)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*args.channels,[0.5]*args.channels)
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    G = Generator(args.latent_dim, args.ngf, args.channels).to(device)
    D = Discriminator(args.ndf, args.channels).to(device)
    optimD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5,0.999))
    optimG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5,0.999))
    bce_loss = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(64,args.latent_dim, device=device)
    g_losses, d_losses, fid_scores = [],[],[]
    global_step = 0
    for epoch in range(args.epochs):
        for i, (real_images,_) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            D.zero_grad()
            real_logits = D(real_images)
            loss_real = bce_loss(real_logits, torch.ones(batch_size, device=device))
            noise = torch.randn(batch_size,args.latent_dim, device=device)
            fake = G(noise)
            fake_logits = D(fake.detach())
            loss_fake = bce_loss(fake_logits, torch.zeros(batch_size, device=device))
            lossD = loss_real+loss_fake
            lossD.backward()
            optimD.step()
            G.zero_grad()
            lossG = bce_loss(D(fake), torch.ones(batch_size, device=device))
            lossG.backward()
            optimG.step()
            if global_step % args.log_every==0:
                g_losses.append(lossG.item())
                d_losses.append(lossD.item())
            global_step+=1
    torch.save({"g_losses":g_losses,"d_losses":d_losses,"fid_scores":fid_scores}, Path(args.save_dir)/"metrics.pt")
    return g_losses, d_losses, fid_scores
