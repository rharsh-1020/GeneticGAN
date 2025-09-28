import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from gan_ga import train as train_ga
from gan import train as train_gan

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--save_dir', type=str, default='outputs')
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--pop_size', type=int, default=128)
    parser.add_argument('--ga_every', type=int, default=50)
    parser.add_argument('--ga_offspring_fraction', type=float, default=0.3)
    parser.add_argument('--tournament_k', type=int, default=3)
    parser.add_argument('--mutation_rate', type=float, default=0.15)
    parser.add_argument('--mutation_scale', type=float, default=0.08)
    parser.add_argument('--lambda_div', type=float, default=0.15)
    return parser.parse_args()

args = get_args()
Path(args.save_dir).mkdir(exist_ok=True)

g_losses_ga, d_losses_ga, fid_ga = train_ga(args)
g_losses_vanilla, d_losses_vanilla, fid_vanilla = train_gan(args)

plt.figure(figsize=(8,5))
plt.plot(g_losses_ga,label="G Loss")
plt.plot(d_losses_ga,label="D Loss")
plt.title("GAN-GA: Generator & Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(Path(args.save_dir)/"gan_ga_losses.png")

plt.figure(figsize=(8,5))
plt.plot(g_losses_vanilla,label="G Loss")
plt.plot(d_losses_vanilla,label="D Loss")
plt.title("Vanilla GAN: Generator & Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(Path(args.save_dir)/"gan_vanilla_losses.png")

plt.figure(figsize=(8,5))
plt.plot(fid_ga,label="GAN-GA FID")
plt.plot(fid_vanilla,label="Vanilla GAN FID")
plt.title("FID Score Comparison")
plt.xlabel("Checkpoints")
plt.ylabel("FID")
plt.legend()
plt.savefig(Path(args.save_dir)/"fid_comparison.png")
