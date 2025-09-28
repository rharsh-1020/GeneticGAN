# GeneticGAN
This repository contains **PyTorch implementations** of two generative models:

1. **GAN-GA**: A DCGAN-based GAN integrated with a Genetic Algorithm (GA) for latent space optimization.
2. **GAN**: A standard DCGAN implementation for comparison.

Both models generate images from an ImageFolder-style dataset and compute **FID scores** for quality evaluation.

---

## Repository Structure

gan-ga/
├── gan_ga.py # GAN-GA model, GA utilities, and training code
├── gan.py # Vanilla GAN model and training code
├── main.py # Runs training for both models and generates performance plots
├── outputs/ # Folder for saving generated images, checkpoints, and metrics
├── README.md # This file

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/gan-ga.git
cd gan-ga
Install dependencies (use a virtual environment):

bash
pip install torch torchvision matplotlib pillow numpy pytorch-fid
GPU is recommended but not required. CUDA will be automatically detected if available.

Dataset
The models expect a dataset in ImageFolder format:

data_dir/
├── class1/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── class2/
│   └── ...
GAN-GA supports RGB images (channels=3).

Vanilla GAN supports grayscale images (channels=1).

Usage

Train and evaluate both models

Run main.py to train both models sequentially and generate plots:

python main.py

This will:

Train GAN-GA and Vanilla GAN.

Save generated images, checkpoints, and metrics in outputs/.

Generate plots:

gan_ga_losses.png
gan_vanilla_losses.png
fid_comparison.png

Training Parameters

Modify hyperparameters directly in gan_ga.py or gan.py:

epochs: Number of training epochs

batch_size: Batch size

latent_dim: Size of latent vector

lr: Learning rate

GAN-GA specific parameters:

pop_size: Population size for GA

ga_every: Frequency of GA steps

mutation_rate: Mutation probability

lambda_div: Weight for diversity in fitness

topk_fraction: Fraction of top latent vectors used to train Generator

Output
Generated images: Saved at intervals during training.

Checkpoints: Model weights and GA population state (GAN-GA).

Metrics: Loss curves and FID scores (metrics.pt).

Plots:

gan_ga_losses.png

gan_vanilla_losses.png

fid_comparison.png

Notes
GAN-GA uses a Genetic Algorithm to evolve latent vectors for better diversity and quality.

FID computation requires a sufficient number of real and generated images.

Both models are modular and can be extended for different datasets, image sizes, or GAN architectures.

