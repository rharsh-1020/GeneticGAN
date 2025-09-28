# GeneticGAN
This repository contains PyTorch implementations of two generative models:
**GAN-GA:** A DCGAN-based GAN integrated with a Genetic Algorithm (GA) for latent space optimization.
**GAN:** A standard DCGAN implementation for comparison.

Both models are designed to generate images from an ImageFolder-style dataset and compute FID scores for quality evaluation

#Installation

1.Clone the repository:
git clone https://github.com/your-username/gan-ga.git
cd gan-ga

2.Install dependencies (preferably in a virtual environment):

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


For grayscale training (Vanilla GAN), the channels are set to 1. For RGB training (GAN-GA), the channels are set to 3.

Usage
Train and evaluate both models

Edit main.py to set your dataset paths, or pass dataset directories as arguments if you adapt main.py to accept CLI arguments.

python main.py


This will:

Train GAN-GA and Vanilla GAN sequentially.

Save generated images, checkpoints, and metrics in the outputs/ directory.

Generate plots:

gan_ga_losses.png

gan_vanilla_losses.png

fid_comparison.png

Training parameters

Modify hyperparameters directly in gan_ga.py or gan.py, such as:

epochs: number of training epochs

batch_size: batch size

latent_dim: size of latent vector

lr: learning rate

GA-specific (GAN-GA only):

pop_size, ga_every, mutation_rate, lambda_div, etc.

Output

Generated images: Saved at intervals during training.

Checkpoints: Model weights and population state (GAN-GA) are saved for resuming training.

Metrics: Loss curves and FID scores saved as metrics.pt.

Plots: Loss and FID score comparison plots.

Notes

GAN-GA uses a Genetic Algorithm to evolve latent vectors for better diversity and quality.

FID computation requires a sufficient number of real and generated images.

Both models are modular and can be extended for different datasets, image sizes, or GAN architectures.

License

MIT License
