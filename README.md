# GeneticGAN

GeneticGAN is a PyTorch project that implements two generative models:

1. **GAN-GA**: A DCGAN-based GAN integrated with a Genetic Algorithm (GA) to optimize latent vectors for better diversity and image quality.
2. **Vanilla GAN**: A standard DCGAN for baseline comparison.

The goal is to generate high-quality images from an ImageFolder-style dataset and compare the models using **loss curves** and **FID scores**.

---

## 🔹 Project Outcome

### 1. Generated Images

**GAN-GA (with GA-enhanced latent space):**

![GAN-GA Sample](outputs/sample_gan_ga.png)

**Vanilla GAN:**

![Vanilla GAN Sample](outputs/sample_vanilla.png)

> GAN-GA produces images with more diversity and sharper details compared to Vanilla GAN.

---

### 2. Training Loss Comparison

**Generator and Discriminator Loss Curves:**

**GAN-GA Losses:**  
![GAN-GA Losses](gan_ga_losses.png)  

**Vanilla GAN Losses:**  
![Vanilla GAN Losses](gan_vanilla_losses.png)  

> GAN-GA shows slightly more stable generator convergence due to GA-guided latent vector optimization.

---

### 3. FID Score Comparison

![FID Comparison](fid_comparison.png)  

> GAN-GA achieves **lower FID scores**, indicating higher image quality and diversity compared to Vanilla GAN.

---

## 🔹 Outcome Summary

- Integrating a Genetic Algorithm in the GAN latent space improves image diversity and quality.  
- GAN-GA outperforms Vanilla GAN in both visual results and FID evaluation.  
- Loss curves suggest better training stability for GAN-GA due to GA-based latent vector evolution.

---

## ⚡ Notes

- The dataset should be organized in ImageFolder format.  
- GAN-GA supports RGB images, while Vanilla GAN supports grayscale.  
- Outputs (images, checkpoints, metrics) are saved in the `outputs/` folder.  
