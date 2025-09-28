# GeneticGAN

GeneticGAN is a PyTorch project implementing two generative models:

1. **GAN-GA**: A DCGAN-based GAN integrated with a Genetic Algorithm (GA) to optimize latent vectors for better diversity and image quality.
2. **GAN**: A standard DCGAN for baseline comparison.

The goal is to generate high-quality images from an ImageFolder-style dataset and compare the models using **loss curves** and **FID scores**.

---

## 🔹 Generated Images

### GAN-GA Samples

| ![GAN-GA 1](outputs/gan_ga_1.png) | ![GAN-GA 2](outputs/gan_ga_2.png) |
|----------------------------------|----------------------------------|
| ![GAN-GA 3](outputs/gan_ga_3.png) | ![GAN-GA 4](outputs/gan_ga_4.png) |

> GAN-GA produces images with higher diversity and sharper details.

### GAN Samples

| ![Vanilla GAN 1](outputs/vanilla_1.png) | ![Vanilla GAN 2](outputs/vanilla_2.png) |
|----------------------------------------|----------------------------------------|
| ![Vanilla GAN 3](outputs/vanilla_3.png) | ![Vanilla GAN 4](outputs/vanilla_4.png) |

> GAN images are less diverse and slightly blurrier compared to GAN-GA.

---

## 🔹 Training Loss Comparison

**GAN-GA Losses:**  
![GAN-GA Losses](gan_ga_losses.png)  

**GAN Losses:**  
![GAN Losses](gan_losses.png)  

> GAN-GA shows slightly more stable generator convergence due to GA-guided latent vector optimization.

---

## 🔹 FID Score Comparison

![FID Comparison](fid_comparison.png)  

> GAN-GA achieves **lower FID scores**, indicating higher image quality and diversity.

---

## 🔹 Outcome Summary

- Integrating a Genetic Algorithm in the GAN latent space improves image diversity and quality.  
- GAN-GA outperforms Vanilla GAN in both visual results and FID evaluation.  
- Loss curves suggest better training stability for GAN-GA due to GA-based latent vector evolution.

---

## ⚡ Notes

- The dataset should be organized in ImageFolder format.  
- GAN-GA supports RGB images, while GAN supports grayscale.  
- Outputs (images, checkpoints, metrics) are saved in the `outputs/` folder.  
