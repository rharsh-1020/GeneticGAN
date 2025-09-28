# GeneticGAN

GeneticGAN is a PyTorch project implementing two generative models:

1. **GAN-GA**: A DCGAN-based GAN integrated with a Genetic Algorithm (GA) to optimize latent vectors for better diversity and image quality.
2. **GAN**: A standard DCGAN for baseline comparison.

The goal is to generate high-quality images from an ImageFolder-style dataset and compare the models using **loss curves** and **FID scores**.

---

## 🔹 Generated Images

### GAN-GA Samples (Epoch 2, 6, 10)

<table>
<tr>
  <td align="center"><img src="output/gan_ga/sample_1_63.png" width="120"/><br>Epoch 2</td>
  <td align="center"><img src="output/gan_ga/sample_5_315.png" width="120"/><br>Epoch 6</td>
  <td align="center"><img src="output/gan_ga/sample_9_567.png" width="120"/><br>Epoch 10</td>
</tr>
</table>

> GAN-GA produces images with higher diversity and sharper details.

### GAN Samples

<table>
<tr>
  <td align="center"><img src="output/gan/sample_1_63 (1).png" width="120"/><br>Epoch 2</td>
  <td align="center"><img src="output/gan/sample_7_441 (1).png" width="120"/><br>Epoch 6</td>
  <td align="center"><img src="output/gan/sample_9_567 (1).png" width="120"/><br>Epoch 10</td>
</tr>
</table>

> GAN images are less diverse and slightly blurrier compared to GAN-GA.

---

## 🔹 Training Loss Comparison

**GAN-GA Losses:**  
<img src="gan_ga_losses.png" width="400"/>  

**GAN Losses:**  
<img src="gan_losses.png" width="400"/>  

> GAN-GA shows slightly more stable generator convergence due to GA-guided latent vector optimization.

---

## 🔹 FID Score Comparison

<img src="fid_comparison.png" width="400"/>  

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
