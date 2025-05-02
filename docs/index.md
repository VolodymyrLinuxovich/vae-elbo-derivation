<!-- MathJax -->
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# vae-elbo-derivation

<div align="center">

**Variational Autoencoder (VAE): Generative Modeling via Variational Inference**  
*Volodymyr Borysenko*  
Johannes Kepler University Linz

</div>

---

## 1. Introduction

In traditional autoencoders, an encoder–decoder pair learns to compress input data $x$ into a lower-dimensional representation $z$ and then reconstruct $x$ from $z$. However, such models lack a mechanism to generate novel samples from the learned data distribution. Variational Autoencoders (Kingma & Welling, 2014) address this by embedding data into a tractable latent space and defining a full generative model.

### 1.1 Generative Model with Latent Variable

Let $x \in \mathbb{R}^D$ denote observed data (e.g., images), and introduce a latent variable $z \in \mathbb{R}^d$. We define the joint distribution:

$$
p_\theta(x, z)
=
p(z)\,p_\theta(x \mid z),
$$

where:

- `$p(z) = \mathcal{N}(0, I)$` is a simple prior in the latent space.  
- `$p_{\theta}(x \mid z)$` is the decoder, a neural network mapping `z` to a distribution over `x`.


---

## 2. Formula

$$
\mathrm{ELBO}
=
\mathbb{E}_{q_\phi(z\mid x)}\bigl[\log p_\theta(x\mid z)\bigr]
\;-\;
D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\,\|\,p(z)\bigr)
$$

---

## 3. Installation

```bash
pip install -r requirements.txt
