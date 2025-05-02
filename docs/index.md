<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<div align="center">

**Variational Autoencoder (VAE): Generative Modeling via Variational Inference**
*Volodymyr Borysenko*
Johannes Kepler University Linz

</div>

---

## 1. Introduction

In traditional autoencoders, an encoder–decoder pair learns to compress input data $$x$$ into a lower-dimensional representation $$z$$ and then reconstruct $$x$$ from $$z$$. However, such models lack a mechanism to generate novel samples from the learned data distribution. Variational Autoencoders (Kingma & Welling, 2014) address this by embedding data into a tractable latent space and defining a full generative model.

### 1.1 Generative Model with Latent Variable

Let $$x$$ in $$\mathbb{R}^D$$ denote observed data (e.g., images), and introduce a latent variable $$z$$ in $$\mathbb{R}^d$$. We define the joint distribution:

$$
p_\theta(x, z)
=
p(z)\,p_\theta(x \mid z),
$$

where:

- $$p(z) = \mathcal{N}(0, I)$$ is a simple prior in the latent space.  
- $$p_{\theta}(x \mid z)$$ is the decoder, a neural network mapping z to a distribution over x.


---

### 1.2 Intractability of Exact Likelihood

The marginal log-likelihood measures how well our generative model explains the observed data. In principle, we seek parameters $$\theta$$ that maximize this quantity, but direct computation is infeasible:

Before optimization, we write the definition of the marginal likelihood as an integral over the latent variable:

$$
\log p_{\theta}(x)
= \log \int p_{\theta}(x, z) \, dz
= \log \int p(z)\,p_{\theta}(x \mid z) \, dz.
$$

Here:
- $$p(z) = \mathcal{N}(0, I)$$ is a simple prior in latent space.
- $$p_{\theta}(x\mid z)$$ is a neural-network decoder mapping latent variables to the data distribution.

Because $$p_{\theta}(x\mid z)$$ is parameterized by a deep network, this integral cannot be solved analytically for general architectures. As a result, we require an alternative variational approach to approximate and optimize $$\log p_{\theta}(x)$$.


## 3. Installation

```bash
pip install -r requirements.txt
