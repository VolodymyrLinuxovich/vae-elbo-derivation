<!-- MathJax -->
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# vae-elbo-derivation

A short implementation of the Variational Autoencoder with complete step-by-step ELBO and KL-term derivation, including mathematical explanations and examples on MNIST.

---

## Formula

$$
\mathrm{ELBO}
=
\mathbb{E}_{q_\phi(z\mid x)}\bigl[\log p_\theta(x\mid z)\bigr]
\;-\;
D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\,\|\,p(z)\bigr)
$$

---

## Installation

```bash
pip install -r requirements.txt
