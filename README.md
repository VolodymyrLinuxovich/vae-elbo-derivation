# vae-elbo-derivation
A short implementation of the Variational Autoencoder with complete step-by-step ELBO and KL-terminus output, including mathematical explanations and examples on MNIST.
\mathrm{ELBO} = \mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
             - D_{\mathrm{KL}}(q_\phi(z\mid x)\,\|\,p(z)).
