# vae-elbo-derivation
A short implementation of the Variational Autoencoder with complete step-by-step ELBO and KL-terminus output, including mathematical explanations and examples on MNIST.

![ELBO formula](https://latex.codecogs.com/png.latex?
\mathrm{ELBO}%20=%20\mathbb{E}_{q_\phi(z%7Cx)}[\log%20p_\theta(x%7Cz)]%20-
%20D_{\mathrm{KL}}(q_\phi(z%7Cx)\|p(z)))
