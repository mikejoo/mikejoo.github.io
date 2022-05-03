---
layout: post
title: Variational Inference and the Evidence Lower Bound (ELBO)
categories: [Bayesian, Variational Inference, VAE, Information Theory]
---

In this post, I provide a brief introduction to variational inference and the evidence lower bound.

# Problem

Bayesian inference requires computing the posterior distribution of a random variable, given a set of observations. In the usual latent variable model setting with observed data: 

$$\mathbf{X} = \{ \mathbf{x}_i\}_{i=1}^{N}$$

where each $$\mathbf{x} \in \mathcal{X}$$, the domain, and corresponding latent variables:

$$ \mathbf{Z} = \{ \mathbf{z}_i\}_{i=1}^{N}$$

where each $$\mathbf{z} \in \mathcal{Z}$$, the latent space, our goal is to identify the posterior:

$$ p(\mathbf{Z} \mid \mathbf{X}) = \frac{p(\mathbf{X} \mid \mathbf{Z}) p(\mathbf{Z})}{p(\mathbf{X})} = \frac{p(\mathbf{X} \mid \mathbf{Z}) p(\mathbf{Z})}{\int_{\mathcal{Z}^N} p(\mathbf{X} \mid \mathbf{Z})p(\mathbf{Z}) d\mathbf{Z}} $$

However, notice the integral in the denominator of the RHS of the equation. The integral that computes the marginal likelihood $$p(\mathbf{X})$$ is often a high-dimensional intractable integral, unless we have a conjugate prior distribution $$p(\mathbf{Z})$$ for our likelihood function $$p(\mathbf{X} \mid \mathbf{Z})$$. Even then, it still remains a high-dimensional integral. 

This has often been the bottleneck in (full) Bayesian approaches to statistical inference and machine learning, compared to point estimation methods such as maximum likelihood (ML) and maximum a posteriori (MAP). The latter can be done with optimization, while the former requires the evaluation, or at least an approximation, of an intractable integral. While we have a myriad of methods for optimization, solving intractable high-dimensional integrals is still a daunting problem. <!--While Monte Carlo (MC) methods such as Markov chain MC (MCMC) and Hamiltonian MC (HMC) work fairly well, as a professor of mine used to say, MC methods should be the last thing to try in Bayesian inference due to their inefficiency (they need large sample sizes) and pathological behavior in exploring the sample space. -->

*Variatonal Inference (VI)*, or variational Bayes, reduces this inference problem into an *optimization* problem using the KL divergence and the evidence lower bound (ELBO). It's main idea is to specify a family of distributions $$\mathcal{Q}$$ over the latent variable and to find the distribution $$q_{\boldsymbol{\nu}}(\mathbf{Z}) \in \mathcal{Q}$$ that best approximates $$p(\mathbf{Z} \mid \mathbf{X})$$ where $$\boldsymbol{\nu}$$ is the *variational parameter* corresponding to the family $$\mathcal{Q}$$. In short, the task is now optimizing $$\boldsymbol{\nu}$$ such that the "distance" between $$p(\mathbf{Z} \mid \mathbf{X})$$ and $$q_{\boldsymbol{\nu}}(\mathbf{Z})$$ is minimized. 

# KL Divergence

I'll discuss the Kullback-Leibler (KL) divergence in depth in another post, but for now, let us simply look at its definition:

> *Definition*: The (reverse) ***KL divergence*** between two distributions $$p(x)$$ and $$q(x)$$ is given as:

$$ D_{KL}(q(x) \mid\mid p(x)) = \mathbb{E}_{q} \bigg[ \log \frac{q(x)}{p(x)} \bigg] $$ 

Here, we generally treat $$p$$ as the true distribution and $$q$$ as our approximation to $$p$$. Then, observe:

$$ D_{KL}(q(x) \mid\mid p(x)) = \mathbb{E}_{q} \bigg[ \log q(x) - \log p(x) \bigg] $$

In other words, with expectation taken with respect to $$x \sim q$$, the KL divergence is the difference between the log likelihoods of $x$ taken with $$q$$ and $$p$$. Broadly speaking, if $$p$$ and $$q$$ are similar, the KL divergence would be small. In particular, the reverse KL divergence rewards "mode-seeking" behavior; if the mode of $$q$$ is near the mode of $$p$$, then the reverse KL divergence is small. 

With the KL divergence, we have a measure of how different $$q_{\boldsymbol{\nu}}(\mathbf{Z})$$ and $$p(\mathbf{Z} \mid \mathbf{X})$$ are. Then, how do we actually optimize $$\boldsymbol{\nu}$$ such that $$D_{KL}(q_{\boldsymbol{\nu}}(\mathbf{Z}) \mid\mid p(\mathbf{Z} \mid \mathbf{X}))$$ is minimized? 

# Evidence Lower Bound (ELBO)

Unfortunately, directly optimizing $$\boldsymbol{\nu}$$ with respect to minimizing $$D_{KL}(q_{\boldsymbol{\nu}}(\mathbf{Z}) \mid\mid p(\mathbf{Z} \mid \mathbf{X}))$$ is impossible without knowing $$p(\mathbf{Z} \mid \mathbf{X})$$, which is exactly what we're trying to approximate. However, we can use a proxy term to optimize instead. As you might have guessed, that proxy is the evidence lower bound (ELBO).

### Jensen's Inequality

Again, I'll discuss Jensen's inequality in a measure-theoretic setting for probability distributions in another post. For now, let's simply use Jensen's inequality:

> *Lemma*: For $$f: \mathcal{X} \to \mathbb{R}$$ a convex function and $$X \in \mathcal{X}$$ a random variable, we have:

$$ \mathbb{E}[f(X)] \leq f(\mathbb{E}[X]) $$

### ELBO

Using Jensen's inequality and the fact that the $$\log$$ function is convex, we have:

$$\begin{align*}
\log p(\mathbf{X}) & = \log \int_{\mathcal{Z}^N} p(\mathbf{X}, \mathbf{Z}) d\mathbf{Z} \\
& = \log \int_{\mathcal{Z}^N} q_{\boldsymbol{\nu}}(\mathbf{Z}) \frac{p(\mathbf{X}, \mathbf{Z})}{q_{\boldsymbol{\nu}}(\mathbf{Z})} d\mathbf{Z} \\
& = \log \mathbb{E}_{q_{\boldsymbol{\nu}}(\mathbf{Z})} \bigg[ \frac{p(\mathbf{X}, \mathbf{Z})}{q_{\boldsymbol{\nu}}(\mathbf{Z})} \bigg] \\
& \geq \mathbb{E}_{q_{\boldsymbol{\nu}}(\mathbf{Z})} \bigg[\log \frac{p(\mathbf{X}, \mathbf{Z})}{q_{\boldsymbol{\nu}}(\mathbf{Z})} \bigg] \\
& =: \mathcal{L}(q_{\boldsymbol{\nu}}, p ; \mathbf{X})
\end{align*}$$

where this last term is the Evidence Lower Bound (ELBO). Notice it is a *lower bound* on the $$\log$$ *evidence* of $$\mathbf{X}$$; $$p(\mathbf{X})$$ is often referred to as the evidence by Bayesians.

### ELBO and KL Divergence

Finally, we may now look at how the ELBO and the KL divergence are related:

$$\begin{align*}
D_{KL}(q_{\boldsymbol{\nu}}(\mathbf{Z}) \mid\mid p(\mathbf{Z} \mid \mathbf{X})) & = \mathbb{E}_{q_{\boldsymbol{\nu}}(\mathbf{Z})} \bigg[ \log \frac{q_{\boldsymbol{\nu}}(\mathbf{Z})}{p(\mathbf{Z} \mid \mathbf{X})} \bigg] \\
& = \mathbb{E}_{q_{\boldsymbol{\nu}}(\mathbf{Z})} [ \log q_{\boldsymbol{\nu}}(\mathbf{Z}) - \log p(\mathbf{Z} \mid \mathbf{X})] \\
& = \mathbb{E}_{q_{\boldsymbol{\nu}}(\mathbf{Z})} [ \log q_{\boldsymbol{\nu}}(\mathbf{Z})- \log \frac{p(\mathbf{Z}, \mathbf{X})}{p(\mathbf{X})} \bigg] \\
& = \mathbb{E}_{q_{\boldsymbol{\nu}}(\mathbf{Z})} [ \log q_{\boldsymbol{\nu}}(\mathbf{Z}) - \log p(\mathbf{Z}, \mathbf{X})] + p(\mathbf{X}) \\
& = - \mathbb{E}_{q_{\boldsymbol{\nu}}(\mathbf{Z})} \bigg[\log \frac{p(\mathbf{X}, \mathbf{Z})}{q_{\boldsymbol{\nu}}(\mathbf{Z})} \bigg] + p(\mathbf{X}) \\
& = - \mathcal{L}(q_{\boldsymbol{\nu}}, p ; \mathbf{X}) + p(\mathbf{X})
\end{align*}$$

In other words, the KL divergence between $$p(\mathbf{Z} \mid \mathbf{X})$$ and $$q_{\boldsymbol{\nu}}(\mathbf{Z})$$ is the sum of the negative ELBO and the $$\log$$ evidence, where the $$\log$$ evidence is constant with respect to $$\boldsymbol{\nu}$$. Therefore:

$$ \arg\min_\nu D_{KL}(q_{\boldsymbol{\nu}}(\mathbf{Z}) \mid\mid p(\mathbf{Z} \mid \mathbf{X})) = \arg\max_\nu \mathcal{L}(q_{\boldsymbol{\nu}}, p ; \mathbf{X}) $$

Therefore, variational inference is now reduced to the problem of maximizing the ELBO. Unlike the KL divergence, the ELBO can be computed since each of its terms, $$\log p(\mathbf{X}, \mathbf{Z})$$, $$\log q_{\boldsymbol{\nu}}(\mathbf{Z})$$, are tractable with the added bonus that they are in $$\log$$ form, allowing us to use:

$$ \log p(\mathbf{X}) =  \log \prod_{i=1}^{N} p(\mathbf{x}_i) = \sum_{i=1}^{N} \log p(\mathbf{x}_i) $$

# Conclusion

By reducing a Bayesian inference problem into an optimization problem, variational inference opens the door to applying Bayesian inference on high-dimensional problems that are becoming the norm today. A few examples, which I will discuss in other posts, include Bayesian deep learning and variational autoencoders.

### Acknowledgement and Recommended Reading

This post uses the following materials for reference:

1. David M. Blei, Alp Kucukelbir, Jon D. McAuliffe. Variational Inference: A Review for Statisticians, 2016.
2. Diedrik P. Kingma, Max Welling. Auto-Encoding Variational Bayes, 2013. 
