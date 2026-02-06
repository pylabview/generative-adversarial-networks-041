# Assignment 4 - Generative Adversarial Networks V1

## Q1: Minimax loss in GANs and why it creates “competitive” training

GANs are framed as a **two-player minimax game** where the discriminator $D$ tries to correctly classify real vs. fake samples, while the generator $G$ tries to produce fake samples that **fool** $D$ (Dingari, n.d.; Goodfellow et al., 2014).040---Module-4---Generative-Adv… The classic objective is:
$$
\min_{G}\; \max_{D}\; V(D,G)=\mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))].
$$
This ensures competitive training because improving $D$ makes it harder for $G$ to succeed, forcing $G$ to generate more realistic samples; conversely, as $G$ improves, $D$ must refine its decision boundary to keep distinguishing real from fake (Dingari, n.d.).040---Module-4---Generative-Adv… In practice, training alternates updates to $D$ and $G$ so neither model “wins” too quickly, helping both improve together (Dingari, n.d.).040---Module-4---Generative-Adv…

------

## Q2: Mode collapse—what it is, why it happens, and mitigation

**Mode collapse** happens when the generator produces **low-diversity outputs**, often repeating the same (or very similar) samples instead of covering the full variety (modes) of the real data distribution (Dingari, n.d.). A simple way to express it is that many latent inputs map to essentially the same output:
$$
G(z_1)\approx G(z_2)\approx \cdots \approx x^{*}\quad \text{for many different } z.
$$
Mode collapse can occur because the adversarial objective rewards $G$ for generating *any* samples that fool $D$, so $G$ may find a “shortcut” by concentrating probability mass on a small set of outputs that reliably trick $D$, especially under unstable dynamics when one network overpowers the other (Dingari, n.d.). Mitigations include stabilizing and diversity-promoting techniques such as **minibatch discrimination**, **batch normalization**, and using more stable objectives like **Wasserstein GAN (WGAN)**, which the course notes highlight as helping reduce instability and lowering the likelihood of mode collapse (Arjovsky et al., 2017; Dingari, n.d.; Salimans et al., 2016).

------

## Q3: Role of the discriminator in adversarial training

The discriminator is the **learning signal provider**: it is trained as a binary classifier that outputs $D(x)\in[0,1]$, the estimated probability that input $x$ is real, and it learns by seeing both real samples and generator-produced fakes (Dingari, n.d.).040---Module-4---Generative-Adv… A standard discriminator loss is:
$$
\mathcal{L}_D
= -\mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)]
\;-\;
\mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))].
$$
During training, $D$ improves its ability to detect fake data, while $G$ updates using gradients that flow *through* $D$, improving generation specifically in directions that make fakes harder to detect (Dingari, n.d.; Goodfellow et al., 2014).040---Module-4---Generative-Adv… This adversarial loop—alternating updates to $D$ and $G$—creates the dynamic competition that drives GAN learning (Dingari, n.d.).040---Module-4---Generative-Adv…

------

## Q4: How IS and FID evaluate GAN performance

**Inception Score (IS)** evaluates generated images using a pretrained classifier (Inception network). It rewards images that (1) yield **confident class predictions** $p(y|x)$ (suggesting sharp/recognizable samples) and (2) yield a **diverse** marginal label distribution $p(y)$ across many samples (Salimans et al., 2016). A common definition is:
$$
IS = \exp\left(\mathbb{E}_{x\sim p_g}\left[ D_{\mathrm{KL}}\big(p(y|x)\,\|\,p(y)\big)\right]\right).
$$
**Fréchet Inception Distance (FID)** compares the **feature distributions** of real vs. generated images (typically Inception features) by modeling each set as a Gaussian with mean $\mu$ and covariance $\Sigma$, then computing a distance between them; lower is better and indicates generated samples match real data statistics more closely (Heusel et al., 2017). The standard formula is:
$$
FID=\|\mu_r-\mu_g\|_2^2+\mathrm{Tr}\left(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\right).
$$
In short, IS focuses on classifier confidence + label diversity, while FID directly measures how close generated and real feature distributions are—often making FID more sensitive to both realism and mode coverage (Heusel et al., 2017; Salimans et al., 2016).

------

## References

Arjovsky, M., Chintala, S., & Bottou, L. (2017). *Wasserstein GAN*. In **Proceedings of the 34th International Conference on Machine Learning (ICML 2017)** (pp. 214–223). PMLR.

Dingari, N. C. (n.d.). *Module 4: Generative Adversarial Networks (GANs)* [Course handout]. DS552 – Generative AI.040---Module-4---Generative-Adv…

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. In **Advances in Neural Information Processing Systems** (Vol. 27).

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In **Advances in Neural Information Processing Systems** (Vol. 30).

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training GANs. In **Advances in Neural Information Processing Systems** (Vol. 29).