# Variational Inference with Normalizing Flows

This project implements Variational Autoencoders (VAEs) with and without Normalizing flows. Normalizing flows include Planar, Radial and Sylvester transformation based flows.

## Getting started
### Prerequisities

This project uses `pytorch` framework. You will need `numpy` as well. Other, optional dependencies include `scipy` and `matplotlib`

### Project layout

The core functionality is in the NF subfolder. This is a python module which covers defining, running and training of the VAEs, with and without Normalizing Flows.

There are also several notebooks showing the application of VAEs on different datasets.

## References

1. Rezende, D. J., & Mohamed, S. (2015). Variational inference with normalizing flows. arXiv preprint arXiv:1505.05770.
2. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
3. Berg, R. V. D., Hasenclever, L., Tomczak, J. M., & Welling, M. (2018). Sylvester normalizing flows for variational inference. arXiv preprint arXiv:1803.05649.
