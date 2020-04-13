# nngpt
This project is for fast-ish reconstruction using the concept of Gaussian
process tomography (see [Svensson, J.](http://www.euro-fusionscipub.org/wp-content/uploads/eurofusion/EFDP11024.pdf), [Li, Dong, eta al](https://doi.org/10.1063/1.4817591),
and [Wang, T., et al](https://doi.org/10.1063/1.5023162)) while applying
nonnegativity constraints.  The motivation for this is the benefit that
nonnegative constraints provide in the context of reconstruction from
generalized detector readout.  Please see
[arXiv:1912.01058](https://arxiv.org/abs/1912.01058) for a preprint article
explaining the motivating use case.

## Examples
### Planar detectors with linear projections and coded segmentation
* 2-D: [launch in Google Colab](https://colab.research.google.com/github/decibelcooper/nngpt/blob/master/notebooks/2-D.ipynb)
* 3-D: [launch in Google Colab](https://colab.research.google.com/github/decibelcooper/nngpt/blob/master/notebooks/3-D.ipynb)
* Coded: [launch in Google Colab](https://colab.research.google.com/github/decibelcooper/nngpt/blob/master/notebooks/Coded.ipynb)
* Hybrid Coded and Pixelated: [launch in Google Colab](https://colab.research.google.com/github/decibelcooper/nngpt/blob/master/notebooks/Hybrid%20Coded%20and%20Pixelated.ipynb)
