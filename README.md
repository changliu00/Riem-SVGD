# [Riemannian Stein Variational Gradient Descent for Bayesian Inference](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17275)
[Chang Liu][changliu] \<<chang-li14@mails.tsinghua.edu.cn> (deprecated); <liuchangsmail@gmail.com>\>,
and [Jun Zhu][junzhu]. AAAI 2018.

\[[Paper](http://ml.cs.tsinghua.edu.cn/~changliu/rsvgd/Liu-Zhu.pdf)\]
\[[Appendix](http://ml.cs.tsinghua.edu.cn/~changliu/rsvgd/Liu-Zhu-appendix.pdf)\]
\[[Slides](http://ml.cs.tsinghua.edu.cn/~changliu/rsvgd/rsvgd_beamer.pdf)\]
\[[Poster](http://ml.cs.tsinghua.edu.cn/~changliu/rsvgd/rsvgd_poster.pdf)\]

# Introduction

The repository implements the proposed methods, Riemannian Stein Variational Inference Descent (RSVGD)
in both coordinate space and embedded space, and their application in Bayesian Logistic Regression (BLR)
and Spherical Admixture Model (SAM) ([Reisinger et al., 2010](https://icml.cc/Conferences/2010/papers/45.pdf)).
The repository also includes implementations of baseline methods:
Stein Variational Gradient Descent (SVGD) [(Liu & Wang, 2016)][svgd-paper]
based on their [codes][svgd-codes] for the BLR experiment,
and Stochastic Gradient Geodesic Monte Carlo (SGGMC) and geodesic Stochastic Gradient Nose-Hoover Thermostats (gSGNHT)
[(Liu et al., 2016)][sggmc-paper] based on their [codes][sggmc-codes] for the SAM experiment.

RSVGD is the first particle-based variational inference method on Riemannian manifolds.
For Bayesian inference tasks with Euclidean support space, as is the case for BLR,
RSVGD can utilize [information geometry](https://en.wikipedia.org/wiki/Information_geometry)
implemented in the coordinate space of the Fisher distribution manifold to speed up convergence over SVGD,
and for tasks with Riemannian manifold support, especially manifolds with no global coordinate systems
like hyperspheres as is the case for SAM, SVGD is not applicable,
while RSVGD can tackle the manifold structure efficiently in the embedded space of the manifold.
In either task, RSVGD achieves better results than classical parameter-based variational inference methods,
and is more iteration- and particle-efficient than MCMC methods.

## Instructions

### Bayesian Logistic Regression (BLR)

Corresponds to the folder "bayesian_logistic_regression/".

* Codes:

  Implemented in Python with NumPy based on the codes by [Liu & Wang (2016)][svgd-codes].

* Data:

  We did not contribute to the data sets here. The data sets used in the experiment are:

  - "covertype.mat":
    Covertype dataset from the [LIBSVM Data Repository](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html).
    It is also used in SVGD by [Liu & Wang (2016)][svgd-paper].
  - "benchmarks.mat":
    benchmark datasets compiled by [Mika et al. (1999)](http://theoval.cmp.uea.ac.uk/matlab/benchmarks/benchmarks.mat).

### Spherical Admixture Model (SAM)

Corresponds to the folder "SAM/".

* Codes:

  - RSVGD, GMC and SGGMC are implemented in C++ based on the codes by [Liu et al. (2016)][sggmc-codes].
    The codes employ [Eigen](http://eigen.tuxfamily.org/) for linear algebra
    and [OpenMP](http://openmp.org/) for paralellization.
  - Variational inference methods are implemented in MATLAB also based on the codes by [Liu et al. (2016)][sggmc-codes].
  - To compile the C++ codes, just type "make" in each subfolder.
    See the instructions by [Liu et al. (2016)][sggmc-codes] for more details.

* Data:

  We did not contribute to the data set here. The data set used in the experiment is:

  - "20News-diff":
    The dataset is processed and used by [Liu et al. (2016)][sggmc-codes].
    It is a subset of the [20Newsgroups dataset](http://www.qwone.com/~jason/20Newsgroups/)
    with normalized tf-idf feature.
    See the [dataset website](http://ml.cs.tsinghua.edu.cn/~changliu/sggmcmc-sam/)
    or its ["README.md" file](./SAM/data/20News-diff/README.md) for more details.

## Citation
```
	@inproceedings{liu2018riemannian,
	  title={{R}iemannian {S}tein variational gradient descent for {B}ayesian inference},
	  author={Liu, Chang and Zhu, Jun},
	  booktitle={The 32nd AAAI Conference on Artificial Intelligence},
	  pages={3627--3634},
	  year={2018},
	  organization={AAAI press},
	  address={New Orleans, Louisiana USA},
	  url = {https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17275},
	  keywords = {Bayesian inference; Riemann manifold; Information geometry; kernel methods},
	}
```

[changliu]: http://ml.cs.tsinghua.edu.cn/~changliu/index.html
[junzhu]: http://ml.cs.tsinghua.edu.cn/~jun/index.shtml
[svgd-paper]: http://papers.nips.cc/paper/6338-stein-variational-gradient-descent-a-general-purpose-bayesian-inference-algorithm
[svgd-codes]: https://github.com/DartML/Stein-Variational-Gradient-Descent
[sggmc-paper]: http://papers.nips.cc/paper/6281-stochastic-gradient-geodesic-mcmc-methods
[sggmc-codes]: https://github.com/chang-ml-thu/SG-Geod-MCMC

