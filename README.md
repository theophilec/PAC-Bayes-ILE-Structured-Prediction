# PAC-Bayesian ILE Structured Prediction

This repository accompanies the JMLR submission **A PAC-Bayesian Perspective on Structured Prediction with Implicit Loss Embeddings** by Théophile Cantelobre (Mines ParisTech, Inria), Benjamin Guedj (Inria, UCL), María Pérez-Ortiz (UCL) and John Shawe-Taylor (UCL).

The pre-print is available here: *link coming soon*.

## Requirements

On top of standard machine learning requirements, this repository requires `jax` (for auto-diff) and `scikit-ml` (for data).

## Getting started

We use the `scikit-learn` API as much as possible... you should be able to jump right in. To get started, you can take a look at `demo_relaxed.py` and `demo_mc.py`.

Jupyter notebooks are for reproducing figures (`Section-7.ipynb` depends on `sensitivity-*.py`, which takes a while to run).

## Contact

If you have any issues with the code, feel free to open an issue. To chat about the paper, ping me at `theophilec#gmail.com`.
