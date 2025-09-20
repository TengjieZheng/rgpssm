# Recursive Gaussian Process State-Space Models

This repository contains implementations of RGPSSM, an online learning method for Gaussian process state space models (GPSSMs), enabling flexible and accurate joint state estimation and model learning.

## Implementations

- Implementation of the **RGPSSM** (see `RGPSSM/`)  
  [Paper: *Recursive Gaussian Process State Space Model*](https://arxiv.org/abs/2411.14679)

- Implementation of the **RGPSSM-H** (see `RGPSSM-H/`)  
  [Paper: *Recursive Inference for Heterogeneous Multi-Output GP State-Space Models with Arbitrary Moment Matching*](https://arxiv.org/abs/2411.14679)

## Installation

To ensure proper functionality, please install the Cholesky update and downdate algorithms provided in the `RGPSSM/cholsky` directory.  

```bash
pip install cholsky/cholup-0.0.0.tar.gz
pip install cholsky/choldown-0.0.0.tar.gz
