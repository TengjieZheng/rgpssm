# Recursive Gaussian Process State Space Model

The implementation of RGPSSM, an online learning method for Gaussian process state space models (GPSSMs), enables flexible and accurate joint state estimation and model learning.

## Installation

To ensure proper functionality, you need to install the Cholesky update and downdate algorithms provided in the `RGPSSM/cholsky` directory. The required packages can be installed using the following commands:

```bash
pip install cholsky/cholup-0.0.0.tar.gz
pip install cholsky/choldown-0.0.0.tar.gz
