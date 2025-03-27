import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt

from algorithms.manifolds.math import _mu_bar


def test_mu_bar():
    mu = 0.01215058560962614
    L_i = 1
    mu_bar = _mu_bar(mu, L_i)
    print(mu_bar)


if __name__ == "__main__":
    test_mu_bar()
