# Understanding the Success of Convex Optimization Methods in Deep Neural Networks
## Description
This repository explores how first-order optimization algorithms (SGD, Adam) achieve surprising effectiveness when applied to the non-convex loss landscapes of neural networks. Through systematic empirical analysis of Multi-Layer Perceptrons trained on MNIST, we investigate why these optimizers exhibit convex-like behavior despite operating on theoretically non-convex surfaces.
Our experiments reveal that overparameterized networks create "optimization highways"—dominant gradient directions that guide different training runs along highly correlated paths (correlation > 0.96) through parameter space. Remarkably, while these runs converge to vastly different weight configurations (cosine similarity < 0.1), they achieve consistent performance (accuracy variance < 0.001), providing empirical evidence for effective convexity in deep learning optimization.
To visualize these high-dimensional optimization dynamics, we employ PCA-based dimensionality reduction to project parameter trajectories onto 2D loss landscape surfaces. By tracking parameter vectors throughout training and applying Principal Component Analysis to capture the directions of greatest variance, we create interpretable visualizations that reveal how different optimizers navigate the loss landscape and converge along their respective optimization paths.
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)


## Installation
1. Clone the repo: `git clone (https://github.com/Vinvin202020/Optimization-on-non-convex-landscapes.git)`
2. Navigate to the directory: `cd project`
3. Libraries required :
   Complete Imports List
### PyTorch and Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
### Scientific Computing and Numerical Analysis
import numpy as np
import scipy.linalg as sclg
from scipy.stats import pearsonr
### Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
### Machine Learning and Analysis
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
### Parallel Processing
import joblib
from joblib import Parallel, delayed
### Standard Library
import copy
from collections import defaultdict
import warnings
from functools import partial
import time
import importlib
### Custom Modules
import helpers
from helpers import *


## Usage
Provide examples and code snippets showing how to use your project.

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request



## Author
Your Name – [@yourhandle](https://github.com/yourhandle)
