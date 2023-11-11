from turbo import TurboM, nearest
import numpy as np
import torch
import math
import matplotlib
import matplotlib.pyplot as plt
import copy

db = {}
with open("db") as db_file:
    lines = db_file.readlines()
    for line in lines:
        config = line.split()
        db[" ".join(config[:-1])] = -float(config[-1])
        
        
class Levy:
    def __init__(self, dim=8):
        self.dim = dim
        self.lb = np.array([14, 5, 5, 1, 1, 4, 5, 0])
        self.ub = np.array([18, 9, 9, 18, 18, 8, 8, 3])
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        y = nearest(x)
        # print(y)
        if " ".join(map(str, [32, int(y[0]), int(y[0]), int(y[1]), int(y[2]), 32, 16, int(y[3]), 1, int(y[4]), 1, int(y[5]), 16, int(y[6]), 32, int(y[7])])) in db:
            return db[" ".join(map(str, [32, int(y[0]), int(y[0]), int(y[1]), int(y[2]), 32, 16, int(y[3]), 1, int(y[4]), 1, int(y[5]), 16, int(y[6]), 32, int(y[7])]))]
        else:
            return 0.

f = Levy()

turbo_m = TurboM(
    f=f,  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=10,  # Number of initial bounds from an Symmetric Latin hypercube design
    max_evals=1000,  # Maximum number of evaluations
    n_trust_regions=10,  # Number of trust regions
    batch_size=10,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
)

turbo_m.optimize()