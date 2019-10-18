import torch
import numpy as np
import kernel as k
import gaussian_process as gp
import bayes_opt as bo

def ackley(x):
    exp1 = -0.2*(0.5*x.pow(2)).sum().pow(0.5)
    exp2 = 0.5*torch.cos(2*np.pi*x).sum()
    y = np.e + 20 - 20*torch.exp(exp1) - torch.exp(exp2)
    return y

n_dimensions = 2
bounds = torch.tensor([[-10.0, 10.0]]).expand(n_dimensions, 2)
iters = 50
kernel = k.Matern32Kernel(1.0, 1.0)
model = gp.GaussianProcess(kernel, 0.001)
acq_func = bo.Acquisition(1.0)
optimiser = bo.BayesOptimiser(model, acq_func)
x_min, y_min, x_hist, y_hist = optimiser.minimise(ackley, bounds, iters,
                                                  iter_loops=10)