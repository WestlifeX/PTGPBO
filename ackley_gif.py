import torch
import numpy as np
import matplotlib.pyplot as plt
import kernel as krn
import gaussian_process as gaussp
import glob
import imageio

class BayesOptimiser(object):
    
    def __init__(self, gp, acq_func, epsilon=1e-8):
        self.gp = gp
        self.acq_func = acq_func
        self.epsilon = epsilon
    
    def minimise(self, objective, bounds, iters, iter_loops=20,
                 sub_opt_iters=100, sub_opt_lr=0.01, n_inits=3):
        x_hist = None
        y_hist = None
        y_min = float('inf')
        x_min = None
        beta1 = 0.9
        beta2 = 0.99
        
        x_lin = torch.linspace(-8, 8, 501)
        exp1 = -0.2*(0.5*x_lin.pow(2)).pow(0.5)
        exp2 = 0.5*torch.cos(2*np.pi*x_lin)
        y_lin = np.e + 20 - 20*torch.exp(exp1) - torch.exp(exp2)
        x_lin = x_lin.view(501, 1)
        y_lin = y_lin.view(501, 1)
        
        for x_sample in bounds[:,0].view(1,-1) + torch.rand((n_inits,1)) * \
        (bounds[:,1] - bounds[:,0]).view(1,-1):
            y_sample = torch.from_numpy(objective(x_sample.data.numpy())).float()
            if y_sample < y_min:
                y_min = y_sample
                x_min = x_sample
            if x_hist is not None:
                x_hist = torch.cat((x_hist, x_sample.view(1,-1)), dim=0)
                y_hist = torch.cat((y_hist, y_sample.view(1,-1)), dim=0)
            else:
                x_hist = x_sample.view(1,-1)
                y_hist = y_sample.view(1,-1)
        
        for i in range(iters):
            self.gp.kernel.train()
            self.gp.kernel.reset_params()
            self.gp.fit(x_hist, y_hist)
            self.gp.kernel.eval()
            
            with torch.no_grad():
                mu, sigma = self.gp.predict(x_lin, return_var=True)
                mu = mu.flatten().data.numpy()
                std = sigma.pow(0.5).flatten().data.numpy()
                
                plt.figure(figsize=(6,4.5))
                plt.plot(x_lin.flatten().data.numpy(), y_lin.flatten().data.numpy(),
                         color='k', alpha=0.5, linestyle='--')
                plt.plot(x_lin.flatten().data.numpy(), mu, c='r', alpha=0.75)
                plt.fill_between(x_lin.flatten().data.numpy(),
                                 mu + std,
                                 mu - std,
                                 color='r', alpha=0.25)
                plt.scatter(x_hist.flatten().data.numpy(),
                            y_hist.flatten().data.numpy(), c='k')
                plt.title('Iteration: %d   Minimum: %4.2f   At: %4.2f' % \
                          (i+1, y_min, x_min))
                plt.xlim([bounds[0,0],bounds[0,1]])
                plt.ylim([-0.5, y_lin.max().item() + 0.5])
                plt.savefig('ackley_frame_{:04d}.png'.format(i+1))
                plt.show()
                
                y_acq = float('inf')
                
                for j in range(iter_loops):
                    x0 = bounds[:,0] + torch.rand(bounds.size(0)) * (bounds[:,1] - bounds[:,0])
                    V_dx = torch.zeros(x0.size())
                    S_dx = torch.zeros(x0.size())
                    for k in range(sub_opt_iters):
                        dx = self.acq_func.grad(x0, y_min, self.gp).flatten()
                        V_dx = beta1 * V_dx + (1 - beta1) * dx
                        V_dx_corr = V_dx / (1 - beta1**(k+1))
                        S_dx = beta2 * S_dx + (1 - beta2) * dx.pow(2)
                        S_dx_corr = S_dx / (1 - beta2**(k+1))
                        grad = V_dx_corr / (torch.sqrt(S_dx_corr + 1e-8))
                        
                        x0 = x0 - sub_opt_lr * grad
                        x_bounded = torch.min(torch.max(x0.data, bounds[:,0]), bounds[:,1])
                        x0.data = x_bounded
                    acq = self.acq_func(x0, y_min, self.gp)
                    if acq < y_acq: 
                        y_acq = acq
                        x_sample = x0
                
                distances = (x_sample.view(1,-1) - x_hist).pow(2).sum(1).pow(0.5)
                if torch.any(distances <= self.epsilon):
                    x_sample = bounds[:,0] + torch.rand(bounds.size(0)) * (bounds[:,1] - bounds[:,0])
                
                y_sample = torch.from_numpy(objective(x_sample.data.numpy())).float()
                x_hist = torch.cat((x_hist, x_sample.view(1,-1)), dim=0)
                y_hist = torch.cat((y_hist, y_sample.view(1,-1)), dim=0)
                
                if y_sample < y_min:
                    y_min = y_sample
                    x_min = x_sample
        
        return x_min, y_min, x_hist, y_hist

class Acquisition(object):
    
    def __init__(self, beta):
        self.beta = beta
    
    def __call__(self, x, y_min, gp):
        if len(x.size()) < 2:
            x = x.view(1,-1)
        mu, sigma = gp.predict(x, return_var=True)
        out = mu - self.beta*sigma
        return out
    
    def grad(self, x, y_min, gp):
        if len(x.size()) < 2:
            x = x.view(1,-1)
        mu_grad = gp.mu_grad(x)
        sigma_grad = gp.sigma_grad(x)
        grad = mu_grad - self.beta*sigma_grad
        return grad

def ackley(x):
    x = torch.from_numpy(x)
    x = x.view(-1,1)
    exp1 = -0.2*(0.5*x.pow(2).sum(1)).pow(0.5)
    exp2 = 0.5*torch.cos(2*np.pi*x).sum(1)
    y = np.e + 20 - 20*torch.exp(exp1) - torch.exp(exp2)
    return y.data.numpy()

bounds = torch.tensor([[-8.0, 8.0]]).expand(1,2)
iters = 20
kernel = krn.Matern32Kernel(1.0, 1.0) * \
krn.Matern52Kernel(1.0, 1.0)
gp = gaussp.GaussianProcess(kernel, alpha=0.0001)
acq_func = Acquisition(1.0)
optimiser = BayesOptimiser(gp, acq_func)
x_min, y_min, xp, yp = optimiser.minimise(ackley, bounds, iters)

filenames = glob.glob('ackley_frame*.png')
filenames = sorted(filenames)

gif_images = []
for filename in filenames:
    gif_images.append(imageio.imread(filename))
imageio.mimsave('ackley.gif', gif_images, duration=1/2)