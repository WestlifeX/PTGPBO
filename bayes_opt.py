import torch
import sys


class BayesOptimiser(object):

    def __init__(self, gp, acq_func, epsilon=1e-8):
        self.gp = gp
        self.acq_func = acq_func
        self.epsilon = epsilon

    def minimise(self, objective, bounds, iters, iter_loops=20,
                 sub_opt_iters=100, sub_opt_lr=0.05, n_inits=5, verbose=True):
        x_hist = None
        y_hist = None
        y_min = float('inf')
        x_min = None
        beta1 = 0.9
        beta2 = 0.99

        for x_sample in bounds[:, 0].view(1, -1) + torch.rand((n_inits, 1)) * \
                        (bounds[:, 1] - bounds[:, 0]).view(1, -1):
            y_sample = objective(x_sample)
            if y_sample < y_min:
                y_min = y_sample
                x_min = x_sample
            if x_hist is not None:
                x_hist = torch.cat((x_hist, x_sample.view(1, -1)), dim=0)
                y_hist = torch.cat((y_hist, y_sample.view(1, -1)), dim=0)
            else:
                x_hist = x_sample.view(1, -1)
                y_hist = y_sample.view(1, -1)

        for i in range(iters):
            self.gp.kernel.train()
            self.gp.kernel.reset_params()
            self.gp.fit(x_hist, y_hist)
            self.gp.kernel.eval()
            y_acq = float('inf')

            with torch.no_grad():
                for j in range(iter_loops):
                    x0 = bounds[:, 0] + torch.rand(bounds.size(0)) * (bounds[:, 1] - bounds[:, 0])
                    V_dx = torch.zeros(x0.size())
                    S_dx = torch.zeros(x0.size())
                    for k in range(sub_opt_iters):
                        dx = self.acq_func.grad(x0, y_min, self.gp).flatten()
                        V_dx = beta1 * V_dx + (1 - beta1) * dx
                        V_dx_corr = V_dx / (1 - beta1 ** (k + 1))
                        S_dx = beta2 * S_dx + (1 - beta2) * dx.pow(2)
                        S_dx_corr = S_dx / (1 - beta2 ** (k + 1))
                        grad = V_dx_corr / (torch.sqrt(S_dx_corr + 1e-8))

                        x0 = x0 - sub_opt_lr * grad
                        x_bounded = torch.min(torch.max(x0.data, bounds[:, 0]), bounds[:, 1])
                        x0.data = x_bounded
                    acq = self.acq_func(x0, y_min, self.gp)
                    if acq < y_acq:
                        y_acq = acq
                        x_sample = x0

                distances = (x_sample.view(1, -1) - x_hist).pow(2).sum(1).pow(0.5)
                if torch.any(distances <= self.epsilon):
                    x_sample = bounds[:, 0] + torch.rand(bounds.size(0)) * (bounds[:, 1] - bounds[:, 0])

                y_sample = objective(x_sample)
                x_hist = torch.cat((x_hist, x_sample.view(1, -1)), dim=0)
                y_hist = torch.cat((y_hist, y_sample.view(1, -1)), dim=0)

                if y_sample < y_min:
                    y_min = y_sample
                    x_min = x_sample

                if verbose:
                    sys.stdout.write('\rIteration: %d   Minimum: %4.2f   ' % \
                                     (i + 1, y_min))
                    sys.stdout.flush()

        return x_min, y_min, x_hist, y_hist


class DynamicBayesOptimiser(object):

    def __init__(self, gp, acq_func, epsilon=1e-8):
        self.gp = gp
        self.acq_func = acq_func
        self.epsilon = epsilon  # epsilon什么用

    def minimise(self, objective, bounds, iters, iter_loops=20,
                 sub_opt_iters=100, sub_opt_lr=0.01, n_inits=5, verbose=True):
        x_hist = None
        t_hist = None
        y_hist = None
        y_min = float('inf')
        x_min = None
        beta1 = 0.9
        beta2 = 0.99

        # 就是从bound的下界遍历到上界，不过上下界加了点高斯噪声
        for x_sample in bounds[:, 0].view(1, -1) + torch.rand((n_inits, 1)) * \
                        (bounds[:, 1] - bounds[:, 0]).view(1, -1):
            y_sample = objective(x_sample)
            if y_sample < y_min:
                y_min = y_sample
                x_min = x_sample
            if x_hist is not None:
                x_hist = torch.cat((x_hist, x_sample.view(1, -1)), dim=0)
                t_hist = torch.cat((t_hist, t_hist[-1] + torch.ones((1, 1))), dim=0)
                y_hist = torch.cat((y_hist, y_sample.view(1, -1)), dim=0)
            else:
                x_hist = x_sample.view(1, -1)
                t_hist = torch.ones((1, 1))
                y_hist = y_sample.view(1, -1)

        for i in range(iters):
            self.gp.space_kernel.train()  # train是哪来的，train啥
            self.gp.time_kernel.train()
            self.gp.space_kernel.reset_params()
            self.gp.time_kernel.reset_params()
            self.gp.fit(x_hist, t_hist, y_hist)
            self.gp.space_kernel.eval()
            self.gp.time_kernel.eval()
            y_acq = float('inf')

            with torch.no_grad():
                for j in range(iter_loops):
                    x0 = bounds[:, 0] + torch.rand(bounds.size(0)) * (bounds[:, 1] - bounds[:, 0])
                    V_dx = torch.zeros(x0.size())
                    S_dx = torch.zeros(x0.size())
                    for k in range(sub_opt_iters):
                        dx = self.acq_func.grad(x0, t_hist[-1] + torch.ones((1, 1)), y_min, self.gp).flatten()
                        V_dx = beta1 * V_dx + (1 - beta1) * dx
                        V_dx_corr = V_dx / (1 - beta1 ** (k + 1))
                        S_dx = beta2 * S_dx + (1 - beta2) * dx.pow(2)
                        S_dx_corr = S_dx / (1 - beta2 ** (k + 1))
                        grad = V_dx_corr / (torch.sqrt(S_dx_corr + 1e-8))

                        x0 = x0 - sub_opt_lr * grad
                        x_bounded = torch.min(torch.max(x0.data, bounds[:, 0]), bounds[:, 1])
                        x0.data = x_bounded
                    acq = self.acq_func(x0, t_hist[-1] + torch.ones((1, 1)), y_min, self.gp)
                    if acq < y_acq:
                        y_acq = acq
                        x_sample = x0

                distances = (x_sample.view(1, -1) - x_hist).pow(2).sum(1).pow(0.5)
                if torch.any(distances <= self.epsilon):
                    x_sample = bounds[:, 0] + torch.rand(bounds.size(0)) * (bounds[:, 1] - bounds[:, 0])

                y_sample = objective(x_sample)
                x_hist = torch.cat((x_hist, x_sample.view(1, -1)), dim=0)
                t_hist = torch.cat((t_hist, t_hist[-1] + torch.ones((1, 1))), dim=0)
                y_hist = torch.cat((y_hist, y_sample.view(1, -1)), dim=0)

                if y_sample < y_min:
                    y_min = y_sample
                    x_min = x_sample

                if verbose:
                    sys.stdout.write('\rIteration: %d   Minimum: %4.2f   ' % \
                                     (i + 1, y_min))
                    sys.stdout.flush()

        return x_min, y_min, x_hist, y_hist


class Acquisition(object):

    def __init__(self, beta):
        self.beta = beta

    def __call__(self, x, y_min, gp):
        if len(x.size()) < 2:
            x = x.view(1, -1)
        mu, sigma = gp.predict(x, return_var=True)
        out = mu - self.beta * sigma
        return out

    def grad(self, x, y_min, gp):
        if len(x.size()) < 2:
            x = x.view(1, -1)
        mu_grad = gp.mu_grad(x)
        sigma_grad = gp.sigma_grad(x)
        grad = mu_grad - self.beta * sigma_grad
        return grad


class DynamicAcquisition(object):

    def __init__(self, beta):
        self.beta = beta

    def __call__(self, x, t, y_min, gp):
        if len(x.size()) < 2:
            x = x.view(1, -1)
            t = t.view(1, -1)
        mu, sigma = gp.predict(x, t, return_var=True)
        out = mu - self.beta * sigma
        return out

    def grad(self, x, t, y_min, gp):
        if len(x.size()) < 2:
            x = x.view(1, -1)
            t = t.view(1, -1)
        mu_grad = gp.mu_grad(x, t)
        sigma_grad = gp.sigma_grad(x, t)
        grad = mu_grad - self.beta * sigma_grad
        return grad
