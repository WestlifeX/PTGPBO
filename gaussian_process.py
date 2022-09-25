import torch
from copy import deepcopy


class GaussianProcess(object):

    def __init__(self, kernel, alpha=1e-5):
        self.kernel = kernel
        self.alpha = alpha
        # Assuming the training data or objective function is noisy,
        # alpha should be set to the standard deviation of the noise

    # nll: negative log likelihood， 为了学超参
    def nll(self, x1, x2, y, det_tol=1e-12):
        b = y.size(0)  # n
        m = y.size(1)  # 1
        k = self.kernel(x1, x2) + self.alpha * torch.eye(b)
        nll = 0.5 * torch.log(torch.det(k) + torch.tensor(det_tol)) + \
              0.5 * y.view(m, b) @ torch.inverse(k) @ y.view(b, m)
        return nll

    # x: nxp, y: nx1
    def fit(self, x, y, lr=0.01, iters=100, restarts=0):
        b = x.size(0)  # n
        n = x.size(1)
        x1 = x.unsqueeze(1).expand(b, b, n)
        x2 = x.unsqueeze(0).expand(b, b, n)
        optimiser = torch.optim.Adam(self.kernel.parameters(), lr)
        assert iters > 0
        for i in range(iters):
            optimiser.zero_grad()
            nll = self.nll(x1, x2, y)
            nll.backward()
            optimiser.step()
        best_nll = nll.item()
        params = self.kernel.get_params()
        for i in range(restarts):
            self.kernel.randomise_params()
            optimiser = torch.optim.Adam(self.kernel.parameters(), lr)
            for j in range(iters):
                optimiser.zero_grad()
                nll = self.nll(x1, x2, y)
                nll.backward()
                optimiser.step()
            if nll.item() < best_nll:
                best_nll = nll
                params = self.kernel.get_params()

        self.kernel.set_params(params)
        self.x = x
        self.y = y
        k = self.kernel(x1, x2).view(b, b) + self.alpha * torch.eye(b)
        self.kinv = torch.inverse(k)

    def predict(self, x, return_var=False):
        b_prime = x.size(0)
        b = self.x.size(0)
        n = self.x.size(1)
        x1 = x.unsqueeze(1).expand(b_prime, b, n)
        x2 = self.x.unsqueeze(0).expand(b_prime, b, n)
        k = self.kernel(x1, x2).view(b_prime, b)
        mu = k @ self.kinv @ self.y
        x = x.unsqueeze(1).expand(b_prime, 1, n)
        sigma = self.kernel(x, x) - (k @ self.kinv @ k.t()).diag().view(mu.size())
        if return_var:
            return mu, sigma
        else:
            return mu

    def mu_grad(self, x):
        b_prime = x.size(0)
        b = self.x.size(0)
        n = self.x.size(1)
        x1 = x.unsqueeze(1).expand(b_prime, b, n)
        x2 = self.x.unsqueeze(0).expand(b_prime, b, n)
        dk = self.kernel.grad(x1, x2)
        grad = (dk * (self.kinv @ self.y)).sum(1)
        return grad

    def sigma_grad(self, x):
        b_prime = x.size(0)
        b = self.x.size(0)
        n = self.x.size(1)
        x1 = x.unsqueeze(1).expand(b_prime, b, n)
        x2 = self.x.unsqueeze(0).expand(b_prime, b, n)
        x = x.unsqueeze(1)
        k = self.kernel(x1, x2)
        dk = self.kernel.grad(x1, x2)
        k_grad = (k @ self.kinv + (self.kinv @ k.t()).t()).unsqueeze(-1)
        grad = (self.kernel.grad(x, x) - k_grad * dk).sum(1)
        return grad


class SpatioTemporalGaussianProcess(object):

    def __init__(self, space_kernel, time_kernel, alpha=1e-5):
        self.space_kernel = space_kernel
        self.time_kernel = time_kernel
        self.alpha = alpha
        # Assuming the training data or objective function is noisy,
        # alpha should be set to the standard deviation of the noise
        # alpha就是超参数σ^2

    # nll: negative log likelihood， 干什么用的还不明确
    def nll(self, x1, x2, t1, t2, y, det_tol=1e-12):
        b = y.size(0)
        m = y.size(1)
        k = self.space_kernel(x1, x2) * self.time_kernel(t1, t2) + \
            self.alpha * torch.eye(b)
        nll = 0.5 * torch.log(torch.det(k) + torch.tensor(det_tol)) + \
              0.5 * y.view(m, b) @ torch.inverse(k) @ y.view(b, m)
        return nll

    # 反正就是用nll作为loss来算k逆
    def fit(self, x, t, y, lr=0.01, iters=100, restarts=0):
        b = x.size(0)
        n = x.size(1)
        x1 = x.unsqueeze(1).expand(b, b, n)
        x2 = x.unsqueeze(0).expand(b, b, n)
        t1 = t.unsqueeze(1).expand(b, b, 1)
        t2 = t.unsqueeze(0).expand(b, b, 1)
        optimiser = torch.optim.Adam(list(self.space_kernel.parameters()) + \
                                     list(self.time_kernel.parameters()), lr)
        for i in range(iters):
            optimiser.zero_grad()
            nll = self.nll(x1, x2, t1, t2, y)
            nll.backward()
            optimiser.step()
        best_nll = nll.item()
        space_params = self.space_kernel.get_params()
        time_params = self.time_kernel.get_params()
        # 为什么还有个restarts
        for i in range(restarts):
            self.kernel.randomise_params()
            optimiser = torch.optim.Adam(list(self.space_kernel.parameters()) + \
                                         list(self.time_kernel.parameters()), lr)
            for j in range(iters):
                optimiser.zero_grad()
                nll = self.nll(x1, x2, t1, t2, y)
                nll.backward()
                optimiser.step()
            if nll.item() < best_nll:
                best_nll = nll
                space_params = self.space_kernel.get_params()
                time_params = self.time_kernel.get_params()

        self.space_kernel.set_params(space_params)
        self.time_kernel.set_params(time_params)
        self.x = x
        self.t = t
        self.y = y
        k = self.space_kernel(x1, x2) * self.time_kernel(t1, t2) + \
            self.alpha * torch.eye(b)
        self.kinv = torch.inverse(k)

    # predict没啥问题，按照公式来
    def predict(self, x, t, return_var=False):
        b_prime = x.size(0)
        b = self.x.size(0)
        n = self.x.size(1)
        x1 = x.unsqueeze(1).expand(b_prime, b, n)
        x2 = self.x.unsqueeze(0).expand(b_prime, b, n)
        t1 = t.unsqueeze(1).expand(b_prime, b, 1)
        t2 = self.t.unsqueeze(0).expand(b_prime, b, 1)
        k = self.space_kernel(x1, x2) * self.time_kernel(t1, t2)
        mu = k @ self.kinv @ self.y
        x = x.unsqueeze(1).expand(b_prime, 1, n)
        t = t.unsqueeze(1).expand(b_prime, 1, 1)
        sigma = self.space_kernel(x, x) * self.time_kernel(t, t) - \
                (k @ self.kinv @ k.t()).diag().view(mu.size())
        if return_var:
            return mu, sigma
        else:
            return mu

    def mu_grad(self, x, t):
        b_prime = x.size(0)
        b = self.x.size(0)
        n = self.x.size(1)
        x1 = x.unsqueeze(1).expand(b_prime, b, n)
        x2 = self.x.unsqueeze(0).expand(b_prime, b, n)
        t1 = t.unsqueeze(1).expand(b_prime, b, 1)
        t2 = self.t.unsqueeze(0).expand(b_prime, b, 1)
        dk = self.space_kernel.grad(x1, x2) * self.time_kernel(t1, t2).unsqueeze(-1)
        grad = (dk * (self.kinv @ self.y)).sum(1)
        return grad

    def sigma_grad(self, x, t):
        b_prime = x.size(0)
        b = self.x.size(0)
        n = self.x.size(1)
        x1 = x.unsqueeze(1).expand(b_prime, b, n)
        x2 = self.x.unsqueeze(0).expand(b_prime, b, n)
        t1 = t.unsqueeze(1).expand(b_prime, b, 1)
        t2 = self.t.unsqueeze(0).expand(b_prime, b, 1)
        x = x.unsqueeze(1)
        t = t.unsqueeze(1).expand(b_prime, 1, 1)
        k = self.space_kernel(x1, x2) * self.time_kernel(t1, t2)
        dk = self.space_kernel.grad(x1, x2) * self.time_kernel(t1, t2).unsqueeze(-1)
        k_grad = (k @ self.kinv + (self.kinv @ k.t()).t()).unsqueeze(-1)
        grad = (self.space_kernel.grad(x, x) * \
                self.time_kernel(t, t).unsqueeze(-1) - k_grad * dk).sum(1)
        return grad


class AutomaticGaussianProcess(object):

    def __init__(self, kernel_list, init_kernel=None, alpha=1e-5):
        self.kernel_list = kernel_list
        self.kernel = init_kernel
        self.alpha = alpha

    def nll(self, x1, x2, y, det_tol=1e-12):
        b = y.size(0)
        m = y.size(1)
        k = self.kernel(x1, x2) + self.alpha * torch.eye(b)
        nll = 0.5 * torch.log(torch.det(k) + torch.tensor(det_tol)) + \
              0.5 * y.view(m, b) @ torch.inverse(k) @ y.view(b, m)
        return nll

    def fit(self, x, y, kernel_search_iters=3, lr=0.01, iters=100, restarts=0,
            verbose=False):
        b = x.size(0)
        n = x.size(1)
        x1 = x.unsqueeze(1).expand(b, b, n)
        x2 = x.unsqueeze(0).expand(b, b, n)

        previous_kernel_nll = None
        no_improvement = False

        for s in range(kernel_search_iters):
            if no_improvement:
                break
            elif self.kernel is None:
                best_kernel_nll = float('inf')
                for k in range(len(self.kernel_list)):
                    self.kernel = deepcopy(self.kernel_list[k])
                    optimiser = torch.optim.Adam(self.kernel.parameters(), lr)
                    for i in range(iters):
                        optimiser.zero_grad()
                        nll = self.nll(x1, x2, y)
                        nll.backward()
                        optimiser.step()
                    best_nll = nll.item()
                    if best_nll < best_kernel_nll:
                        best_kernel_nll = best_nll
                        previous_kernel_nll = best_kernel_nll
                        best_kernel = deepcopy(self.kernel)
                    for i in range(restarts):
                        self.kernel.randomise_params()
                        optimiser = torch.optim.Adam(self.kernel.parameters(), lr)
                        for j in range(iters):
                            optimiser.zero_grad()
                            nll = self.nll(x1, x2, y)
                            nll.backward()
                            optimiser.step()
                        if nll.item() < best_nll:
                            best_nll = nll
                        if best_nll < best_kernel_nll:
                            best_kernel_nll = best_nll
                            best_kernel = deepcopy(self.kernel)
                self.kernel = best_kernel
                previous_kernel_nll = best_kernel_nll
                if verbose:
                    print('\nsearch iter: %d, best nll: %f' % (s + 1, best_kernel_nll))
                    print(self.kernel)
            else:
                if previous_kernel_nll is None:
                    previous_kernel = deepcopy(self.kernel)
                    self.kernel.reset_params()
                    optimiser = torch.optim.Adam(self.kernel.parameters(), lr)
                    for i in range(iters):
                        optimiser.zero_grad()
                        nll = self.nll(x1, x2, y)
                        nll.backward()
                        optimiser.step()
                    best_nll = nll.item()
                    best_kernel_nll = best_nll
                    best_kernel = deepcopy(self.kernel)
                    for i in range(restarts):
                        self.kernel.randomise_params()
                        optimiser = torch.optim.Adam(self.kernel.parameters(), lr)
                        for j in range(iters):
                            optimiser.zero_grad()
                            nll = self.nll(x1, x2, y)
                            nll.backward()
                            optimiser.step()
                        if nll.item() < best_nll:
                            best_nll = nll
                            best_kernel_nll
                            best_kernel = deepcopy(self.kernel)
                    previous_kernel_nll = best_kernel_nll
                else:
                    previous_kernel = deepcopy(self.kernel)
                    best_kernel_nll = previous_kernel_nll
                    best_kernel = previous_kernel

                for k in range(len(self.kernel_list)):
                    self.kernel = deepcopy(previous_kernel) + \
                                  deepcopy(self.kernel_list[k])
                    self.kernel.reset_params()
                    optimiser = torch.optim.Adam(self.kernel.parameters(), lr)
                    for i in range(iters):
                        optimiser.zero_grad()
                        nll = self.nll(x1, x2, y)
                        nll.backward()
                        optimiser.step()
                    best_nll = nll.item()
                    if best_nll < best_kernel_nll:
                        best_kernel_nll = best_nll
                        best_kernel = deepcopy(self.kernel)
                    for i in range(restarts):
                        self.kernel.randomise_params()
                        optimiser = torch.optim.Adam(self.kernel.parameters(), lr)
                        for j in range(iters):
                            optimiser.zero_grad()
                            nll = self.nll(x1, x2, y)
                            nll.backward()
                            optimiser.step()
                        if nll.item() < best_nll:
                            best_nll = nll
                        if best_nll < best_kernel_nll:
                            best_kernel_nll = best_nll
                            best_kernel = deepcopy(self.kernel)

                    self.kernel = deepcopy(previous_kernel) * \
                                  deepcopy(self.kernel_list[k])
                    optimiser = torch.optim.Adam(self.kernel.parameters(), lr)
                    for i in range(iters):
                        optimiser.zero_grad()
                        nll = self.nll(x1, x2, y)
                        nll.backward()
                        optimiser.step()
                    best_nll = nll.item()
                    if best_nll < best_kernel_nll:
                        best_kernel_nll = best_nll
                        best_kernel = deepcopy(self.kernel)
                    for i in range(restarts):
                        self.kernel.randomise_params()
                        optimiser = torch.optim.Adam(self.kernel.parameters(), lr)
                        for j in range(iters):
                            optimiser.zero_grad()
                            nll = self.nll(x1, x2, y)
                            nll.backward()
                            optimiser.step()
                        if nll.item() < best_nll:
                            best_nll = nll
                        if best_nll < best_kernel_nll:
                            best_kernel_nll = best_nll
                            best_kernel = deepcopy(self.kernel)
                if previous_kernel_nll == best_kernel_nll:
                    no_improvement = True
                self.kernel = best_kernel
                previous_kernel_nll = best_kernel_nll
                if verbose:
                    print('\nsearch iter: %d, best nll: %f' % (s + 1, best_kernel_nll))
                    print(self.kernel)

        self.x = x
        self.y = y
        k = self.kernel(x1, x2).view(b, b) + self.alpha * torch.eye(b)
        self.kinv = torch.inverse(k)

    def predict(self, x, return_var=False):
        b_prime = x.size(0)
        b = self.x.size(0)
        n = self.x.size(1)
        x1 = x.unsqueeze(1).expand(b_prime, b, n)
        x2 = self.x.unsqueeze(0).expand(b_prime, b, n)
        k = self.kernel(x1, x2).view(b_prime, b)
        mu = k @ self.kinv @ self.y
        x = x.unsqueeze(1).expand(b_prime, 1, n)
        sigma = self.kernel(x, x) - (k @ self.kinv @ k.t()).diag().view(mu.size())
        if return_var:
            return mu, sigma
        else:
            return mu

    def mu_grad(self, x):
        b_prime = x.size(0)
        b = self.x.size(0)
        n = self.x.size(1)
        x1 = x.unsqueeze(1).expand(b_prime, b, n)
        x2 = self.x.unsqueeze(0).expand(b_prime, b, n)
        dk = self.kernel.grad(x1, x2)
        grad = (dk * (self.kinv @ self.y)).sum(1)
        return grad

    def sigma_grad(self, x):
        b_prime = x.size(0)
        b = self.x.size(0)
        n = self.x.size(1)
        x1 = x.unsqueeze(1).expand(b_prime, b, n)
        x2 = self.x.unsqueeze(0).expand(b_prime, b, n)
        x = x.unsqueeze(1)
        k = self.kernel(x1, x2)
        dk = self.kernel.grad(x1, x2)
        k_grad = (k @ self.kinv + (self.kinv @ k.t()).t()).unsqueeze(-1)
        grad = (self.kernel.grad(x, x) - k_grad * dk).sum(1)
        return grad
