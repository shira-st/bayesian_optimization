import numpy as np
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from skopt.acquisition import gaussian_ei


class PrecomputedKernel(Kernel):
    def __init__(self, K):
        self.K = K

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        K = self.K[X.flatten(), :][:, Y.flatten()]
        if eval_gradient:
            return K, np.zeros((X.shape[0], X.shape[0], 0))
        else:
            return K

    def diag(self, X):
        return np.diag(self.K)[X.flatten()]

    def is_stationary(self):
        return False


class BayesianOptimization:
    def __init__(self, objective, kernel, initial_size=10, seed=0, noise_level=1e-10):
        self.data_size = len(kernel)
        self.initial_size = initial_size
        self.initial, self.rest = self.split(seed)
        self.objective = objective
        self.kernel = PrecomputedKernel(kernel) + WhiteKernel(noise_level)
        self.best = np.min([self.objective.f(i) for i in range(self.data_size)])
        self.result = np.array([])
        self.acq = gaussian_ei

    # split indices into the indices for initial evaluation and the rest
    def split(self, seed):
        indices = np.array(range(self.data_size))
        np.random.seed(seed)
        np.random.shuffle(indices)
        return np.split(indices, [self.initial_size])

    # execute Bayesian optimization algorithm
    def optimize(self, max_iter=1000):

        # initial evaluation
        y = np.array([self.objective.f(i) for i in self.initial])
        gpr = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True).fit(self.initial.reshape(-1, 1), y)

        i = 0
        t_best = np.min(y)
        self.result = np.append(self.result, t_best)
        while t_best != self.best and i < max_iter:

            # select a next point
            i_next = np.argmax(self.acq(self.rest.reshape(-1, 1), gpr, y_opt=t_best))

            # evaluate next point
            y_next = self.objective.f(self.rest[i_next])
            y = np.append(y, y_next)

            # update indices
            self.initial = np.append(self.initial, self.rest[i_next])
            self.rest = np.delete(self.rest, i_next)

            gpr = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True).fit(self.initial.reshape(-1, 1), y)

            i += 1
            t_best = min(t_best, y_next)
            self.result = np.append(self.result, t_best)

        return
