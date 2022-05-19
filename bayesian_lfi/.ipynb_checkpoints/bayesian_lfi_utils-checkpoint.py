from typing import Union
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from sbi.utils import MultipleIndependent
from tqdm import tqdm


def func(start, end, posterior, n, b_prime_x):
    posterior_means_qr = []
    posterior_vars_qr = []
    for i in tqdm(range(start, end)):
        posterior_samples_qr = posterior.sample(sample_shape=(n,), x=b_prime_x[i, :])
        posterior_means_qr.append(torch.mean(posterior_samples_qr, dim=0))
        posterior_vars_qr.append(torch.cov(torch.transpose(posterior_samples_qr, 0, 1)))
    return posterior_means_qr, posterior_vars_qr


class GMMSimulator:
    
    """
    Mixture of two Gaussian distributions with different covariance, parameterized by same mean theta.
    Mixing param is assumed to be that of first component.
    When dim(theta) > 1, diagonal covariance matrix (again, possibly different across components).
    """

    def __init__(self, 
                 param_dims: int, observed_dims: int, sample_size: int,
                 param_grid_bounds: tuple, param_grid_size: int, 
                 likelihood_kwargs: dict,
                 prior: Union[str, torch.distributions.Distribution], prior_kwargs: Union[dict, None]):
        
        assert all([key in likelihood_kwargs for key in ['mixing_param', 'sigma']])
        
        self.d = param_dims
        self.observed_d = observed_dims
        self.sample_size = sample_size
        
        if self.d == 1:
            self.param_grid = torch.linspace(param_grid_bounds[0], param_grid_bounds[1], steps=param_grid_size)
            self.param_grid_size = param_grid_size
        elif self.d == 2:
            param_grid_1d = np.linspace(param_grid_bounds[0], param_grid_bounds[1], num=param_grid_size)
            # 2-dimensional grid of (grid_sample_size X grid_sample_size) points
            self.param_grid = torch.from_numpy(np.transpose([np.tile(param_grid_1d, len(param_grid_1d)),
                                                             np.repeat(param_grid_1d, len(param_grid_1d))]))
            self.grid_sample_size = param_grid_size**2
        else:
            raise NotImplementedError

        self.likelihood_kwargs = likelihood_kwargs
        self.prior_kwargs = prior_kwargs
        if prior == 'uniform':
            if self.d == 1:
                self.prior = torch.distributions.Uniform(**prior_kwargs)
            else:
                self.prior = MultipleIndependent(
                    dists=[torch.distributions.Uniform(low=torch.Tensor([prior_kwargs['low']]),
                                                       high=torch.Tensor([prior_kwargs['high']]))]*self.d
                )
        elif isinstance(prior, torch.distributions.Distribution):
            self.prior = prior
        else:
            raise NotImplementedError

    def __call__(self, parameters):
        return self.generate_sample(parameters).reshape(-1, self.d)

    def _likelihood_sample(self, mean: torch.Tensor, n_simulations: int):
        # same for each simulation (of size self.sample_size) in this setting
        which_mean = mean.repeat_interleave(self.sample_size, dim=0).float()

        # choose component covariance structure for each simulation (of size self.sample_size)
        which_sigma = torch.from_numpy(
            np.random.choice(a=self.likelihood_kwargs['sigma'],
                             p=(self.likelihood_kwargs['mixing_param'], 1-self.likelihood_kwargs['mixing_param']),
                             size=n_simulations*self.sample_size, replace=True)
        ).float()

        # simulate n_simulations*self.sample_size samples and then concatenate and reshape
        samples = [MultivariateNormal(loc=which_mean[i, :],
                                      covariance_matrix=(which_sigma[i]**2) * torch.eye(self.d)).sample().reshape(1, self.d)
                   for i in range(n_simulations*self.sample_size)]
        return torch.cat(samples, dim=0).reshape(n_simulations, self.sample_size, self.d)

    def generate_sample(self,
                        parameters: Union[torch.Tensor, None],
                        n_simulations: Union[int, None] = None,
                        return_parameters=False):
        assert not all((parameters is None, n_simulations is None))
        if parameters is None:
            parameters = self.prior.sample(sample_shape=(n_simulations, )).reshape(n_simulations, self.d)
        else:
            parameters = parameters.reshape(-1, self.d)
            n_simulations = parameters.shape[0]

        samples = self._likelihood_sample(mean=parameters, n_simulations=n_simulations)
        if return_parameters:
            return parameters, samples
        else:
            return samples


class GMMSimulatorSeparation:
    """
    Mixture of two Gaussian distributions with different covariance, parameterized by (-theta, theta)
    Mixing param is assumed to be that of first component.
    When dim(theta) > 1, diagonal covariance matrix (again, possibly different across components).
    """

    def __init__(self,
                 param_dims: int, observed_dims: int, sample_size: int,
                 param_grid_bounds: tuple, param_grid_size: int,
                 likelihood_kwargs: dict,
                 prior: Union[str, torch.distributions.Distribution], prior_kwargs: Union[dict, None]):

        assert all([key in likelihood_kwargs for key in ['mixing_param', 'sigma']])

        self.d = param_dims
        self.observed_d = observed_dims
        self.sample_size = sample_size

        if self.d == 1:
            self.param_grid = torch.linspace(param_grid_bounds[0], param_grid_bounds[1], steps=param_grid_size)
            self.param_grid_size = param_grid_size
        elif self.d == 2:
            param_grid_1d = np.linspace(param_grid_bounds[0], param_grid_bounds[1], num=param_grid_size)
            # 2-dimensional grid of (grid_sample_size X grid_sample_size) points
            self.param_grid = torch.from_numpy(np.transpose([np.tile(param_grid_1d, len(param_grid_1d)),
                                                             np.repeat(param_grid_1d, len(param_grid_1d))]))
            self.grid_sample_size = param_grid_size ** 2
        else:
            raise NotImplementedError

        self.likelihood_kwargs = likelihood_kwargs
        self.prior_kwargs = prior_kwargs
        if prior == 'uniform':
            if self.d == 1:
                self.prior = torch.distributions.Uniform(**prior_kwargs)
            else:
                self.prior = MultipleIndependent(
                    dists=[torch.distributions.Uniform(low=torch.Tensor([prior_kwargs['low']]),
                                                       high=torch.Tensor([prior_kwargs['high']]))] * self.d
                )
        elif isinstance(prior, torch.distributions.Distribution):
            self.prior = prior
        else:
            raise NotImplementedError

    def __call__(self, parameters):
        return self.generate_sample(parameters).reshape(-1, self.d)

    def _likelihood_sample(self, mean: torch.Tensor, n_simulations: int):
        # choose component for each sample
        which_component = np.random.choice(a=[0, 1],
                                           p=(self.likelihood_kwargs['mixing_param'],
                                              1 - self.likelihood_kwargs['mixing_param']),
                                           size=n_simulations * self.sample_size, replace=True)

        # same for each simulation (of size self.sample_size) in this setting
        which_mean = mean.repeat_interleave(self.sample_size, dim=0).float()
        which_mean[which_component == 0, :] = -1*which_mean[which_component == 0, :]

        # simulate n_simulations*self.sample_size samples and then concatenate and reshape
        samples = [MultivariateNormal(loc=which_mean[i, :],
                                      covariance_matrix=(self.likelihood_kwargs['sigma'][which_component[i]] ** 2) * torch.eye(self.d)
                                      ).sample().reshape(1, self.d)
                   for i in range(n_simulations * self.sample_size)]
        return torch.cat(samples, dim=0).reshape(n_simulations, self.sample_size, self.d)

    def generate_sample(self,
                        parameters: Union[torch.Tensor, None],
                        n_simulations: Union[int, None] = None,
                        return_parameters=False):
        assert not all((parameters is None, n_simulations is None))
        if parameters is None:
            parameters = self.prior.sample(sample_shape=(n_simulations,)).reshape(n_simulations, self.d)
        else:
            parameters = parameters.reshape(-1, self.d)
            n_simulations = parameters.shape[0]

        samples = self._likelihood_sample(mean=parameters, n_simulations=n_simulations)
        if return_parameters:
            return parameters, samples
        else:
            return samples