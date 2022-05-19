import numpy as np
from math import ceil, sqrt, pi
from scipy.integrate import quad
import sys

sys.path.append('..')

from scipy.stats import multivariate_normal, uniform, norm, gamma
from scipy.stats import binom
from scipy.optimize import Bounds
from scipy.special import erf


class ToyMVG:
    
    def __init__(self, 
                 observed_dims=2, 
                 true_param=0.0, true_std=1.0, 
                 param_grid_width=10, grid_sample_size=1000,
                 prior_type='uniform', normal_mean_prior=0.0, normal_std_prior=2.0,
                 empirical_marginal=True):
        
        self.d = observed_dims
        self.observed_dims = observed_dims
        
        # without loss of generality, we only consider points with equal coordinates, for simplicity
        self.true_param = np.repeat(true_param, self.d)
        self.param_grid_width = param_grid_width
        self.low_int = true_param - (param_grid_width/2)
        self.high_int = true_param + (param_grid_width/2)
        assert (self.high_int - self.low_int) == param_grid_width
        
        self.true_cov = true_std * np.eye(observed_dims)

        self.prior_type = prior_type
        if prior_type == 'uniform':
            self.prior_distribution = uniform(loc=self.low_int, scale=(self.high_int - self.low_int))
        elif prior_type == 'normal':
            self.prior_distribution = norm(loc=normal_mean_prior, scale=normal_std_prior)
        else:
            raise ValueError('The variable prior_type needs to be either uniform or normal.'
                             ' Currently %s' % prior_type)
            
        if self.d == 1:
            self.param_grid = np.linspace(self.low_int, self.high_int, num=grid_sample_size)
            self.t0_grid_granularity = grid_sample_size
        elif self.d == 2: 
            a = np.linspace(self.low_int, self.high_int, num=grid_sample_size)
            # 2-dimensional grid of (grid_sample_size X grid_sample_size) points
            self.param_grid = np.transpose([np.tile(a, len(a)), np.repeat(a, len(a))])
            self.t0_grid_granularity = grid_sample_size**2
        else:
            # easier to sample from a d-dimensional uniform for d > 2
            self.param_grid = np.random.uniform(low=self.low_int, high=self.high_int, 
                                                size=grid_sample_size * self.d).reshape(-1, self.d)
            self.t0_grid_granularity = grid_sample_size
            
        #if self.true_param not in self.param_grid:
        #    self.param_grid = np.append(self.param_grid.reshape(-1, self.d),
        #                                self.true_param.reshape(-1,self.d), axis=0)
        #    self.t0_grid_granularity += 1
        
        if not empirical_marginal:
            raise NotImplementedError("We are only using empirical marginal for now.")
        self.empirical_marginal = True
    
    def sample_sim(self, sample_size, true_param):
        return multivariate_normal(mean=true_param,
                                   cov=self.true_cov).rvs(sample_size).reshape(sample_size, self.observed_dims)
    
    def sample_sim_gamma(self, sample_size, true_param):
        return gamma.rvs(true_param, size=sample_size).reshape(sample_size, self.observed_dims)

    def sample_param_values(self, sample_size):
        unique_theta = self.prior_distribution.rvs(size=sample_size * self.d)
        return unique_theta.reshape(sample_size, self.d)
    
    def sample_param_values_qr(self, sample_size):
        # enlarge support to make sure observed sample is not on the boundaries
        if self.prior_type == "uniform":
            qr_prior_distribution = uniform(loc=self.true_param[0] - self.param_grid_width,
                                            scale=(2*self.param_grid_width))
        else:
            raise NotImplementedError
        unique_theta = qr_prior_distribution.rvs(size=sample_size * self.d)
        return unique_theta.reshape(sample_size, self.d)
    
    def sample_empirical_marginal(self, sample_size):
        theta_vec_marg = self.sample_param_values(sample_size=sample_size)
        return np.apply_along_axis(arr=theta_vec_marg.reshape(-1, self.d), axis=1,
                                   func1d=lambda row: self.sample_sim(
                                   sample_size=1, true_param=row)).reshape(-1, self.observed_dims)
    
    def generate_sample(self, sample_size, p=0.5, gamma=False, **kwargs):
        theta_vec = self.sample_param_values(sample_size=sample_size)
        bern_vec = np.random.binomial(n=1, p=p, size=sample_size)
        concat_mat = np.hstack((
            bern_vec.reshape(-1, 1),
            theta_vec.reshape(-1, self.d)
        ))
        
        if gamma:
            sample = np.apply_along_axis(arr=concat_mat, axis=1,
                                         func1d=lambda row: self.sample_sim_gamma(sample_size=1, 
                                                                                  true_param=row[1:1+self.d]) 
                                         if row[0] else self.sample_empirical_marginal(sample_size=1))
        else:
            sample = np.apply_along_axis(arr=concat_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(sample_size=1, 
                                                                            true_param=row[1:1+self.d]) 
                                         if row[0] else self.sample_empirical_marginal(sample_size=1))
        
        return np.hstack((concat_mat, sample.reshape(sample_size, self.observed_dims)))

    def sample_msnh(self, b_prime, obs_sample_size, gamma=False):
        theta_mat = self.sample_param_values(sample_size=b_prime).reshape(-1, self.d)
        assert theta_mat.shape == (b_prime, self.d)
        
        if gamma:
            sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                             func1d=lambda row: self.sample_sim_gamma(sample_size=obs_sample_size, 
                                                                                      true_param=row[:self.d]))
        else:            
            sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                             func1d=lambda row: self.sample_sim(sample_size=obs_sample_size, 
                                                                                true_param=row[:self.d]))
        return theta_mat, sample_mat.reshape(b_prime, obs_sample_size, self.observed_dims)
    
    def generate_observed_sample(self, n_samples, obs_sample_size, theta_mat=None, gamma=False):
        if theta_mat is None:
            theta_mat = self.sample_param_values(sample_size=n_samples).reshape(-1, self.d)
        assert theta_mat.shape == (n_samples, self.d)
        
        if gamma:
            sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                             func1d=lambda row: self.sample_sim_gamma(sample_size=obs_sample_size, 
                                                                                      true_param=row[:self.d]))
        else:
            sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                             func1d=lambda row: self.sample_sim(sample_size=obs_sample_size, 
                                                                                true_param=row[:self.d]))
        return theta_mat, sample_mat.reshape(n_samples, obs_sample_size, self.observed_dims)
    
    @staticmethod
    def compute_mle(x_obs):
        return np.mean(x_obs, axis=0)
    
    @staticmethod
    def clopper_pearson_interval(n, x, alpha):
        theta_grid = np.linspace(0, 1, 1000)
        left = np.min([p for p in theta_grid if binom.cdf(x, n, p) > alpha/2])
        right = np.max([p for p in theta_grid if (1-binom.cdf(x, n, p)) > alpha/2])
        return left, right
    
    def _compute_exact_lr_simplevcomp(self, t0, mle, obs_sample_size):
        return (-1)*(obs_sample_size/2)*(np.linalg.norm((mle - t0).reshape(-1, self.d), ord=2, axis=1)**2)
    
    def _compute_multivariate_normal_pdf(self, x, mu, cov=None, obs_sample_size=1):
        if cov is None:
            cov=np.eye(self.d)/obs_sample_size
        return multivariate_normal(mean=mu, cov=cov).pdf(x)
    
    def _compute_marginal_pdf(self, x_obs):
        '''
        In this calculation we are assuming that the covariance matrix is diagonal with all entries being equal, so
        we only consider the first element for every point.
        '''
        density = 0.5 * (erf((self.high_int - x_obs) / (np.sqrt(2) * self.true_cov[0, 0])) -
                         erf((self.low_int - x_obs) / (np.sqrt(2) * self.true_cov[0, 0])))
        return np.prod(density)
    
    def compute_exact_odds(self, theta_vec, x_vec, p=0.5):
        x_vec = x_vec.reshape(-1, self.observed_dims)
        theta_vec = theta_vec.reshape(-1, self.d)

        f_val = np.array([self._compute_multivariate_normal_pdf(
            x=x, mu=theta_vec[ii, :]) for ii, x in enumerate(x_vec)]).reshape(-1, )

        if self.empirical_marginal:
            g_val = np.array([self._compute_marginal_pdf(x_obs=x_obs) for x_obs in x_vec]).reshape(-1, )
        else:
            g_val = self.g_distribution.pdf(x=x_vec).reshape(-1, )
        return (f_val * p) / (g_val * (1 - p))
    
    def _compute_marginal_bf_denominator(self, x_obs, mu, prior_type='uniform', obs_sample_size=1):
        '''
        In this calculation we are assuming that the covariance matrix is the Identity matrix.
        '''
        if prior_type == 'uniform':
            unif_distr = (1 / ((mu + (self.param_grid_width/2)) - (mu - (self.param_grid_width/2)))) ** self.observed_dims
            density = 0.5 * (erf(((mu + (self.param_grid_width/2)) - x_obs)/np.sqrt(2*obs_sample_size)) - erf(((mu - (self.param_grid_width/2)) - x_obs)/np.sqrt(2*obs_sample_size)))
            assert len(density) == self.observed_dims
            denominator = unif_distr * np.prod(density)
        else:
            raise ValueError("The prior type needs to be 'uniform'. Currently %s" % self.prior_type)
        return denominator
    
    def _compute_marginal_bf_denominator_old(self, x_obs, prior_type='uniform'):
        '''
        In this calculation we are assuming that the covariance matrix is diagonal with all entries being equal, so
        we only consider the first element for every point.
        '''
        if prior_type == 'uniform':
            unif_distr = (1 / (self.high_int - self.low_int)) ** self.observed_dims
            density = 0.5 * unif_distr * (erf((self.high_int - x_obs) / (np.sqrt(2) * self.true_cov[0, 0])) -
                             erf((self.low_int - x_obs) / (np.sqrt(2) * self.true_cov[0, 0])))
        else:
            raise ValueError("The prior type needs to be 'uniform'. Currently %s" % self.prior_type)
        return np.prod(density)

    def compute_exact_bayes_factor_with_marginal(self, x, mu, cov=None, obs_sample_size=1):
        '''
        In this calculation we are assuming that the covariance matrix is the Identity matrix.
        '''
        if self.prior_type == 'uniform':
            x_vec = x.reshape(-1, self.observed_dims)
            theta_vec = mu.reshape(-1, self.d)
            
            f_val = np.array([self._compute_multivariate_normal_pdf(
                x=x_val, mu=theta_vec[ii, :], cov=cov, obs_sample_size=obs_sample_size) for ii, x_val in enumerate(x_vec)]).reshape(-1, )
            g_val = np.array([self._compute_marginal_bf_denominator(x_val, mu, prior_type='uniform', obs_sample_size=obs_sample_size) for x_val in x_vec]
                             ).reshape(-1, )
        else:
            raise ValueError("The prior type needs to be 'uniform'. Currently %s" % self.prior_type)
        return f_val / g_val   
    
    def compute_exact_bayes_factor_with_marginal_old(self, x, mu, cov=None, obs_sample_size=1):
        '''
        In this calculation we are assuming that the covariance matrix is the Identity matrix.
        '''
        if self.prior_type == 'uniform':
            x_vec = x.reshape(-1, self.observed_dims)
            theta_vec = mu.reshape(-1, self.d)
            
            f_val = np.array([self._compute_multivariate_normal_pdf(
                x=x_val, mu=theta_vec[ii, :], cov=cov, obs_sample_size=obs_sample_size) for ii, x_val in enumerate(x_vec)]).reshape(-1, )
            g_val = np.array([self._compute_marginal_bf_denominator_old(x_obs=x_val, prior_type='uniform') for x_val in x_vec]
                             ).reshape(-1, )
        else:
            raise ValueError("The prior type needs to be 'uniform'. Currently %s" % self.prior_type)
        return f_val / g_val  

    
    
def manually_compute_normal_pdf(x, mean, cov, d):
    
    x = x.reshape(1, d)
    mean = mean.reshape(1, d)
    cov = cov.reshape(d, d)
    
    normalization_const = ((2*np.pi)**(-d/2))*(np.linalg.det(cov)**(-1/2))
    return (normalization_const * np.exp((-1/2)*((x-mean) @ np.linalg.inv(cov) @ (x-mean).T))).item()    
