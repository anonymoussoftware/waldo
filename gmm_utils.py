import numpy as np
import pandas as pd
from scipy.stats import norm, uniform
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style('whitegrid')

from tqdm import tqdm
import os


class GMMSimulator:
    
    # assumes two components in the mixture
    
    # mixing param is assumed to be that of component with -theta as mean
    def __init__(self, 
                 param_grid_bounds: tuple, param_grid_size: int, 
                 likelihood_kwargs: dict, prior: str, prior_kwargs: dict,
                 observed_sample_size: int,  # single-sample size
                 param_dims: int, observed_dims: int):
        
        if (param_dims > 1) or (observed_dims > 1): raise NotImplementedError
        # TODO: better to have 'sigma_mixture_neg' and 'sigma_mixture_pos'
        assert all([key in likelihood_kwargs for key in ['mixing_param', 'sigma_mixture']])
        
        self.param_grid = np.linspace(param_grid_bounds[0], param_grid_bounds[1], num=param_grid_size)
        self.param_grid_size = param_grid_size
        
        self.observed_sample_size = observed_sample_size
        self.d = param_dims
        self.observed_d = observed_dims
        
        self.likelihood_kwargs = likelihood_kwargs
        self.prior_kwargs = prior_kwargs
        if prior == 'uniform': 
            self.prior = uniform(**prior_kwargs)
            self.results_path = None  # TODO. Probably independent of prior choice
        else:
            raise NotImplementedError
            
    def likelihood_rvs(self, mean: np.ndarray, sample_size):
        # sample_size can be any number if mean is an array with only one elem. Else, it must be sample_size == len(mean). Weird behaviour otherwise?
        if sample_size != len(mean) > 1:
            raise ValueError
        
        which_cluster = np.random.binomial(n=1, p=self.likelihood_kwargs['mixing_param'], size=sample_size*self.observed_sample_size).reshape(sample_size, self.observed_sample_size)
        cluster_means = np.take_along_axis(np.hstack((  # array of shape (sample_size, observed_sample_size). Each elem is the mean of the selected mixture component for the ith sample/row
            -mean.reshape(-1, 1),
            mean.reshape(-1, 1)
        )), indices=which_cluster, axis=1)
        cluster_sigmas = np.take_along_axis(np.hstack((
            np.repeat(self.likelihood_kwargs['sigma_mixture'][0], mean.shape[0]).reshape(-1, 1),
            np.repeat(self.likelihood_kwargs['sigma_mixture'][1], mean.shape[0]).reshape(-1, 1)
        )), indices=which_cluster, axis=1)
        return norm(loc=cluster_means, scale=cluster_sigmas).rvs(size=(sample_size, self.observed_sample_size))
    
    def likelihood_pdf(self, parameter, sample):
        # parameter and sample should be of broadcastable shapes. -param and param ordered in same way as in rvs
        first_cluster = (1-self.likelihood_kwargs['mixing_param'])*norm(loc=-parameter, scale=self.likelihood_kwargs['sigma_mixture'][0]).pdf(x=sample)
        second_cluster = self.likelihood_kwargs['mixing_param']*norm(loc=parameter, scale=self.likelihood_kwargs['sigma_mixture'][1]).pdf(x=sample)
        return first_cluster + second_cluster
    
    def calibration_sample(self, b_prime, parameters=None):
        if parameters is None:
            parameters = self.prior.rvs(size=b_prime)
        samples = self.likelihood_rvs(mean=parameters.reshape(-1, 1), sample_size=b_prime)
        return parameters, samples
    
    def observed_sample(self, size, parameters=None):
        # observed sample can be generated in the same way as calibration sample, i.e. all samples from simulators
        return self.calibration_sample(b_prime=size, parameters=parameters)
    
    def compute_mle(self, x):
        assert x.shape == (self.observed_sample_size, self.observed_d), 'One sample at a time for consistency with case where n>1'
        gmm_mixture = mixture.GaussianMixture(n_components=2, covariance_type='full')
        if self.observed_sample_size == 1:
            # TODO: is the mle for the two component means just +- the observed sample when n=1?
            # return an array of shape (n_components, param_dims) == (2, 1) for consistency with ouput when n>1.
            gmm_mixture.means_ = np.vstack(( x.reshape(1, -1), -x.reshape(1, 1) ))
        else:
            gmm_mixture.fit(X=x)
        return gmm_mixture
        
    def compute_exact_lr(self, x, param_h0):
        assert any([isinstance(param_h0, float), isinstance(param_h0, int)])
        x = x.reshape(-1, self.observed_sample_size)
        ll_gmm_t0 = np.sum(np.log(self.likelihood_pdf(parameter=param_h0, sample=x)), axis=1)
        if self.observed_sample_size == 1:
            ll_gmm_t1 = np.array([self.likelihood_pdf(parameter=x[i, :], sample=x[i, :]) for i in range(x.shape[0])])
        else:
            ll_gmm_t1 = []
            for i in range(x.shape[0]):
                gmm_mixture_x = self.compute_mle(x=x[i, :].reshape(-1, 1))
                ll_gmm_t1.append( np.sum(gmm_mixture_x.score_samples(X=x[i, :].reshape(-1, 1))) )
            ll_gmm_t1 = np.array(ll_gmm_t1)
        return ll_gmm_t0 - ll_gmm_t1
    
    
def plot_1D_density(x, figsize=(12, 8), dpi=300):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.histplot(x, stat='probability', kde=True)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)

    previous_dpi = mpl.rcParams['figure.dpi']
    mpl.rcParams['figure.dpi'] = dpi
    plt.show()
    mpl.rcParams['figure.dpi'] = previous_dpi

    
def plot_1D_data(x, parameter, figsize=(12, 8), dpi=300):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.scatterplot(x=x, y=parameter)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel(r'$\theta$', fontsize=14)

    previous_dpi = mpl.rcParams['figure.dpi']
    mpl.rcParams['figure.dpi'] = dpi
    plt.show()
    mpl.rcParams['figure.dpi'] = previous_dpi

    
def plot_1D_statistics(parameters, statistics: dict, figsize=(12, 8), dpi=300, scale='linear', save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for method, cutoffs in predicted_cutoffs.items():
        sns.lineplot(x=parameters, y=statistics, label=method)
    ax.set_xlabel(r'$\theta$', fontsize=18)
    ax.set_ylabel('Statistics', fontsize=18, rotation=0)
    ax.set_yscale(scale)

    previous_dpi = mpl.rcParams['figure.dpi']
    mpl.rcParams['figure.dpi'] = dpi
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    mpl.rcParams['figure.dpi'] = previous_dpi    

    
def plot_1D_cutoffs(parameters, predicted_cutoffs: dict, figsize=(12, 8), dpi=300, scale='linear', save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for method, cutoffs in predicted_cutoffs.items():
        if 'LF2I' in method:
            sns.lineplot(x=parameters, y=cutoffs, label=method, zorder=2)
        else:
            sns.lineplot(x=parameters, y=cutoffs, label=method, zorder=1)
    ax.set_xlabel(r'$\theta$', fontsize=20)
    ax.set_ylabel(r'$C_{{\theta}}$', fontsize=20, rotation=0)
    ax.set_title('Critical values as a function of the parameter', fontsize=20)
    ax.set_yscale(scale)
    ax.legend(prop={'size': 14})

    previous_dpi = mpl.rcParams['figure.dpi']
    mpl.rcParams['figure.dpi'] = dpi
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    mpl.rcParams['figure.dpi'] = previous_dpi
    
    
def plot_1D_statistics_cutoffs(parameters, stats_cutoffs: dict, true_param=None, figsize=(12, 8), dpi=300, scale='linear', xlim=None, ylim=None, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = ['deepskyblue']
    for i, (method, (stats, cutoffs)) in enumerate(stats_cutoffs.items()):
        ax.plot(np.sort(parameters), stats[np.argsort(parameters)], label=method + ' Statistics', linestyle='--', color=colors[i])
        ax.plot(np.sort(parameters), cutoffs[np.argsort(parameters)], label=method + ' Cutoffs', linestyle='-', color=colors[i])
    if true_param is not None:
        rounded_param = np.round(true_param, 2)
        ax.axvline(x=true_param, label=f'True parameter {rounded_param}')
    ax.set_xlabel(r'$\theta$', fontsize=18)
    ax.set_ylabel('Statistics and Cutoffs', fontsize=18)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_yscale(scale)
    ax.legend()

    previous_dpi = mpl.rcParams['figure.dpi']
    mpl.rcParams['figure.dpi'] = dpi
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    mpl.rcParams['figure.dpi'] = previous_dpi    
    

def lrt_cutoffs_mc(simulator,
                   param_grid, param_grid_size,
                   mc_n_draws,
                   confidence_level): 
    lrt_cutoffs = np.empty(shape=(param_grid_size, ))
    for idx, param in tqdm(enumerate(param_grid), desc=f"Computing cutoffs for {param_grid_size} tests via Monte Carlo"):
        sample = simulator.likelihood_rvs(mean=np.array([param]), sample_size=mc_n_draws)
        statistics = -2*simulator.compute_exact_lr(x=sample, param_h0=param)
        lrt_cutoffs[idx] = np.quantile(statistics, q=confidence_level)
    return lrt_cutoffs
    
    
def coverage_diagnostics(true_parameters, covered_indicator, confidence_level, method, figsize=(12, 8), dpi=300, color='blue', save_path=None, is_azure=False):
    
    if is_azure:
        source_path = './gam_diagnostics.csv'
    else:
        source_path = './gam_diagnostics.csv'
    
    # estimate coverage
    pd.DataFrame({"w": covered_indicator.reshape(-1,), 
                  "theta": true_parameters.reshape(-1,)}).to_csv(source_path, index=False)
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    base = importr("base")
    if is_azure:
        ro.r(f'''source('./gam_diagnostics.r')''')
    else:
        ro.r(f'''source('./gam_diagnostics.r')''')
    print("fitting GLM")
    is_azure = "yes" if is_azure else 'no'
    predict_dict = ro.globalenv['helper_function'](is_azure=is_azure)
    predict_dict = dict(zip(predict_dict.names, list(predict_dict)))
    probabilities = np.array(predict_dict["predictions"])
    upper = np.maximum(0, np.minimum(1, probabilities + np.array(predict_dict["se"]) * 2))
    lower = np.maximum(0, np.minimum(1, probabilities - np.array(predict_dict["se"]) * 2))
    
    # plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    df_plot = pd.DataFrame({"observed_parameter": true_parameters.reshape(-1,),
                            "probabilities": probabilities,
                            "lower": lower,
                            "upper": upper}).sort_values(by="observed_parameter")
    sns.lineplot(x=df_plot.observed_parameter, y=df_plot.probabilities,
                 ax=ax, color=color, label=f"Estimated coverage")
    sns.lineplot(x=df_plot.observed_parameter, y=df_plot.lower, ax=ax, color=color)
    sns.lineplot(x=df_plot.observed_parameter, y=df_plot.upper, ax=ax, color=color)
    ax.fill_between(x=df_plot.observed_parameter, y1=df_plot.lower, y2=df_plot.upper, alpha=0.2, color=color)
    ax.hlines(xmin=np.min(true_parameters), xmax=np.max(true_parameters), y=confidence_level, color='black', linestyle='--', 
             label=f'Nominal coverage = {int(confidence_level*100)}%', linewidth=1.5)
    
    ax.set_xlabel(r'', fontsize=30) #$\theta$
    ax.set_ylabel(r'', fontsize=30) #Coverage. $n=10$
    ax.tick_params(axis='both', which='major', labelsize=20, bottom=True, left=True, labelleft=False, labelbottom=False)
    ax.set_ylim(bottom=0, top=1)
    ax.set_title(method, fontsize=30)
    ax.legend(loc=(0.37, 0.61), prop={'size': 26})
    
    previous_dpi = mpl.rcParams['figure.dpi']
    mpl.rcParams['figure.dpi'] = dpi
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    mpl.rcParams['figure.dpi'] = previous_dpi
        
    
