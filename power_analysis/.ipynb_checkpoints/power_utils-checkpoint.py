from functools import partial
import numpy as np
from scipy.stats import chi2, gamma, uniform, pareto, t, norm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style('whitegrid')
from tqdm import tqdm
import os

import jax.numpy as jax_np
from jax import jit, hessian
from jax.scipy.stats import t as t_jax
from jax.scipy.stats import pareto as pareto_jax
from jax.scipy.stats import norm as norm_jax


def exact_lrt_intervals(model,
                        observed_x, 
                        confidence_level):

    lrt_confidence_sets = []
    for obs_x in tqdm(observed_x):
        # KNOWN STATISTICS
        mle = model.compute_mle(obs_x)
        observed_lrt_statistics = (-2)*model._compute_exact_lr_simplevcomp(t0=model.param_grid.reshape(-1, 1),
                                                                           mle=mle.reshape(-1, 1),
                                                                           obs_sample_size=1)
        # TRUE CRITICAL VALUE
        chi_squared = chi2(df=model.d)
        lrt_alpha_quantiles = np.repeat(chi_squared.ppf(q=confidence_level), len(model.param_grid))
        # EXACT LRT CONFIDENCE SET
        lrt_confidence_set = model.param_grid.reshape(-1,)[observed_lrt_statistics.reshape(-1,) <=
                                                           lrt_alpha_quantiles.reshape(-1,)]
        lrt_confidence_sets.append(lrt_confidence_set)

    return lrt_confidence_sets


def critical_values_mc(simulator,
                       param_grid, param_grid_size,
                       mc_n_draws,
                       confidence_level,
                       wald=True,
                       exact_waldo=True,
                       wald_restricted_size=1000):  # otherwise too much memory required to compute Fisher Information
    
    exact_waldo_cutoffs = np.empty(shape=(param_grid_size, ))
    exact_waldo_novar_cutoffs = np.empty(shape=(param_grid_size, ))
    wald_cutoffs = np.empty(shape=(param_grid_size, ))
    wald_novar_cutoffs = np.empty(shape=(param_grid_size, ))
    for idx, param in tqdm(enumerate(param_grid), desc=f"Computing cutoffs for {param_grid_size} tests via Monte Carlo"):
        sample = simulator.likelihood(param).rvs(size=(mc_n_draws, simulator.observed_sample_size))
        
        if exact_waldo:
            # EXACT WALDO
            exact_waldo_stat = simulator.compute_exact_waldo(sample=sample, parameter=np.array([param])).reshape(-1, )
            exact_waldo_cutoffs[idx] = np.quantile(exact_waldo_stat, q=confidence_level)
        
            # EXACT WALDO NOVAR
            exact_waldo_novar = simulator.compute_exact_waldo_novar(sample=sample, parameter=np.array([param])).reshape(-1, )
            exact_waldo_novar_cutoffs[idx] = np.quantile(exact_waldo_novar, q=confidence_level)
        
        if wald:
            # WALD
            wald_stats = simulator.compute_wald(sample=sample[:wald_restricted_size, :], parameter=np.array([param])).reshape(-1, )
            wald_cutoffs[idx] = np.quantile(wald_stats, q=confidence_level)

            # WALD NOVAR
            wald_novar_stats = simulator.compute_wald_novar(sample=sample, parameter=np.array([param])).reshape(-1, )
            wald_novar_cutoffs[idx] = np.quantile(wald_novar_stats, q=confidence_level)
        
    return exact_waldo_cutoffs, exact_waldo_novar_cutoffs, wald_cutoffs, wald_novar_cutoffs

def power_neyman(n_draws,
                 simulator,
                 true_param, 
                 cond_mean_predictor, cond_var_predictor, algo_name, 
                 waldo_cutoffs, waldo_cutoffs_novar, waldo_cutoffs_exact_var,
                 confidence_level, 
                 mc_n_draws,
                 saved_cutoffs=None, waldo=True, exact_waldo=False, wald=False,
                 run_name='', save_path=None):
    
    param_grid = np.concatenate((simulator.param_grid, np.array([true_param])))
    param_grid_size = simulator.param_grid_size + 1
    
    # compute critical values via Monte-Carlo
    if (saved_cutoffs is None) and (exact_waldo or wald):
        exact_waldo_cutoffs, exact_waldo_novar_cutoffs, wald_cutoffs, wald_novar_cutoffs = \
            critical_values_mc(simulator=simulator, param_grid=param_grid, param_grid_size=param_grid_size, mc_n_draws=mc_n_draws, confidence_level=confidence_level, 
                               wald=wald, exact_waldo=exact_waldo, wald_restricted_size=1000)
    else:
        exact_waldo_cutoffs, exact_waldo_novar_cutoffs, wald_cutoffs, wald_novar_cutoffs = saved_cutoffs
    
    exact_waldo_power = np.zeros(shape=(param_grid_size, ))
    exact_waldo_novar_power = np.zeros(shape=(param_grid_size, ))
    wald_power = np.zeros(shape=(param_grid_size, ))
    wald_novar_power = np.zeros(shape=(param_grid_size, ))
    observations = simulator.likelihood(true_param).rvs(size=(n_draws, simulator.observed_sample_size))
    empty_sets = 0  # just to check things are ok
    # TODO: is the loop needed? Probably not (everything vectorized), check implementation of stats methods in simulator
    wald_coverage = 0
    wald_novar_coverage = 0
    exact_waldo_coverage = 0
    exact_waldo_novar_coverage = 0
    for obs in tqdm(observations, desc="Computing power for each observation"):
        
        if exact_waldo:
            # EXACT WALDO
            exact_waldo_stats = simulator.compute_exact_waldo(sample=obs, parameter=param_grid).reshape(-1, )  # remove reshape if remove loop
            exact_waldo_conf_set = param_grid[exact_waldo_stats <= exact_waldo_cutoffs]
            if len(exact_waldo_conf_set) > 0:
                exact_waldo_power += (param_grid < np.min(exact_waldo_conf_set)) | (param_grid > np.max(exact_waldo_conf_set))
            else:
                empty_sets += 1
                exact_waldo_power += np.ones(shape=(param_grid_size, ))  # if conf set is empty than reject all
            exact_waldo_coverage += (exact_waldo_stats[-1] <= exact_waldo_cutoffs[-1])

            # EXACT WALDO NOVAR
            exact_waldo_novar_stats = simulator.compute_exact_waldo_novar(sample=obs, parameter=param_grid).reshape(-1, )
            exact_waldo_novar_conf_set = param_grid[exact_waldo_novar_stats <= exact_waldo_novar_cutoffs]
            if len(exact_waldo_novar_conf_set) > 0:
                exact_waldo_novar_power += (param_grid < np.min(exact_waldo_novar_conf_set)) | (param_grid > np.max(exact_waldo_novar_conf_set))
            else:
                empty_sets += 1
                exact_waldo_novar_power += np.ones(shape=(param_grid_size, ))
            exact_waldo_novar_coverage += (exact_waldo_novar_stats[-1] <= exact_waldo_novar_cutoffs[-1])
        
        if wald:
            # WALD
            wald_stats = simulator.compute_wald(sample=obs, parameter=param_grid).reshape(-1 )
            wald_cutoffs[np.isnan(wald_cutoffs)] = 0
            wald_conf_set = param_grid[wald_stats <= wald_cutoffs]
            if len(wald_conf_set) > 0:
                wald_power += (param_grid < np.min(wald_conf_set)) | (param_grid > np.max(wald_conf_set))
            else:
                empty_sets += 1
                wald_power += np.ones(shape=(param_grid_size, ))
            wald_coverage += (wald_stats[-1] <= wald_cutoffs[-1])

            # WALD NOVAR
            wald_novar_stats = simulator.compute_wald_novar(sample=obs, parameter=param_grid).reshape(-1 )
            wald_novar_cutoffs[np.isnan(wald_cutoffs)] = 0
            wald_novar_conf_set = param_grid[wald_novar_stats <= wald_novar_cutoffs]
            if len(wald_novar_conf_set) > 0:
                wald_novar_power += (param_grid < np.min(wald_novar_conf_set)) | (param_grid > np.max(wald_novar_conf_set))
            else:
                empty_sets += 1
                wald_novar_power += np.ones(shape=(param_grid_size, ))
            wald_novar_coverage += (wald_novar_stats[-1] <= wald_novar_cutoffs[-1])
    
    # divide by n_draws to obtain proportion, i.e. power/coverage
    if exact_waldo:
        exact_waldo_power /= n_draws
        exact_waldo_novar_power /= n_draws
        exact_waldo_coverage /= n_draws
        exact_waldo_novar_coverage /= n_draws
    if wald:
        wald_power /= n_draws
        wald_novar_power /= n_draws
        wald_coverage /= n_draws
        wald_novar_coverage /= n_draws
        wald_coverage = round(wald_coverage, 2)
        
    
    if waldo:
        print("Computing power for estimated Waldo and Waldo-NOVAR...", flush=True)
        # WALDO
        if simulator.sample_method == 'mle':
            mles = simulator.compute_mle(sample=observations)
            if algo_name == 'kernel_reg':
                obs_cond_means, _ = cond_mean_predictor.fit(data_predict=mles.reshape(-1, simulator.observed_d))
                obs_cond_vars, _ = cond_var_predictor.fit(data_predict=mles.reshape(-1, simulator.observed_d))
                obs_cond_vars[obs_cond_vars <= 0] = 1e-6
            elif algo_name == 'kernelregmean':
                obs_cond_means, _ = cond_mean_predictor.fit(data_predict=mles.reshape(-1, simulator.observed_d))
                obs_cond_vars = cond_var_predictor.predict(X=mles.reshape(-1, 1))
                obs_cond_vars[obs_cond_vars <= 0] = 1e-6
            elif algo_name == 'kernelregvar':
                obs_cond_means = cond_mean_predictor.predict(X=mles.reshape(-1, 1))
                obs_cond_vars, _ = cond_var_predictor.fit(data_predict=mles.reshape(-1, simulator.observed_d))
                obs_cond_vars[obs_cond_vars <= 0] = 1e-6
            else:
                obs_cond_means = cond_mean_predictor.predict(X=mles.reshape(-1, 1))
                obs_cond_vars = cond_var_predictor.predict(X=mles.reshape(-1, 1)) + 1e-3
                # obs_cond_vars[obs_cond_vars <= 0] = 1e-6
        else: 
            raise NotImplementedError

        if 'lincorr' in algo_name:
            obs_cond_vars += linreg.predict(X=cond_var_predictor.predict(X=mles.reshape(-1, 1)).reshape(-1, 1))

        tile_param_grid = np.tile(param_grid, observations.shape[0]).reshape(observations.shape[0], param_grid_size)
        waldo_stats = (np.subtract(obs_cond_means.reshape(-1,1), tile_param_grid)**2)/obs_cond_vars.reshape(-1, 1)
        waldo_power = (waldo_stats > waldo_cutoffs.reshape(1, -1)).mean(axis=0)
        waldo_coverage = (waldo_stats[:, -1] <= waldo_cutoffs[-1]).mean()  # last param is true param, see param_grid

        # WALDO EXACT VAR
        #obs_exact_cond_var = simulator.exact_cond_var(sample=observations, theta_support=None, posterior_samples=50_000)
        #waldo_stats_exact_var = (np.subtract(obs_cond_means.reshape(-1,1), tile_param_grid)**2)/obs_exact_cond_var.reshape(-1,1)
        #waldo_power_exact_var = (waldo_stats_exact_var > waldo_cutoffs_exact_var.reshape(1, -1)).mean(axis=0)
        #waldo_coverage_exact_var = (waldo_stats_exact_var[:, -1] <= waldo_cutoffs_exact_var[-1]).mean()  # last param is true param, see param_grid

        # WALDO NOVAR
        waldo_stats_novar = np.subtract(obs_cond_means.reshape(-1,1), tile_param_grid)**2
        waldo_power_novar = (waldo_stats_novar > waldo_cutoffs_novar.reshape(1, -1)).mean(axis=0)
        waldo_coverage_novar = (waldo_stats_novar[:, -1] <= waldo_cutoffs_novar[-1]).mean()  # last param is true param, see param_grid

    # PLOT
    print("Plotting...", flush=True)
    fig, ax = plt.subplots(4, 1, figsize=(15, 30), gridspec_kw={'height_ratios':[2,1,1,1]})
    if exact_waldo:
        ax[0].plot(np.sort(param_grid), exact_waldo_power[np.argsort(param_grid)], linestyle=(0, (5, 1)), c='orange', label='Exact Waldo')
        ax[0].plot(np.sort(param_grid), exact_waldo_novar_power[np.argsort(param_grid)], linestyle='-', c='darkgreen', label='Exact Waldo - NOVAR')
    if wald:
        ax[0].plot(np.sort(param_grid), wald_power[np.argsort(param_grid)], linestyle=(0, (1, 1)), c='blue', label='Wald')
        # ax[0].plot(np.sort(param_grid), wald_novar_power[np.argsort(param_grid)], linestyle=':', c='deepskyblue', label='Wald - NOVAR')
    if waldo:
        ax[0].plot(np.sort(param_grid), waldo_power[np.argsort(param_grid)], linestyle='-', linewidth=5, c='crimson', label='Waldo')
        ax[0].plot(np.sort(param_grid), waldo_power_novar[np.argsort(param_grid)], linestyle='--', linewidth=5, c='black', label='Waldo-novar')
        #ax[0].plot(np.sort(param_grid), waldo_power_exact_var[np.argsort(param_grid)], linestyle=(0, (3, 5, 1, 5, 1, 5)), c='pink', label='Estimated Waldo - Exact Var')
    ax[0].hlines(y=1-confidence_level, xmin=np.min(param_grid), xmax=np.max(param_grid), colors='dimgray', linestyles=':', linewidth=5, 
              label=r'$\alpha=${}'.format(round(1-confidence_level, 2)))
    max_vline = np.max(waldo_power) if waldo else np.max(wald_power)
    ax[0].vlines(x=true_param, ymin=0, ymax=max_vline, colors='dimgray', linestyles='--', linewidth=5, 
              label=r'$\theta^*=${}'.format(true_param))
    ax[0].set_xlim(left=np.min(param_grid)-1, right=np.max(param_grid)+1)
    # ax[0].set_xlim(left=35, right=45)
    ax[0].set_ylim(bottom=0, top=1)
    ax[0].set_xlabel(r'$\theta$', fontsize=45)
    ax[0].set_ylabel('Power', fontsize=45)
    ax[0].tick_params(axis='both', which='major', labelsize=30, bottom=True, left=True, labelleft=True, labelbottom=True)
    if waldo:
        ax[0].set_title(r'Power Curves', fontsize=45, pad=8)#. Coverage at $\theta*$: Waldo={}, Waldo-novar={}'.format(round(waldo_coverage, 2), round(waldo_coverage_novar, 2)))
    else:
        ax[0].set_title(r'Power as a function of $\theta$. Coverage at $\theta*$: Wald={wld:.2f}, Wald-novar={wldn}, Exact-Waldo={ewldo}, Exact-Waldo-novar={ewldon}'.format(wld=round(wald_coverage, 2),
                                                                                                                                                       wldn=round(wald_novar_coverage, 2),
                                                                                                                                                       ewldo=round(exact_waldo_coverage, 2),
                                                                                                                                                       ewldon=round(exact_waldo_novar_coverage, 2)), 
                        fontsize=10)
    ax[0].legend(prop={'size': 40}, loc=(0.55, 0.1))
    
    if waldo:
        ax[1].plot(np.sort(param_grid), waldo_stats.mean(axis=0)[np.argsort(param_grid)], color='crimson', linestyle='--', linewidth=5, label='Test Statistic')
        ax[1].plot(np.sort(param_grid), waldo_cutoffs[np.argsort(param_grid)], color='crimson', linestyle='-', linewidth=5, label='Critical Values')
        #ax[1].plot(np.sort(param_grid), waldo_stats_exact_var.mean(axis=0)[np.argsort(param_grid)], color='pink', linestyle='--', label='Estimated Waldo-ExactVar Statistics')
        #ax[1].plot(np.sort(param_grid), waldo_cutoffs_exact_var[np.argsort(param_grid)], color='pink', linestyle='-', label='Estimated Waldo-ExactVar Cutoffs')
        ax[1].set_ylim(bottom=0, top=np.mean(waldo_stats.mean(axis=0)))
        #ax[1].set_xlabel(r'$\Theta$', fontsize=30)
        ax[1].set_ylabel('Distributions', fontsize=45)
        ax[1].tick_params(axis='both', which='major', labelsize=30, bottom=True, left=True, labelleft=True, labelbottom=False)
        ax[1].set_title(r'Waldo', fontsize=45)# Test Statistics and Critical Values
        ax[1].legend(prop={'size': 40})

        ax[2].plot(np.sort(param_grid), waldo_stats_novar.mean(axis=0)[np.argsort(param_grid)], color='black', linestyle='--', linewidth=5)#, label='Waldo-novar Test Statistic')
        ax[2].plot(np.sort(param_grid), waldo_cutoffs_novar[np.argsort(param_grid)], color='black', linestyle='-', linewidth=5)#, label='Waldo-novar Critical Values')
        ax[2].set_ylim(bottom=min(np.min(waldo_cutoffs_novar), 0), top=np.mean(waldo_stats_novar.mean(axis=0)))
        ax[2].set_xlabel(r'$\theta$', fontsize=45)
        ax[2].set_ylabel('Distributions', fontsize=45)
        ax[2].tick_params(axis='both', which='major', labelsize=30, bottom=True, left=True, labelleft=True, labelbottom=True)
        ax[2].set_title('Waldo-novar', fontsize=45)
        # ax[2].legend(prop={'size': 40})
    else:
        # TODO: maybe should plot wald and exact waldo
        ax[1].axis('off')
        ax[2].axis('off')
    
    likelihood_foo = np.prod(simulator.likelihood(param_grid.reshape(-1, 1)).pdf(obs), axis=1)  # use last observation from above
    prior_pdf = simulator.prior.pdf(param_grid)
    if simulator.prior_name == 'gamma':
        # assumes gamma-pareto model
        posterior_pdf = gamma(simulator.prior_kwargs['alpha']+simulator.observed_sample_size, 
                              scale=1/(simulator.prior_kwargs['beta'] + np.sum(np.log(obs/simulator.likelihood_kwargs['scale'])))).pdf(param_grid)
    elif simulator.prior_name == 'gaussian':
        # assumes gaussian-gaussian model
        posterior_pdf = norm(loc=simulator.exact_cond_mean(sample=obs), scale=simulator.exact_cond_var()).pdf(param_grid)
    elif simulator.prior_name == 'uniform':
        posterior_pdf = np.multiply(likelihood_foo, prior_pdf)
    else:
        raise NotImplementedError
    ax[3].plot(np.sort(param_grid), (likelihood_foo/np.sum(likelihood_foo))[np.argsort(param_grid)], label='likelihood', color='blue')
    ax[3].plot(np.sort(param_grid), (prior_pdf/np.sum(prior_pdf))[np.argsort(param_grid)], label='prior', color='crimson')
    ax[3].plot(np.sort(param_grid), (posterior_pdf/np.sum(posterior_pdf))[np.argsort(param_grid)], label='posterior', color='green', linestyle='--')
    ax[3].set_xlabel(r'$\theta$')
    ax[3].set_ylabel('Distributions')
    ax[3].set_title(r'Prior - Likelihood - Posterior, given a fixed observation with $\widehat{{\theta}}^{{mle}}=${num}'.format(num=round(simulator.compute_mle(obs).item(), 2)))
    ax[3].legend()
    fig.tight_layout()
    mpl.rcParams['figure.dpi'] = 600
    if save_path is not None:
        fig_path = os.path.join("waldo/results/power_analysis/", simulator.results_path.replace('--true_param--', str(true_param)).replace('--run_name--', run_name).replace('--algo_name--', algo_name))
        plt.savefig(os.path.join(save_path, fig_path), bbox_inches='tight', dpi=600)
    print("Empty sets: ", empty_sets)
    plt.show()

@jit
def t_neg_ll(param, dfs, sample):
    return -1*t_jax.logpdf(x=sample, df=dfs, loc=param).sum(axis=1)

@jit
def pareto_neg_ll(param, scale, sample):
    return -1*pareto_jax.logpdf(x=sample, b=param, scale=scale).sum(axis=1)

@jit
def norm_neg_ll(param, scale, sample):
    return -1*norm_jax.logpdf(x=sample, loc=param, scale=scale).sum(axis=1)

@jit
def norm_neg_ll_precision(param, mu, sample):
    return -1*norm_jax.logpdf(x=sample, loc=mu, scale=jax_np.sqrt(1/param)).sum(axis=1)

def fisher_info(neg_ll, sample, param, n_draws, **kwargs):
    return hessian(partial(neg_ll, sample=sample, **kwargs))(param).diagonal().reshape(n_draws, n_draws).diagonal()


class ParetoShape:
    
    def __init__(self, 
                 param_grid_bounds, param_grid_size,
                 likelihood_kwargs, prior, prior_kwargs,
                 observed_sample_size,
                 sample_method,  # mle or unfold
                 param_dims=1,
                 observed_dims=1):
        
        assert observed_sample_size > 1, "Cannot estimate shape parameter with an observed sample of size 1"
        if (param_dims > 1) or (observed_dims > 1): raise NotImplementedError
        
        self.param_grid = np.linspace(param_grid_bounds[0], param_grid_bounds[1], num=param_grid_size)
        self.param_grid_size = param_grid_size
        
        self.observed_sample_size = observed_sample_size
        self.sample_method = sample_method
        
        self.d = param_dims
        self.observed_d = observed_dims
        
        self.likelihood_kwargs = likelihood_kwargs
        self.likelihood = partial(pareto, **likelihood_kwargs)
        self.prior_name = prior
        self.prior_kwargs = prior_kwargs
        if prior == 'gamma': 
            self.prior = gamma(a=prior_kwargs['alpha'], scale=1/prior_kwargs['beta'])
            self.results_path = "pareto_gamma_shape/a{alpha}_b{beta}_xmin{xmin}_k--true_param--/power_conf_set/--run_name--_paretoGamma_n{obs_sample_size}_--algo_name--.pdf".format(alpha=prior_kwargs['alpha'], beta=prior_kwargs['beta'], xmin=likelihood_kwargs['scale'], obs_sample_size=observed_sample_size)
        elif prior == 'uniform':
            self.prior = uniform(**prior_kwargs)
            self.results_path = 'pareto_gamma_shape/uniform_prior/power_conf_set/--run_name--_paretoUniform_n{obs_sample_size}_k--true_param--_--algo_name--.pdf'.format(obs_sample_size=observed_sample_size)
        else: 
            raise ValueError
                
    def generate_sample(self, size, return_raw=False):
        
        params = self.prior.rvs(size=size)
        sample_raw = self.likelihood(params).rvs(size=(self.observed_sample_size, size)).T
        
        if self.observed_sample_size == 1:
            return params, sample_raw
        else:
            if self.sample_method == 'mle':
                sample = self.compute_mle(sample_raw).reshape(-1, 1)
            elif self.sample_method == 'unfold':
                params = np.repeat(params, sample_raw.shape[1])
                sample = sample_raw.reshape(-1, 1)
            if return_raw:
                return params, sample, sample_raw
            else:
                return params, sample
    
    def compute_mle(self, sample):
        return self.observed_sample_size/np.sum(np.log(sample.reshape(-1, self.observed_sample_size) / self.likelihood_kwargs['scale']), axis=1)
    
    def compute_exact_waldo(self, sample, parameter):
        # axis 0 along samples, axis 1 along parameters
        alpha_posterior = self.prior_kwargs['alpha'] + self.observed_sample_size
        beta_posterior = self.prior_kwargs['beta'] + np.sum(np.log(sample.reshape(-1, self.observed_sample_size) / self.likelihood_kwargs['scale']), axis=1)
        return (((alpha_posterior/beta_posterior).reshape(-1, 1) - parameter.reshape(1, -1))**2) / (alpha_posterior/(beta_posterior**2)).reshape(-1, 1) 
        
    def compute_exact_waldo_novar(self, sample, parameter):
        alpha_posterior = self.prior_kwargs['alpha'] + self.observed_sample_size
        beta_posterior = self.prior_kwargs['beta'] + np.sum(np.log(sample.reshape(-1, self.observed_sample_size) / self.likelihood_kwargs['scale']), axis=1)
        return ((alpha_posterior/beta_posterior).reshape(-1, 1) - parameter.reshape(1, -1))**2
    
    def compute_wald(self, sample, parameter):
        mle = self.compute_mle(sample=sample).reshape(-1, 1)
        fisher_information = fisher_info(neg_ll=pareto_neg_ll, sample=sample, param=mle, n_draws=mle.shape[0], scale=self.likelihood_kwargs['scale'])
        return ((mle - parameter.reshape(1, -1))**2) * fisher_information.reshape(-1, 1)
    
    def compute_wald_novar(self, sample, parameter):
        mle = self.compute_mle(sample=sample).reshape(-1, 1)
        return ((mle - parameter.reshape(1, -1))**2)
    
    def exact_cond_mean(self, sample, theta_support=None, posterior_samples=50_000):
        if self.prior_name == 'gamma':
            alpha_posterior = self.prior_kwargs['alpha'] + self.observed_sample_size
            beta_posterior = self.prior_kwargs['beta'] + np.sum(np.log(sample / self.likelihood_kwargs['scale']), axis=1)
            return alpha_posterior/beta_posterior
        elif self.prior_name == 'uniform':
            if theta_support is None:
                theta_support = np.linspace(0.001, 200, 1_000)
            prior_pdf = uniform(loc=self.prior_kwargs['loc'], scale=self.prior_kwargs['scale']).pdf(theta_support).reshape(1, len(theta_support))
            likelihood_foo = np.prod(pareto(theta_support.reshape(1, 1, -1), scale=self.likelihood_kwargs['scale']).pdf(sample.reshape(-1, self.observed_sample_size, 1)), axis=1)
            posterior_pdf = np.multiply(likelihood_foo.reshape(-1, len(theta_support)), prior_pdf)
            posterior_pdf /= np.sum(posterior_pdf, axis=1).reshape(-1, 1)
            posterior_mean = np.array([np.mean(np.random.choice(theta_support, replace=True, size=posterior_samples, p=posterior_pdf[i, :])) for i in tqdm(range(sample.shape[0]))])
            return posterior_mean
        else:
            raise NotImplementedError
    
    def exact_cond_var(self, sample, theta_support=None, posterior_samples=50_000):
        if self.prior_name == 'gamma':
            alpha_posterior = self.prior_kwargs['alpha'] + self.observed_sample_size
            beta_posterior = self.prior_kwargs['beta'] + np.sum(np.log(sample / self.likelihood_kwargs['scale']), axis=1)
            return alpha_posterior/(beta_posterior**2)
        elif self.prior_name == 'uniform':
            if theta_support is None:
                theta_support = np.linspace(0.001, 200, 1_000)
            prior_pdf = uniform(loc=self.prior_kwargs['loc'], scale=self.prior_kwargs['scale']).pdf(theta_support).reshape(1, len(theta_support))
            likelihood_foo = np.prod(pareto(theta_support.reshape(1, 1, -1), scale=self.likelihood_kwargs['scale']).pdf(sample.reshape(-1, self.observed_sample_size, 1)), axis=1)
            posterior_pdf = np.multiply(likelihood_foo.reshape(-1, len(theta_support)), prior_pdf)
            posterior_pdf /= np.sum(posterior_pdf, axis=1).reshape(-1, 1)
            posterior_var = np.array([np.var(np.random.choice(theta_support, replace=True, size=posterior_samples, p=posterior_pdf[i, :])) for i in tqdm(range(sample.shape[0]))])
            return posterior_var
        else:
            raise NotImplementedError
            
# TODO: make base class and use inheritance for each simulator     
class GaussianMean:
    
    def __init__(self, 
                 param_grid_bounds, param_grid_size,
                 likelihood_kwargs, prior, prior_kwargs,
                 observed_sample_size,
                 sample_method,  # mle or unfold
                 param_dims=1,
                 observed_dims=1):
    
        if (param_dims > 1) or (observed_dims > 1): raise NotImplementedError
        
        self.param_grid = np.linspace(param_grid_bounds[0], param_grid_bounds[1], num=param_grid_size)
        self.param_grid_size = param_grid_size
        
        self.observed_sample_size = observed_sample_size
        self.sample_method = sample_method
        
        self.d = param_dims
        self.observed_d = observed_dims
        
        self.likelihood_kwargs = likelihood_kwargs
        self.likelihood = partial(norm, **likelihood_kwargs)
        self.prior_name = prior
        self.prior_kwargs = prior_kwargs
        self.qr_prior = uniform(loc=param_grid_bounds[0], scale=param_grid_bounds[1]-param_grid_bounds[0])
        if prior == 'gaussian': 
            self.prior = norm(loc=prior_kwargs['loc'], scale=prior_kwargs['scale'])
            self.results_path = "gaussian_mean/muprior{muprior}_sigmaprior{sigmaprior}_sigma{sigma}_mu--true_param--/power_conf_set/--run_name--_n{obs_sample_size}_--algo_name--.png".format(muprior=prior_kwargs['loc'], sigmaprior=prior_kwargs['scale'], sigma=likelihood_kwargs['scale'], obs_sample_size=observed_sample_size)
        elif prior == 'uniform':
            self.prior = uniform(**prior_kwargs)
            self.results_path = 'gaussian_mean/uniform_prior/power_conf_set/--run_name--_n{obs_sample_size}_mu--true_param--_--algo_name--.png'.format(obs_sample_size=observed_sample_size)
        else: 
            raise ValueError
                
    def generate_sample(self, size, params=None, return_raw=False):
        
        if params is None:
            params = self.prior.rvs(size=size)
        sample_raw = self.likelihood(params).rvs(size=(self.observed_sample_size, size)).T
        
        if self.observed_sample_size == 1:
            return params, sample_raw
        else:
            if self.sample_method == 'mle':
                sample = self.compute_mle(sample_raw).reshape(-1, 1)
            elif self.sample_method == 'unfold':
                params = np.repeat(params, sample_raw.shape[1])
                sample = sample_raw.reshape(-1, 1)
            if return_raw:
                return params, sample, sample_raw
            else:
                return params, sample
            
    def generate_qr_sample(self, size, return_raw=False):
        
        params = self.qr_prior.rvs(size=size)
        sample_raw = self.likelihood(params).rvs(size=(self.observed_sample_size, size)).T
        
        if self.observed_sample_size == 1:
            return params, sample_raw
        else:
            if self.sample_method == 'mle':
                sample = self.compute_mle(sample_raw).reshape(-1, 1)
            elif self.sample_method == 'unfold':
                params = np.repeat(params, sample_raw.shape[1])
                sample = sample_raw.reshape(-1, 1)
            if return_raw:
                return params, sample, sample_raw
            else:
                return params, sample

    def compute_mle(self, sample):
        return np.mean(sample.reshape(-1, self.observed_sample_size), axis=1)
    
    # TODO: refactor to 'posterior_mean'
    def exact_cond_mean(self, sample, theta_support=None, posterior_samples=50_000):
        mles = self.compute_mle(sample)
        if self.prior_name == 'gaussian':
            return (self.prior_kwargs['loc']*(self.likelihood_kwargs['scale']**2) + (self.prior_kwargs['scale']**2)*self.observed_sample_size*mles) / \
                    ((self.prior_kwargs['scale']**2)*self.observed_sample_size + (self.likelihood_kwargs['scale']**2))
        elif self.prior_name == 'uniform':
            std_normal = norm(loc=0, scale=1)
            a, b = np.min(self.param_grid), np.max(self.param_grid)
            alpha = (a - mles.reshape(-1, 1))/(self.likelihood_kwargs['scale']/np.sqrt(self.observed_sample_size))
            beta = (b - mles.reshape(-1, 1))/(self.likelihood_kwargs['scale']/np.sqrt(self.observed_sample_size))
            Z = std_normal.cdf(x=beta.reshape(-1,)).reshape(-1, 1) - std_normal.cdf(x=alpha.reshape(-1,)).reshape(-1, 1)
            posterior_mean = mles.reshape(-1, 1) + (std_normal.pdf(x=alpha.reshape(-1, )).reshape(-1,1) - std_normal.pdf(x=beta.reshape(-1, )).reshape(-1,1))*(self.likelihood_kwargs['scale']/np.sqrt(self.observed_sample_size))/Z 
            """
            if theta_support is None:
                theta_support = np.linspace(-100, 100, 10_000)
            prior_pdf = uniform(loc=self.prior_kwargs['loc'], scale=self.prior_kwargs['scale']).pdf(theta_support).reshape(1, len(theta_support))
            likelihood_foo = np.prod(norm(theta_support.reshape(1, 1, -1), scale=self.likelihood_kwargs['scale']).pdf(sample.reshape(-1, self.observed_sample_size, 1)), axis=1)
            posterior_pdf = np.multiply(likelihood_foo.reshape(-1, len(theta_support)), prior_pdf)
            posterior_pdf /= np.sum(posterior_pdf, axis=1).reshape(-1, 1)
            posterior_mean = np.array([np.mean(np.random.choice(theta_support, replace=True, size=posterior_samples, p=posterior_pdf[i, :])) for i in tqdm(range(sample.shape[0]))])
            """
            return posterior_mean
        else:
            raise NotImplementedError
    
    # TODO: refactor to 'posterior_var'
    def exact_cond_var(self, sample=None, theta_support=None, posterior_samples=50_000):
        if self.prior_name == 'gaussian':
            numerator = ((self.likelihood_kwargs['scale']**2)*(self.prior_kwargs['scale']**2))
            denominator = ((self.prior_kwargs['scale']**2)*self.observed_sample_size + (self.likelihood_kwargs['scale']**2))
            return np.array([numerator / denominator])
        elif self.prior_name == 'uniform':
            mles = self.compute_mle(sample)
            std_normal = norm(loc=0, scale=1)
            a, b = np.min(self.param_grid), np.max(self.param_grid)
            alpha = (a - mles.reshape(-1, 1))/(self.likelihood_kwargs['scale']/np.sqrt(self.observed_sample_size))
            beta = (b - mles.reshape(-1, 1))/(self.likelihood_kwargs['scale']/np.sqrt(self.observed_sample_size))
            Z = std_normal.cdf(x=beta.reshape(-1,)).reshape(-1, 1) - std_normal.cdf(x=alpha.reshape(-1,)).reshape(-1, 1)
            sigma_squared = np.square(self.likelihood_kwargs['scale'])/self.observed_sample_size
            central_term = (alpha*std_normal.pdf(x=alpha.reshape(-1, )).reshape(-1, 1) - beta*std_normal.pdf(x=beta.reshape(-1, )).reshape(-1, 1))/Z
            right_term = ((std_normal.pdf(x=alpha.reshape(-1, )).reshape(-1, 1) - std_normal.pdf(x=beta.reshape(-1, )).reshape(-1, 1))/Z)**2
            posterior_var = sigma_squared*(1 + central_term - right_term)
            """
            if theta_support is None:
                theta_support = np.linspace(-100, 100, 10_000)
            prior_pdf = uniform(loc=self.prior_kwargs['loc'], scale=self.prior_kwargs['scale']).pdf(theta_support).reshape(1, len(theta_support))
            likelihood_foo = np.prod(norm(theta_support.reshape(1, 1, -1), scale=self.likelihood_kwargs['scale']).pdf(sample.reshape(-1, self.observed_sample_size, 1)), axis=1)
            posterior_pdf = np.multiply(likelihood_foo.reshape(-1, len(theta_support)), prior_pdf)
            posterior_pdf /= np.sum(posterior_pdf, axis=1).reshape(-1, 1)
            posterior_var = np.array([np.var(np.random.choice(theta_support, replace=True, size=posterior_samples, p=posterior_pdf[i, :])) for i in tqdm(range(sample.shape[0]))])
            """
            return posterior_var
        else:
            raise NotImplementedError
    
    def compute_exact_waldo(self, sample, parameter):
        # axis 0 along samples, axis 1 along parameters
        posterior_mean = self.exact_cond_mean(sample)
        posterior_variance = self.exact_cond_var(sample=sample)
        return ((posterior_mean.reshape(-1, 1) - parameter.reshape(1, -1))**2) / posterior_variance.reshape(-1, 1)
        
    def compute_exact_waldo_novar(self, sample, parameter):
        posterior_mean = self.exact_cond_mean(sample)
        return (posterior_mean.reshape(-1, 1) - parameter.reshape(1, -1))**2
    
    def compute_wald(self, sample, parameter):
        mle = self.compute_mle(sample=sample).reshape(-1, 1)
        fisher_information = fisher_info(neg_ll=norm_neg_ll, sample=sample, param=mle, n_draws=mle.shape[0], scale=self.likelihood_kwargs['scale'])
        return ((mle - parameter.reshape(1, -1))**2) * fisher_information.reshape(-1, 1)
    
    def compute_wald_novar(self, sample, parameter):
        mle = self.compute_mle(sample=sample).reshape(-1, 1)
        return ((mle - parameter.reshape(1, -1))**2)
        
    
class TStudentLocation:
    
    def __init__(self, 
                 param_grid_bounds, param_grid_size,
                 t_df, prior, prior_kwargs,
                 observed_sample_size,
                 sample_method,  # mle or unfold
                 param_dims=1,
                 observed_dims=1):
        
        if (param_dims > 1) or (observed_dims > 1): raise NotImplementedError
        
        self.param_grid = np.linspace(param_grid_bounds[0], param_grid_bounds[1], num=param_grid_size)
        self.param_grid_size = param_grid_size
        
        self.t_df = t_df
        self.observed_sample_size = observed_sample_size
        self.sample_method = sample_method
        
        self.d = param_dims
        self.observed_d = observed_dims
        
        if prior == 'gamma': self.prior = gamma(a=prior_kwargs['alpha'], scale=1/prior_kwargs['beta'])
        elif prior == 'uniform': self.prior = uniform(**prior_kwargs)
        else: raise ValueError
        
    def compute_mle(self, sample):
        sample = sample.reshape(-1, self.observed_sample_size)
        return np.mean(sample, axis=1)
        
    def generate_sample(self, size):
        
        params = self.prior.rvs(size=size)
        sample = t(self.t_df, loc=params).rvs(size=(self.observed_sample_size, size)).T
        
        if self.observed_sample_size > 1:
            if self.sample_method == 'mle':
                sample = self.compute_mle(sample).reshape(-1, 1)
            elif self.sample_method == 'unfold':
                params = np.repeat(params, sample.shape[1])
                sample = sample.reshape(-1, 1)
        return params, sample
    
    
class TStudentLocationVaryingDoF:
    
    def __init__(self, 
                 param_grid_bounds, param_grid_size,
                 prior, prior_kwargs,
                 observed_sample_size,
                 sample_method,  # mle or unfold
                 param_dims=1,
                 observed_dims=1):
        
        if (param_dims > 1) or (observed_dims > 1): raise NotImplementedError
        
        self.param_grid = np.linspace(param_grid_bounds[0], param_grid_bounds[1], num=param_grid_size)
        self.param_grid_size = param_grid_size
        
        self.observed_sample_size = observed_sample_size
        self.sample_method = sample_method
        
        self.d = param_dims
        self.observed_d = observed_dims
        
        if prior == 'gamma': self.prior = gamma(a=prior_kwargs['alpha'], scale=1/prior_kwargs['beta'])
        elif prior == 'uniform': self.prior = uniform(**prior_kwargs)
        else: raise ValueError
        
    def compute_mle(self, sample):
        sample = sample.reshape(-1, self.observed_sample_size)
        return np.mean(sample, axis=1)
        
    def generate_sample(self, size):
        
        params = self.prior.rvs(size=size)
        sample = t(params, loc=params).rvs(size=(self.observed_sample_size, size)).T
        
        if self.observed_sample_size > 1:
            if self.sample_method == 'mle':
                sample = self.compute_mle(sample).reshape(-1, 1)
            elif self.sample_method == 'unfold':
                params = np.repeat(params, sample.shape[1])
                sample = sample.reshape(-1, 1)
        return params, sample
