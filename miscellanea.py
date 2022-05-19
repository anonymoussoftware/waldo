import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import chi2
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from typing import Union
import seaborn as sns
import torch



def binning(n_bins: int, start, end,
            true_params: np.array, confidence_sets,
            coverage=True, length=True, quantiles=(0.25, 0.5, 0.75), 
            prediction_sets=None):
    
    if prediction_sets is None: prediction_sets = []
    bin_edges = np.linspace(start, end, num=n_bins+1)
    binned_confidence_sets = {method: [] for method in confidence_sets}
    binned_prediction_sets = {method: [] for method in prediction_sets}
    if coverage:
        binned_coverage = {method: np.zeros(shape=(n_bins,)) for method in confidence_sets}
        binned_prediction_coverage = {method: np.zeros(shape=(n_bins,)) for method in prediction_sets}
    else:
        binned_coverage = None
        binned_prediction_coverage = None
    if length:
        binned_length = {method: [] for method in confidence_sets}
        binned_prediction_length = {method: [] for method in prediction_sets}
    else:
        binned_length = None
        binned_prediction_length = None
        
    bin_samples_count = np.zeros(shape=(n_bins,), dtype=int)
    for bin_idx in tqdm(range(len(bin_edges)-1)):
        # find samples with param in current bin
        bin_samples_idxs = np.where(((true_params >= bin_edges[bin_idx]) &
                                     (true_params < bin_edges[bin_idx+1])).reshape(-1,))[0]
        bin_samples_count[bin_idx] = len(bin_samples_idxs)

        # each key is the name of the method by which the corresponding confidence sets have been computed
        for method in confidence_sets:                              # avoid empty confidence sets
            bin_idxs_nonempty = [idx for idx in bin_samples_idxs if len(confidence_sets[method][idx]) > 0]

            if coverage:  # coverage of confidence sets within bin
                # those conf sets with len==0 are not covering; if len==0 np.min/max raises error
                bin_coverage = np.mean([1 if ((true_params[i] >= np.min(confidence_sets[method][i])) &
                                              (true_params[i] <= np.max(confidence_sets[method][i]))) else 0
                                        for i in bin_idxs_nonempty]+[0]*(len(bin_samples_idxs)-len(bin_idxs_nonempty)))
                binned_coverage[method][bin_idx] = bin_coverage
            if length:  # quantiles of length of confidence sets within bin
                bin_length_q = np.quantile([np.max(confidence_sets[method][i])-np.min(confidence_sets[method][i])
                                            for i in bin_idxs_nonempty], q=quantiles)
                binned_length[method].append(bin_length_q)

            # bin upper and lower ends of confidence intervals within each bin
            binned_confidence_sets[method].append([
                np.quantile([np.min(confidence_sets[method][idx]) for idx in bin_idxs_nonempty], q=quantiles),
                np.quantile([np.max(confidence_sets[method][idx]) for idx in bin_idxs_nonempty], q=quantiles),
            ])
        
        # prediction sets
        for method in prediction_sets:           
            if prediction_sets is not None:
                if coverage:
                    bin_pred_coverage = np.mean([1 if (true_params[i] >= np.min(prediction_sets[method][i])) & (true_params[i] <= np.max(prediction_sets[method][i])) else 0
                                                 for i in bin_samples_idxs])
                    binned_prediction_coverage[method][bin_idx] = bin_pred_coverage
                if length:
                    bin_pred_len_q = np.quantile([np.max(prediction_sets[method][i]) - np.min(prediction_sets[method][i]) for i in bin_samples_idxs], q=quantiles)
                    binned_prediction_length[method].append(bin_pred_len_q)
                    
                binned_prediction_sets[method].append([
                np.quantile([np.min(prediction_sets[method][idx]) for idx in bin_samples_idxs], q=quantiles),
                np.quantile([np.max(prediction_sets[method][idx]) for idx in bin_samples_idxs], q=quantiles),
            ])
            
    return binned_confidence_sets, binned_prediction_sets, binned_coverage, binned_prediction_coverage, binned_length, binned_prediction_length, bin_samples_count


def coverage_diagnostics_branch(true_parameters, estimated_statistics=None, predicted_quantiles=None, indicators=None, is_azure=False, d=1, c=2):
    'Either one of (estimated_statistics, predicted_quantiles) or indicators must be provided'
    if estimated_statistics is None:
        w = indicators
    else:
        check_matrix = np.hstack((
            estimated_statistics.reshape(len(true_parameters), 1),
            predicted_quantiles.reshape(len(true_parameters), 1),
            np.zeros(shape=(len(true_parameters), 1))
        ))
        # if in acceptance region
        mask_acceptance_region = (check_matrix[:, 0] <= check_matrix[:, 1])
        # then we are covering!
        check_matrix[mask_acceptance_region, -1] = 1
        w = check_matrix[:, -1].reshape(len(true_parameters), 1)

    #w_pred_interval = np.array([1 if ((model.obs_param[i] >= np.clip((point_estimates[i]-c*np.sqrt(var_estimates[i])), a_min=100, a_max=2000)) & (model.obs_param[i] <= np.clip((point_estimates[i]+c*np.sqrt(var_estimates[i])), a_min=100, a_max=2000))) else 0
    #                            for i in range(len(model.obs_param))])


    # diagnostics = {0: [], 1: []}
    #for i, w in enumerate([w_waldo, w_pred_interval]):
    if d==1:
        if is_azure:
            pd.DataFrame({"w": w.reshape(-1,), 
                          "theta": true_parameters.reshape(-1,)}).to_csv("./gam_diagnostics.csv", index=False)
        else:
            pd.DataFrame({"w": w.reshape(-1,), 
                          "theta": true_parameters.reshape(-1,)}).to_csv("./gam_diagnostics.csv", index=False)
    elif d==2:
        pd.DataFrame({"w": w.reshape(-1,), 
                      "theta0": true_parameters.reshape(-1, 2)[:, 0],
                      "theta1": true_parameters.reshape(-1, 2)[:, 1],}).to_csv("./gam_diagnostics.csv", index=False)
    else:
        raise NotImplementedError
        
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro
    base = importr("base")
    if is_azure:
        is_azure = "yes"
        ro.r('''source('./muons/gam_diagnostics.r')''')
    else:
        is_azure = "no"
        ro.r('''source('./muons/gam_diagnostics.r')''')
    print("fitting GAM")
    predict_dict = ro.globalenv['helper_function'](is_azure, d)
    predict_dict = dict(zip(predict_dict.names, list(predict_dict)))
    probabilities = np.array(predict_dict["predictions"])
    upper = np.maximum(0, np.minimum(1, probabilities + np.array(predict_dict["se"]) * c))
    lower = np.maximum(0, np.minimum(1, probabilities - np.array(predict_dict["se"]) * c))
    # diagnostics[0].extend([probabilities, upper, lower])
    return probabilities, upper, lower


def diagnostics_2D(true_parameters, probabilities, confidence_level, probabilities_cbar=None, upper=None, lower=None, save_fig_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # norm = mpl.colors.Normalize(vmin=min(probabilities), vmax=max(probabilities), clip=True)
    #mapper = cm.ScalarMappable()
    
    scatter = plt.scatter(true_parameters[:, 0], true_parameters[:, 1], c=np.round(probabilities*100, 2), cmap=cm.get_cmap(name='inferno'), 
                          vmin=0, vmax=100, alpha=1)
    cbar = fig.colorbar(scatter, format='%1.2f')
    cbar.set_label('Estimated Coverage', fontsize=45)
    cbar.ax.tick_params(labelsize=30)
    cbar.set_ticks(np.linspace(0, 100, 11))
    cbar.set_ticklabels(np.linspace(0, 100, 11).astype(int))
    plt.xlabel(r"$\theta^{{(1)}}$", fontsize=45)
    plt.ylabel(r"$\theta^{{(2)}}$", fontsize=45, rotation=0, labelpad=40)
    plt.tick_params(axis='both', which='major', labelsize=30, bottom=True, left=True, labelleft=True, labelbottom=True)
    #plt.title(f'Nominal Coverage = {confidence_level*100}%', fontsize=20)
    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    mpl.rcParams['figure.dpi'] = 100
    plt.show()
    
    
def barplot_diagnostics(probabilities, upper, lower, confidence_level, save_fig_path):
    proportion_UC = np.sum(upper < confidence_level) / len(upper)
    proportion_OC = np.sum(lower > confidence_level) / len(lower)
    df_barplot = pd.DataFrame({"args_comb": ["Waldo"]*3,
                      "coverage":  ["Undercoverage", "Correct Coverage", "Overcoverage"],
                      "proportion": [proportion_UC, 1-(proportion_OC+proportion_UC), proportion_OC]})
    # barplot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.barplot(data=df_barplot, x="args_comb", y="proportion", hue="coverage", ci=None, ax=ax)
    ax.tick_params(labelsize=17)
    ax.set_xlabel("Method", fontsize=15)
    ax.set_ylabel("Proportion", fontsize=15)
    ax.set_title("Estimated coverage", fontdict={"fontsize":15})
    plt.legend(fontsize="medium")
    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
    mpl.rcParams['figure.dpi'] = 100
    plt.show()


def plot_binned(true_params: np.array,
                confidence_sets,
                prediction_sets,
                n_bins, start, end, confidence_level,
                quantiles=(0.25, 0.5, 0.75),
                plot_length_quantiles=False,
                coverage_diagnostics=None,
                plot_coverage_bins=False,
                fig_res=None,
                save_fig_path=None,
                return_bin_samples_count=False,
                custom_bin_centers=None, 
                is_azure=False):

    assert len(confidence_sets) <= 4
    if fig_res is None:
        fig_res = {'figsize': (10, 30), 'dpi': 600}
    if 'dpi' in fig_res:
        import matplotlib as mpl
        mpl.rcParams['figure.dpi'] = fig_res['dpi']
        
    
    fig, ax = plt.subplots(3, 1, figsize=fig_res['figsize'])
    methods_offset = (end - start)/200
    
    binned_confidence_sets, binned_prediction_sets, binned_coverage, binned_prediction_coverage, binned_length, binned_prediction_length, bin_samples_count = \
        binning(n_bins=n_bins, start=start, end=end, true_params=true_params, coverage=True,
                length=True, quantiles=quantiles, confidence_sets=confidence_sets, prediction_sets=prediction_sets)
    
    for method in binned_prediction_sets:
        binned_confidence_sets[method] = binned_prediction_sets[method]
        binned_coverage[method] = binned_prediction_coverage[method]
        binned_length[method] = binned_prediction_length[method]
    
    colors = ['mediumblue', 'orange', 'crimson', 'green']
    linestyles = ['-', '-.', '--', ':'] #
    length_markers = ['s', 'P', 'o', 'd']#
    bin_edges = np.linspace(start, end, num=n_bins+1)
    for i, method in enumerate(binned_confidence_sets):
        bin_center = ((bin_edges[1]+bin_edges[0])/2)+(i*methods_offset) if custom_bin_centers is None else custom_bin_centers[0]+(i*methods_offset)
        bin_center_interval_length = (bin_edges[1]+bin_edges[0])/2

        
        if plot_coverage_bins:
            # BINNED COVERAGE
            ax[0].plot(np.sort(true_params),
                       np.concatenate([np.repeat(binned_coverage[method][idx], bin_samples_count[idx]) 
                                       for idx in range(n_bins)]),
                       zorder=1, linestyle="--", color=colors[i], label=f"{method} coverage - bins")
        
        if coverage_diagnostics is not None:
            # DIAGNOSTICS COVERAGE
            if 'prediction' in method.lower():
                proba, upper, lower = coverage_diagnostics_branch(true_params, 
                                                                  indicators=coverage_diagnostics[method]['indicators'],
                                                                  is_azure=is_azure)
            else:
                proba, upper, lower = coverage_diagnostics_branch(true_params, 
                                                                  coverage_diagnostics[method]['statistics'],
                                                                  coverage_diagnostics[method]['cutoffs'],
                                                                  is_azure=is_azure)
            
            df_plot = pd.DataFrame({"observed_param": true_params.reshape(-1,),
                                    "probabilities": proba,
                                    "lower": lower,
                                    "upper": upper}).sort_values(by="observed_param")
            if 'energy' in method:
                zorder = 1
                alpha = 0.3
            elif '28' in method:
                zorder = 2
                alpha = 0.3
            elif 'full' in method:
                zorder = 3
                alpha = 0.7
            elif 'prediction' in method:
                zorder = 4
                alpha = 0.6
            else:
                zorder = 10
                alpha = 0.8
            ax[0].plot(df_plot.observed_param, df_plot.probabilities,
                       color=colors[i], label=f"{method}", zorder=zorder, linestyle=linestyles[i])
            ax[0].plot(df_plot.observed_param, df_plot.lower, color=colors[i], zorder=zorder, linestyle=linestyles[i])
            ax[0].plot(df_plot.observed_param, df_plot.upper, color=colors[i], zorder=zorder, linestyle=linestyles[i])
            ax[0].fill_between(x=df_plot.observed_param, y1=df_plot.lower, y2=df_plot.upper,
                               alpha=0.2, color=colors[i], zorder=zorder, linestyle=linestyles[i])     
        
        if plot_length_quantiles:  # only plot medians below
            # BINNED LENGTHS
            ax[1].vlines(x=bin_center, ymin=binned_length[method][0][0], ymax=binned_length[method][0][-1], color=colors[i],
                         linestyle="--", linewidth=1)
            ax[1].hlines(y=binned_length[method][0], xmin=bin_center-0.025, xmax=bin_center+0.025, color=colors[i])

        # BINNED INTERVALS
        # min ci
        ax[2].vlines(ymin=binned_confidence_sets[method][0][0][0], ymax=binned_confidence_sets[method][0][0][-1],
                     x=bin_center, color=colors[i], linestyle="--", linewidth=1)
        ax[2].hlines(y=binned_confidence_sets[method][0][0], xmin=bin_center-0.025, xmax=bin_center+0.025,
                     color=colors[i])
        ax[2].scatter(x=bin_center, y=binned_confidence_sets[method][0][0][1], color=colors[i], marker=length_markers[i], s=50, label=f"{method}")
        # max ci
        ax[2].vlines(ymin=binned_confidence_sets[method][0][1][0], ymax=binned_confidence_sets[method][0][1][-1],
                     x=bin_center, color=colors[i], linestyle="--", linewidth=1)
        ax[2].hlines(y=binned_confidence_sets[method][0][1], xmin=bin_center-0.025, xmax=bin_center+0.025,
                     color=colors[i])
        ax[2].scatter(x=bin_center, y=binned_confidence_sets[method][0][1][1], color=colors[i], marker=length_markers[i], s=50)
        if custom_bin_centers is None:
            bin_centers = [bin_center]
            bin_centers_interval_length = [bin_center_interval_length]
        for bin_idx in range(1, len(bin_edges)-1):
            if custom_bin_centers is None:
                bin_center = ((bin_edges[bin_idx + 1] + bin_edges[bin_idx]) / 2)+(i*methods_offset)
                bin_centers.append(bin_center)
                
                bin_center_interval_length = (bin_edges[bin_idx + 1] + bin_edges[bin_idx]) / 2
                bin_centers_interval_length.append(bin_center_interval_length)
            else:
                bin_center = custom_bin_centers[bin_idx]+(i*methods_offset)
            
            if plot_length_quantiles:
                # BINNED LENGTHS
                ax[1].vlines(x=bin_center, ymin=binned_length[method][bin_idx][0], ymax=binned_length[method][bin_idx][-1],
                             color=colors[i], linestyle="--", linewidth=1)
                ax[1].hlines(y=binned_length[method][bin_idx], xmin=bin_center-0.025, xmax=bin_center+0.025,
                             color=colors[i])

            # BINNED INTERVALS
            # min ci
            ax[2].vlines(x=bin_center, ymin=binned_confidence_sets[method][bin_idx][0][0],
                         ymax=binned_confidence_sets[method][bin_idx][0][-1], color=colors[i],
                         linestyle="--", linewidth=1)
            ax[2].hlines(y=binned_confidence_sets[method][bin_idx][0], xmin=bin_center-0.025, xmax=bin_center+0.025,
                         color=colors[i])
            ax[2].scatter(x=bin_center, y=binned_confidence_sets[method][bin_idx][0][1],
                          color=colors[i], marker=length_markers[i], s=50)
            # max ci
            ax[2].vlines(x=bin_center, ymin=binned_confidence_sets[method][bin_idx][1][0],
                         ymax=binned_confidence_sets[method][bin_idx][1][-1], color=colors[i],
                         linestyle="--", linewidth=1)
            ax[2].hlines(y=binned_confidence_sets[method][bin_idx][1], xmin=bin_center-0.025, xmax=bin_center+0.025,
                         color=colors[i])
            ax[2].scatter(x=bin_center, y=binned_confidence_sets[method][bin_idx][1][1],
                          color=colors[i], marker=length_markers[i], s=50)
        
        # BINNED LENGTHS
        # lines connecting medians
        if custom_bin_centers is not None:
            bin_centers = custom_bin_centers+(i*methods_offset)
        ax[1].plot(bin_centers_interval_length, [binned_length[method][bin_idx][1] for bin_idx in range(len(bin_centers))], 
                   color=colors[i], linewidth=2, label=f"{method}", linestyle=linestyles[i], marker=length_markers[i], markersize=10)
    
    # non-repeating elements
    ax[2].plot([start, end], [start, end], color="black", linestyle="--", linewidth=1, zorder=0)
    ax[0].hlines(y=confidence_level, xmin=start, xmax=end, color='black', linestyle="--", linewidth=3,
                     label=f"Nominal coverage = {round(100 * confidence_level, 1)} %", zorder=10)
    
    # formatting
    ax[0].set_xlabel(r"True Muon Energy $\theta$ [GeV]", fontsize=30)
    ax[0].set_ylabel("Coverage", fontsize=30)
    ax[0].set_ylim(bottom=0, top=1)#max(0, confidence_level-0.4)
    ax[0].tick_params(axis='both', which='major', labelsize=20, bottom=True, left=True, labelleft=True, labelbottom=True)
    ax[0].set_title("Coverage Diagnostics", fontsize=30)
    ax[0].legend(loc=(0.01, 0.01), prop={'size': 24})

    ax[1].set_xlabel(r"True Muon Energy $\theta$ [GeV]", fontsize=30)
    ax[1].set_ylabel("Median Length [GeV]", fontsize=30)
    ax[1].tick_params(axis='both', which='major', labelsize=20, bottom=True, left=True, labelleft=True, labelbottom=True)
    if custom_bin_centers is not None:
        ax[1].set_xticks(list(custom_bin_centers))
        ax[1].set_xlim(left=start, right=end)
    if plot_length_quantiles:
        ax[1].set_title(r"Percentiles ($25^{th}, 50^{th}, 75^{th}$) of intervals' length within each bin", fontsize=30)
    else:
        ax[1].set_title(r"Interval Length", fontsize=30)#.format(n_bins)
    ax[1].legend(loc=(0.16, 0.01), prop={'size': 24})
    
    ax[2].set_xlabel(r"True Muon Energy $\theta$ [GeV]", fontsize=30)
    ax[2].set_ylabel("Upper/Lower Bounds [GeV]", fontsize=30)
    if custom_bin_centers is not None:
        ax[2].set_xticks(list(custom_bin_centers))
        ax[2].set_xlim(left=start, right=end)
    ax[2].tick_params(axis='both', which='major', labelsize=20, bottom=True, left=True, labelleft=True, labelbottom=True)
    # ax[2].legend(loc=(0.30, 0.01), prop={'size': 20})
    ax[2].set_title("Confidence and Prediction Sets", fontsize=30)
    ax[2].legend(prop={'size': 20})
    
    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches='tight')
        
    mpl.rcParams['figure.dpi'] = 100  # reset to default
    plt.show()
    
    if return_bin_samples_count:
        return bin_samples_count

    
class Simulator:

    def __init__(self, d, observed_dims, param_grid_min, param_grid_max, grid_sample_size):
        self.d = d
        self.observed_dims = observed_dims
        if d == 1:
            self.param_grid = np.linspace(param_grid_min, param_grid_max, num=grid_sample_size)
        elif d == 2:
            param_grid_1d = np.linspace(param_grid_min, param_grid_max, num=grid_sample_size)
            # 2-dimensional grid of (grid_sample_size X grid_sample_size) points
            self.param_grid = np.transpose([np.tile(param_grid_1d, len(param_grid_1d)), np.repeat(param_grid_1d, len(param_grid_1d))])
            self.grid_sample_size = grid_sample_size**2
        else:
            raise NotImplementedError
        

def np_to_pd(array, names):
    return pd.DataFrame({names[i]: array[:, i] for i in range(len(names))})


def hpd_region(posterior, prior, param_grid, x, confidence_level, n_p_stars=100_000, tol=0.01):
    if posterior is None:
        # actually using prior here; naming just for consistency (should be changed)
        posterior_probs = torch.exp(prior.log_prob(torch.from_numpy(param_grid)))
    else:
        posterior_probs = torch.exp(posterior.log_prob(theta=param_grid, x=x))
    posterior_probs /= torch.sum(posterior_probs)  # normalize
    p_stars = np.linspace(0.99, 0, n_p_stars)  # thresholds to include or not parameters
    current_confidence_level = 1
    new_confidence_levels = []
    idx = 0
    while np.abs(current_confidence_level - confidence_level) > tol:
        if idx == n_p_stars:  # no more to examine
            break
        new_confidence_level = torch.sum(posterior_probs[posterior_probs >= p_stars[idx]])
        new_confidence_levels.append(new_confidence_level)
        if np.abs(new_confidence_level - confidence_level) < np.abs(current_confidence_level - confidence_level):
            current_confidence_level = new_confidence_level
        idx += 1
    # all params such that p(params|x) > p_star, where p_star is the last chosen one
    return current_confidence_level, param_grid[posterior_probs >= p_stars[idx-1], :], new_confidence_levels  