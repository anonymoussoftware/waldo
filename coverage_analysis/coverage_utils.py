import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_band(simulator, 
              observed_theta,
              wald_confidence_sets,
              posterior_sets,
              waldo_confidence_sets,
              n_bins, confidence_level, save_fig_path=None):
             
    # binning
    bin_edges = np.linspace(-7, 7, num=n_bins)
    bin_aggregate_wald = []
    bin_aggregate_posterior = []
    bin_aggregate_waldo = []
    bin_n_samples = []
    for bin_idx in tqdm(range(len(bin_edges)-1)):
        bin_samples_idx = np.where(((observed_theta >= bin_edges[bin_idx]) & (observed_theta <= bin_edges[bin_idx+1])).reshape(-1,))[0]
        bin_n_samples.append(len(bin_samples_idx))  # append before deleting those with empty confidence sets, otherwise mismatch in length of x and y when plotting

        # wald
        bin_samples_idx_wald = [idx for idx in bin_samples_idx if len(wald_confidence_sets[idx]) > 0]  # avoid empty confidence sets
        bin_confidence_sets_wald = [wald_confidence_sets[idx] for idx in bin_samples_idx_wald]
        ci_min_wald = [np.min(ci) for ci in bin_confidence_sets_wald]
        ci_max_wald = [np.max(ci) for ci in bin_confidence_sets_wald]
        bin_aggregate_wald.append(
           np.array([
               np.median(ci_min_wald),
               np.median(ci_max_wald),
           ])
        )

        # posterior
        bin_samples_idx_posterior = bin_samples_idx
        bin_confidence_sets_posterior = [posterior_sets[idx, :] for idx in bin_samples_idx_posterior]   
        ci_min_posterior = [ci[0] for ci in bin_confidence_sets_posterior]
        ci_max_posterior = [ci[1] for ci in bin_confidence_sets_posterior]
        bin_aggregate_posterior.append(
           np.array([
               np.median(ci_min_posterior),
               np.median(ci_max_posterior),
           ])
        )

        # waldo
        bin_samples_idx_waldo = [idx for idx in bin_samples_idx if len(waldo_confidence_sets[idx]) > 0]
        bin_confidence_sets_waldo = [waldo_confidence_sets[idx] for idx in bin_samples_idx_waldo]
        ci_min_waldo = [np.min(ci) for ci in bin_confidence_sets_waldo]
        ci_max_waldo = [np.max(ci) for ci in bin_confidence_sets_waldo]
        bin_aggregate_waldo.append(
           np.array([
               np.median(ci_min_waldo),
               np.median(ci_max_waldo),
           ])
        )

        # waldo pred
        #bin_samples_idx_waldo_pred = bin_samples_idx
        #bin_confidence_sets_waldo_pred = [waldo_prediction_sets[idx, :] for idx in bin_samples_idx_waldo_pred]   
        #ci_min_waldo_pred = [ci[0] for ci in bin_confidence_sets_waldo_pred]
        #ci_max_waldo_pred = [ci[1] for ci in bin_confidence_sets_waldo_pred]
        #bin_aggregate_waldo_pred.append(
        #   np.array([
        #       np.median(ci_min_waldo_pred),
        #       np.median(ci_max_waldo_pred),
        #   ])
        #)
    
    # PLOT
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    # wald
    ax.plot(np.sort(observed_theta), np.concatenate([np.repeat(bin_aggregate_wald[idx][0], bin_n_samples[idx]) for idx in range(len(bin_aggregate_wald))]), 
             zorder=1, color="crimson", linestyle="-", label=f"Wald", linewidth=5)
    ax.plot(np.sort(observed_theta), np.concatenate([np.repeat(bin_aggregate_wald[idx][1], bin_n_samples[idx]) for idx in range(len(bin_aggregate_wald))]), 
             zorder=1, color="crimson", linestyle="-", linewidth=5)

    # posterior
    ax.plot(np.sort(observed_theta), np.concatenate([np.repeat(bin_aggregate_posterior[idx][0], bin_n_samples[idx]) for idx in range(len(bin_aggregate_posterior))]), 
             zorder=3, color="darkgreen", linestyle="-.", label=f"Prediction Sets", linewidth=5)
    ax.plot(np.sort(observed_theta), np.concatenate([np.repeat(bin_aggregate_posterior[idx][1], bin_n_samples[idx]) for idx in range(len(bin_aggregate_posterior))]), 
             zorder=3, color="darkgreen", linestyle="-.", linewidth=5)

    # waldo
    ax.plot(np.sort(observed_theta), np.concatenate([np.repeat(bin_aggregate_waldo[idx][0], bin_n_samples[idx]) for idx in range(len(bin_aggregate_waldo))]), 
             zorder=2, color="orange", linestyle="--", label=f"Waldo", linewidth=5)
    ax.plot(np.sort(observed_theta), np.concatenate([np.repeat(bin_aggregate_waldo[idx][1], bin_n_samples[idx]) for idx in range(len(bin_aggregate_waldo))]), 
             zorder=2, color="orange", linestyle="--", linewidth=5)

    # waldo pred
    #ax.plot(np.sort(observed_theta), np.concatenate([np.repeat(bin_aggregate_waldo_pred[idx][0], bin_n_samples[idx]) for idx in range(len(bin_aggregate_waldo_pred))]), 
    #         zorder=1, color="orange", linestyle="--", label=f"{int(confidence_level*100)}% Waldo prediction sets")
    #ax.plot(np.sort(observed_theta), np.concatenate([np.repeat(bin_aggregate_waldo_pred[idx][1], bin_n_samples[idx]) for idx in range(len(bin_aggregate_waldo_pred))]), 
    #         zorder=1, color="orange", linestyle="--")


    ax.plot([-7, 7], [-7, 7], color="black", linestyle="--", linewidth=5, label="Bisector", zorder=0)
    ax.set_xlabel(r"$\theta$", fontsize=45)
    ax.set_ylabel("Regions", fontsize=45)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    #ax.set_xticks(range(-7, 8, 1))
    #ax.set_yticks(range(-7, 8, 1))
    # ax.set_yticks(range(int(model.low_int), int(model.high_int + 0.2*model.high_int), int(0.2*model.high_int)))
    ax.tick_params(axis='both', which='major', labelsize=30, bottom=True, left=True, labelleft=True, labelbottom=True)
    ax.legend(loc="best", prop={'size': 40})
    ax.set_title(r"Parameter Regions", fontsize=45, pad=8)
        
    if save_fig_path is not None:
        plt.savefig(save_fig_path, bbox_inches="tight")
    plt.show()