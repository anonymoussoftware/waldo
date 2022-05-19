import torch
from functools import partial
import sys
import numpy as np
from pathlib import Path
import timeit
from typing import Optional, List, Dict, Tuple, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


def conditional_var_response(fy_cond_mean, fy_cond_var, pred_name: str, return_fy: bool = False):
    '''
    Compute (theta - E[theta|x])^2 and set it as new target in fy_cond_var to estimate conditional variance
    '''
    for fold_name in fy_cond_mean.foldfile:
        if fold_name != 'meta_data':
            fy_cond_var.foldfile[fold_name]['targets'][...] = ((fy_cond_var.foldfile[fold_name]['targets'][()].reshape(-1, 1) - 
                                                                fy_cond_mean.foldfile[fold_name][pred_name][()].reshape(-1, 1))**2).reshape(-1, )
    if return_fy:
        return fy_cond_mean, fy_cond_var
    

def unfold_data(fy, name: str, exclude_name: Union[str, None]):
    
    if exclude_name is None:
        exclude_name = 'None'
    
    data = [fy.foldfile[fold_name][name][()] for fold_name in tqdm(fy.foldfile) if (fold_name != 'meta_data') and (exclude_name not in fold_name)]
    if data[0].ndim == 1:
        return np.concatenate(data)
    elif (data[0].ndim == 2) and (data[0].shape[1] == 1):
        data = [array.reshape(-1, ) for array in data]
        return np.concatenate(data)
    elif data[0].ndim == 2:
        return np.concatenate(data, axis=0)
    else:
        raise NotImplementedError
        

def plot_waldo_predictions(x, y, prediction, which_prediction, observed_d):
    if observed_d == 1:
        fig, ax = plt.subplots(1, 2, figsize=(24, 8))
        
        sns.scatterplot(x=x, y=y, alpha=0.1, label="Training data", ax=ax[0])
        sns.scatterplot(x=x, y=prediction, alpha=0.1, label=f"Predicted {which_prediction}", ax=ax[0])
        ax[0].set_xlabel(r"$x$", fontsize=18)
        ax[0].set_ylabel(r"$y$", fontsize=18)

        sns.scatterplot(x=y, y=prediction, alpha=0.1)
        ax[1].plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], linestyle='--', linewidth=1, color='black', label="bisector")
        ax[1].set_xlabel('True', fontsize=18)
        ax[1].set_ylabel('Predicted', fontsize=18)
        plt.show()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        sns.scatterplot(x=y, y=prediction, alpha=0.1)
        ax.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], linestyle='--', linewidth=1, color='black', label="bisector")
        ax.set_xlabel('True', fontsize=18)
        ax.set_ylabel('Predicted', fontsize=18)
        ax.set_title(f'RMSE: {round(np.sqrt(np.mean(np.square(np.subtract(y.reshape(-1, 1), prediction.reshape(-1, 1))))), 2)}', fontsize=18)
        plt.show()
        

class MuonFeatures:
    
    def __init__(self, 
                 param_grid_bounds, param_grid_size,
                 simulated_data, observed_data, 
                 observed_d, param_d,
                 param_column=-1):
        
        self.param_grid = np.linspace(param_grid_bounds[0], param_grid_bounds[1], num=param_grid_size)
        self.param_grid_size = param_grid_size
        
        self.observed_d = observed_d
        self.d = param_d
        
        self.simulated_data = simulated_data
        self.observed_data = observed_data
        
        self.param_column = -1
        
    def split_simulated_data(self, calibration_size: Union[int, float]):
        
        data_shuffled = self.simulated_data[np.random.permutation(self.simulated_data.shape[0]), :]
        if calibration_size > 1:
            estimation_set_end_idx = self.simulated_data.shape[0] - calibration_size
        else:
            estimation_set_end_idx = int(np.floor((1-calibration_size)*self.simulated_data.shape[0]))
        self.b_sample_theta, self.b_sample_x = self.simulated_data[:estimation_set_end_idx, self.param_column], self.simulated_data[:estimation_set_end_idx, :self.param_column]
        self.b_prime_sample_theta, self.b_prime_sample_x = self.simulated_data[estimation_set_end_idx:, self.param_column], self.simulated_data[estimation_set_end_idx:, :self.param_column]
        

class TempQRSimulator:
    
    def __init__(self, d):
        self.d = d
        