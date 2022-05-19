import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from tqdm import tqdm
import argparse
import os
import pickle

from muons_utils import MuonFeatures

    
class NeuralNetwork(torch.nn.Module):
    
    def __init__(self, 
                 input_size, hidden_layers, output_size, 
                 hidden_activation='relu', output_activation='linear', 
                 dropout_p=0.5):
        
        super().__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        
        self.dropout_p = dropout_p
        
        # hidden activation
        if hidden_activation == 'relu': self.hidden_activation = torch.nn.ReLU()
        else: self.hidden_activation = hidden_activation
        # output activation
        if output_activation == 'relu': self.output_activation = torch.nn.ReLU()
        else: self.output_activation = output_activation
        
        # input
        self.layers = [torch.nn.Linear(self.input_size, hidden_layers[0]), self.hidden_activation, torch.nn.BatchNorm1d(hidden_layers[0]), torch.nn.Dropout(p=self.dropout_p)]
        # hidden
        for i in range(1, len(hidden_layers)-1):
            self.layers += [torch.nn.Linear(hidden_layers[i], hidden_layers[i+1]), 
                            self.hidden_activation, 
                            torch.nn.BatchNorm1d(hidden_layers[i+1]), 
                            torch.nn.Dropout(p=self.dropout_p)]
        # output
        self.layers += [torch.nn.Linear(hidden_layers[-1], output_size)]
        if output_activation != 'linear':
            self.layers += [self.output_activation]
        # put together
        self.layers = torch.nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)
    

class NetworkLearner:
    
    def __init__(self, model, optimizer='adam', loss='mse', device=None):
        
        if device is None:
            if (torch.cuda.is_available()):
                print('CUDA AVAILABLE')
                print('Number of devices {}'.format(torch.cuda.device_count()))
                print('First Device name {}'.format(torch.cuda.get_device_name(0)))
                self.device = torch.device('cuda')
            else:
                print('NO GPU DETECTED, DEFAULTING TO CPU')
                self.device = torch.device('cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        self.model = model.to(self.device)
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters())
        else:
            self.optimizer = optimizer(self.model.parameters())
        if loss == 'mse':
            self.loss = torch.nn.MSELoss().to(self.device)
        else:
            self.loss = loss.to(self.device)
        self.loss_trajectory = []
    
    def forward(self, x):
        return self.model(x)
    
    def fit(self, X, y, epochs, batch_size):
        
        self.model.train()
        for epc in tqdm(range(epochs), desc="Training NN"):
            shuffle_idx = np.arange(X.shape[0])
            np.random.shuffle(shuffle_idx)
            X = X[shuffle_idx, :]
            y = y[shuffle_idx]
            epoch_losses = []
            for idx in range(0, X.shape[0], batch_size):
                self.optimizer.zero_grad()
                
                batch_X = torch.from_numpy(
                    X[idx: min(idx + batch_size, X.shape[0]), :]
                ).float().to(self.device).requires_grad_(False)
                batch_y = torch.from_numpy(
                    y[idx: min(idx + batch_size, y.shape[0])].reshape(-1,1)
                ).float().to(self.device).requires_grad_(False)
                
                batch_predictions = self.model(batch_X)
                batch_loss = self.loss(batch_predictions, batch_y)
                batch_loss.backward()
                self.optimizer.step()
                epoch_losses.append(batch_loss.item())
            self.loss_trajectory.append(np.mean(epoch_losses))
    
    def fit_cv(self, X, y, n_folds, epochs, batch_size):
        # get folds idxs
        splitter = KFold(n_splits=n_cv_folds, shuffle=True)
        
        rmse_scores = []
        for train_index, test_index in splitter.split(X):
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            torch_nn = NetworkLearner(model=NeuralNetwork(input_size=self.model.input_size, 
                                                          hidden_layers=self.model.hidden_layers, 
                                                          output_size=self.model.output_size, 
                                                          hidden_activation=self.model.hidden_activation, 
                                                          output_activation=self.model.output_activation, 
                                                          dropout_p=self.model.dropout_p), 
                                      optimizer=self.optimizer,
                                      loss=self.loss,
                                      device=self.device)
            torch_nn.fit(X=X_train, y=y_train, epochs=epochs, batch_size=batch_size)
            rmse_score = mean_squared_error(y_true=y_test, y_pred=torch_nn.predict(X=X_test), squared=False)
            rmse_scores.append(rmse_score)
        return np.mean(rmse_scores)
        
    def predict(self, X):
        self.model.eval()
        X_torch = torch.from_numpy(X).float().to(self.device).requires_grad_(False)
        return self.model(X_torch).cpu().detach().numpy()
    
    def plot_epoch_loss(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(range(len(self.loss_trajectory)), self.loss_trajectory)
        ax.set_xlabel('Epoch', fontsize=18)
        ax.set_ylabel('Loss', fontsize=18)
        ax.set_title(f'Final RMSE: {round(np.sqrt(self.loss_trajectory[-1]), 2)}', fontsize=18)
        plt.show()

        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_data_path', action="store", type=str,
                        help='Path at which simulated data is stored, withouth filename')
    parser.add_argument('--obs_data_path', action="store", type=str,
                        help='Path at which observed data is stored')
    parser.add_argument('--observed_d', action="store", type=int,
                        help='Dimensionality of data')
    parser.add_argument('--param_grid_bounds', action="store", nargs='+', type=int,
                        help='Lower and upper bounds of parameter space')
    parser.add_argument('--calibration_size', action="store", type=int, 
                        help='Sample size used for estimating critical values')
    
    parser.add_argument('--hidden_layers', action="store", nargs='+', type=int,
                        help='Tuple where the i-th entry is the number of neurons in the i-th hidden layer')
    parser.add_argument('--hidden_activation', action="store", type=str, default='relu',
                        help='Activation function used in hidden layers')
    #parser.add_argument('--output_activation', action="store", type=str,
    #                    help='Activation function used in output layer')
    parser.add_argument('--dropout_p', action="store", type=float, default=0.5,
                        help='Dropout probability')
    parser.add_argument('--optimizer', action="store", type=str, default='adam',
                        help='Optimizer for the network')
    parser.add_argument('--loss', action="store", type=str, default='mse',
                        help='Loss to be minimized during training')
    parser.add_argument('--device', action="store", type=str, default='cuda',
                        help='Loss to be minimized during training')
    parser.add_argument('--epochs', action="store", type=int,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', action="store", type=int,
                        help='Batch size to be used during training')
    parser.add_argument('--save_path', action="store", type=str,
                        help='Path where to save stuff')
    argument_parsed = parser.parse_args()
    
    # data and simulator
    simulated_data = pd.read_csv(argument_parsed.sim_data_path, sep=" ", header=None)
    observed_data = pd.read_csv(argument_parsed.obs_data_path, sep=" ", header=None)
    simulator = MuonFeatures(param_grid_bounds=argument_parsed.param_grid_bounds, 
                         param_grid_size=10_000, 
                         simulated_data=simulated_data.to_numpy(), observed_data=observed_data.to_numpy(),
                         observed_d=argument_parsed.observed_d, param_d=1, 
                         param_column=-1)
    simulator.split_simulated_data(calibration_size=argument_parsed.calibration_size)
    
    # conditional mean
    cond_mean_arch = NeuralNetwork(input_size=simulator.observed_d, output_size=simulator.d, 
                                   hidden_layers=argument_parsed.hidden_layers, 
                                   hidden_activation=argument_parsed.hidden_activation, output_activation='linear', 
                                   dropout_p=argument_parsed.dropout_p)
    cond_mean_learner = NetworkLearner(model=cond_mean_arch, optimizer=argument_parsed.optimizer, loss=argument_parsed.loss, device=argument_parsed.device)
    cond_mean_learner.fit(X=simulator.b_sample_x, y=simulator.b_sample_theta, epochs=argument_parsed.epochs, batch_size=argument_parsed.batch_size)
    with open(os.path.join(argument_parsed.save_path, f'cond_mean_nn_{simulator.observed_d}d.pickle'), 'wb') as f:
        pickle.dump(cond_mean_learner, f)
        
    # conditional var
    conditional_var_response = ((simulator.b_sample_theta.reshape(-1,1) - cond_mean_learner.predict(X=simulator.b_sample_x).reshape(-1,1))**2).reshape(-1,)
    cond_var_arch = NeuralNetwork(input_size=simulator.observed_d, output_size=simulator.d, 
                                  hidden_layers=argument_parsed.hidden_layers, 
                                  hidden_activation=argument_parsed.hidden_activation, output_activation='relu', 
                                  dropout_p=argument_parsed.dropout_p)
    cond_var_learner = NetworkLearner(model=cond_var_arch, optimizer=argument_parsed.optimizer, loss=argument_parsed.loss, device=argument_parsed.device)
    cond_var_learner.fit(X=simulator.b_sample_x, y=conditional_var_response, epochs=argument_parsed.epochs, batch_size=argument_parsed.batch_size)
    with open(os.path.join(argument_parsed.save_path, f'cond_var_nn_{simulator.observed_d}d.pickle'), 'wb') as f:
        pickle.dump(cond_var_learner, f)
        
    with open(os.path.join(argument_parsed.save_path, f'simulator_nn_{simulator.observed_d}d.pickle'), 'wb') as f:
        pickle.dump(simulator, f)