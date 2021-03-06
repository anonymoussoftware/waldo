import torch
import lightgbm as lgb
import numpy as np
import sys
sys.path.append("..")

from sklearn.ensemble import GradientBoostingRegressor
# from skgarden import RandomForestQuantileRegressor
from statsmodels.api import QuantReg
from functools import partial
from qr_utils.pytorch_functions import q_model, Learner, QuantileLoss, q_model_3l


def train_qr_algo(model_obj, theta_mat, stats_mat, algo_name, learner_kwargs, pytorch_kwargs, alpha, prediction_grid, nn_dropout=0.1):
    # Train the regression quantiles algorithms
    if algo_name == 'xgb':
        model = GradientBoostingRegressor(loss='quantile', alpha=alpha, **learner_kwargs)
        model.fit(theta_mat.reshape(-1, model_obj.d), stats_mat.reshape(-1, ))
        pred_vec = model.predict(prediction_grid.reshape(-1, model_obj.d))
    # TODO: fix dependency of skgarden on old version of sklearn
    #elif algo_name == 'rf':
    #    model = RandomForestQuantileRegressor(**learner_kwargs)
    #    model.fit(theta_mat.reshape(-1, model_obj.d), stats_mat.reshape(-1, ))
    #    pred_vec = model.predict(prediction_grid.reshape(-1, model_obj.d), quantile=alpha * 100)
    elif algo_name == 'lgb':
        model = lgb.LGBMRegressor(objective='quantile', alpha=alpha, **learner_kwargs)
        model.fit(theta_mat.reshape(-1, model_obj.d), stats_mat.reshape(-1, ))
        pred_vec = model.predict(prediction_grid.reshape(-1, model_obj.d))
    elif algo_name == 'pytorch':
        model = q_model([alpha], dropout=nn_dropout, in_shape=model_obj.d, **pytorch_kwargs)
        loss_func = QuantileLoss(quantiles=[alpha])
        learner = Learner(model, partial(torch.optim.Adam, weight_decay=1e-6),
                          loss_func, device="cuda" if torch.cuda.is_available() else 'cpu')
        learner.fit(theta_mat.reshape(-1, model_obj.d), stats_mat.reshape(-1, ),
                    **learner_kwargs)
        pred_vec = learner.predict(prediction_grid.reshape(-1, model_obj.d).astype(np.float32))
        model = learner  # just for returning it
    elif algo_name == 'pytorch_3l':
        model = q_model_3l([alpha], dropout=nn_dropout, in_shape=model_obj.d, **pytorch_kwargs)
        loss_func = QuantileLoss(quantiles=[alpha])
        learner = Learner(model, partial(torch.optim.Adam, weight_decay=1e-6),
                          loss_func, device="cuda" if torch.cuda.is_available() else 'cpu')
        learner.fit(theta_mat.reshape(-1, model_obj.d), stats_mat.reshape(-1, ),
                    **learner_kwargs)
        pred_vec = learner.predict(prediction_grid.reshape(-1, model_obj.d).astype(np.float32))
        model = learner  # just for returning it
    elif algo_name == 'linear':
        model = None
        pred_vec = QuantReg(theta_mat.reshape(-1, model_obj.d), stats_mat.reshape(-1, )).fit(q=alpha).predict(
            prediction_grid.reshape(-1, model_obj.d)
        )
    else:
        raise ValueError('CDE Classifier not defined in the file.')

    return model, pred_vec
