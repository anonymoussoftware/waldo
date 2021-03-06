classifier_cde_dict = {
    'linear': ('linear', None),  # None is just for consistency of the interface with other models
    'xgb_d3_n100': ('xgb', {'max_depth': 3, 'n_estimators': 100}),
    'xgb_d3_n500': ('xgb', {'max_depth': 3, 'n_estimators': 500}),
    'xgb_d3_n250': ('xgb', {'max_depth': 3, 'n_estimators': 250}),
    'xgb_d5_n250': ('xgb', {'max_depth': 5, 'n_estimators': 250}),
    'xgb_d5_n100': ('xgb', {'max_depth': 5, 'n_estimators': 100}),
    'xgb_d5_n500': ('xgb', {'max_depth': 5, 'n_estimators': 500}),
    'xgb_d10_n100': ('xgb', {'max_depth': 10, 'n_estimators': 100}),
    'xgb_d10_n250': ('xgb', {'max_depth': 10, 'n_estimators': 100}),
    'RF100': ('rf', {'n_estimators': 100}),
    'RF200': ('rf', {'n_estimators': 200}),
    'RF250': ('rf', {'n_estimators': 250}),
    'RF500': ('rf', {'n_estimators': 500}),
    'lgb': ('lgb', {'num_leaves': 128, 'learning_rate': 0.1, 'n_estimators': 100,
                    'reg_sqrt': True, 'max_depth': 5}),
    'pytorch': ('pytorch', {'epochs': 500, 'batch_size': 50}, {'neur_shapes': (64, 64)}),
    'pytorch_a': ('pytorch', {'epochs': 500, 'batch_size': 50}, {'neur_shapes': (64, 32)}),
    'pytorch_b': ('pytorch', {'epochs': 500, 'batch_size': 50}, {'neur_shapes': (32, 16)}),
    'pytorch_3l': ('pytorch_3l', {'epochs': 500, 'batch_size': 50}, {'neur_shapes': (64, 64, 32)}),
    'pytorch_3l_a': ('pytorch_3l', {'epochs': 500, 'batch_size': 50}, {'neur_shapes': (64, 32, 32)}),
    'pytorch_3l_b': ('pytorch_3l', {'epochs': 500, 'batch_size': 50}, {'neur_shapes': (64, 32, 16)}),
    'pytorch_3l_c': ('pytorch_3l', {'epochs': 500, 'batch_size': 50}, {'neur_shapes': (32, 32, 16)})
}