"""HOME CREDIT COMPETITION 2018"""

debug_cfg = {'DEBUG': False,
             'choose_debug_on_gpu_availability': False,
             'n_debug': 1000,  # first n timestamps to use if debug
             }

data_cfg = {
    'file_path': "data/input/measures.csv",
    'db_path': 'data/results.db',
    'save_predictions': True,
    }


plot_cfg = {'do_plot': True, }

keras_cfg = {
    'early_stop_patience': 30,
    'use_cpu': False,

    'params': {
        'batch_size': 64,
        'n_layers': 1,
        'n_units': 64,
        'epochs': 50,
        'arch': 'gru',  # gru, lstm or rnn
        'kernel_reg': 1e-2,
        'activity_reg': 1e-2,
        'recurrent_reg': 1e-2,
    },
    'hp_skopt_space': {
        'arch': ['lstm', 'gru'],
        'epochs': [100, 150, 200],
        'n_layers': (1, 5),
        'n_units': (4, 2048),
        'kernel_reg': (1e-9, 1e-1, 'log-uniform'),
        'activity_reg': (1e-9, 1e-1, 'log-uniform'),
        'recurrent_reg': (1e-9, 1e-1, 'log-uniform'),
        'dropout_rate': (0.3, 0.7, 'uniform'),
        'optimizer': ['adam', 'nadam', 'adamax', 'sgd', 'rmsprop']
    },

}

lgbm_cfg = {
    'params': {'n_estimators':4000,
                'learning_rate':0.03,
                'num_leaves':30,
                'colsample_bytree':.8,
                'subsample':.9,
                'max_depth':7,
                'reg_alpha':.1,
                'reg_lambda':.1,
                'min_split_gain':.01,
                'min_child_weight':2,
                'silent':True,
                'verbose':-1,
                'random_state': 10
       },
    'params_found_by_skopt': {
        'max_depth': 10,
         'random_state': 2649,
         'scale_pos_weight': 5.999009745067044,
         'learning_rate': 0.012297696592148952,
         'min_child_weight': 36.53309768210956,
         'colsample_bytree': 0.6526428869408192,
        'num_leaves': 44,
         'min_child_samples': 30,
         'reg_alpha': 1e-09,
         'n_estimators': 4530,
         'subsample_freq': 10,
         'reg_lambda': 1000.0,
         'subsample': 0.9408339680936121
    },
    'hp_skopt_space': {
        'n_estimators': (1000, 8000),
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'num_leaves': (2, 64),
        'max_depth': (2, 16),
        'min_child_weight': (0.01, 100, 'log-uniform'),
        'min_child_samples': (1, 50),
        'subsample': (0.01, 1.0, 'uniform'),
        'subsample_freq': (0, 10),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'random_state': (2000, 3000),  # arbitrary
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'scale_pos_weight': (1e-6, 10000, 'log-uniform'),
        },

}

skopt_cfg = {
    'n_iter': 100
}

