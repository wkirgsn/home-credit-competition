from hyperopt import tpe
from hyperopt.fmin import fmin
from skopt import BayesSearchCV
import uuid
import pandas as pd
import numpy as np
import lightgbm
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, \
    StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score

from preprocessing.data import DataManager
import preprocessing.config as cfg

N_SEARCH_ITERATIONS = cfg.skopt_cfg['n_iter']
SEED = 2018
N_FOLDS = 5
col_user_id = 'SK_ID_CURR'
col_y = 'TARGET'


def status_print(optim_result):
    """Status callback during bayesian hyperparameter search"""

    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(opt_search.cv_results_)

    # Get current parameters and the best parameters
    best_params = pd.Series(opt_search.best_params_)
    print('Model #{}\nBest AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(opt_search.best_score_, 4),
        best_params))

    # Save all model results
    clf_name = opt_search.estimator.__class__.__name__
    all_models.to_csv('data/out/' + clf_name + "_cv_results.csv")


if __name__ == '__main__':
    # config
    if cfg.data_cfg['save_predictions']:
        model_uuid = str(uuid.uuid4())[:6]
        print('model uuid: {}'.format(model_uuid))

    dm = DataManager()
    dm.factorize_categoricals()
    data_train, data_test = dm.merge_tables()

    data_train_y = data_train.pop(col_y)
    _ = data_train.pop(col_user_id)  # do not predict on user id
    data_test_user_id_col = data_test.pop(col_user_id)

    data_train, data_test = dm.handle_na(data_train, data_test)

    # MODEL PIPELINE

    skfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    model = lightgbm.LGBMClassifier(verbose=-1)

    hyper_params = cfg.lgbm_cfg['hp_skopt_space']
    opt_search = \
        BayesSearchCV(model,
                      n_iter=N_SEARCH_ITERATIONS,
                      search_spaces=hyper_params,
                      iid=True,
                      cv=skfold,
                      random_state=SEED,
                      scoring='roc_auc',
                      fit_params={'eval_metric': 'auc',
                                  'verbose': 100})
    opt_search.fit(data_train, data_train_y, callback=status_print)

    status_print(None)

