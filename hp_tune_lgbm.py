from hyperopt import tpe
from hyperopt.fmin import fmin
from skopt import BayesSearchCV
import uuid
import pandas as pd
import numpy as np
import lightgbm
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, \
    cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

from preprocessing.data import DataManager
import preprocessing.config as cfg


def hyperopt_objective(sampled_params):
    # todo: Ugly shit
    param_dict = {'num_leaves': False,
                  'max_depth': False,
                  'scale_pos_weight': True,
                  'colsample_bytree': True,
                  'min_child_weight': True,
                  'random_state': False
                  }
    converted_params = {}
    for p_name, p_range in sampled_params.items():
        if param_dict[p_name]:
            # True -> real_valued
            converted_params[p_name] = '{:.3f}'.format(p_range)
        else:
            # False -> integer
            converted_params[p_name] = int(p_range)
    clf = lightgbm.LGBMRegressor(n_estimators=10, **converted_params)
    score = cross_val_score(clf, X, Y, scoring=make_scorer(mean_squared_error),
                            cv=TimeSeriesSplit())
    print("MSE: {:.3f} params {:}".format(score.mean(), converted_params))
    return score


def status_print(optim_result):
    """Status callback during bayesian hyperparameter search"""

    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(opt_search.cv_results_)

    # Get current parameters and the best parameters
    best_params = pd.Series(opt_search.best_params_)
    print('Model #{}\nBest L2: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(opt_search.best_score_, 4),
        best_params))

    # Save all model results
    clf_name = opt_search.estimator.__class__.__name__
    all_models.to_csv(clf_name + "_cv_results.csv")


if __name__ == '__main__':
    # config
    if cfg.data_cfg['save_predictions']:
        model_uuid = str(uuid.uuid4())[:6]
        print('model uuid: {}'.format(model_uuid))

    dm = DataManager(cfg.data_cfg['file_path'], create_hold_out=False)

    # featurize dataset (feature engineering)
    tra_df, _, tst_df = dm.get_featurized_sets()

    if False:
        # hyperopt
        print("Start hyperopt")
        X = tra_df[dm.cl.x_cols]
        Y = tra_df[dm.cl.y_cols[0]]
        space = cfg.lgbm_cfg['hp_hyperopt_space']

        best = fmin(fn=hyperopt_objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=50)
        print("Hyperopt estimated optimum {}".format(best))
    else:
        # skopt bayes search

        model = lightgbm.LGBMRegressor(n_estimators=10000)
        tscv = TimeSeriesSplit()

        hyper_params = cfg.lgbm_cfg['hp_skopt_space']
        opt_search = \
            BayesSearchCV(model, n_iter=2, search_spaces=hyper_params,
                          iid=False, cv=tscv, random_state=2018)
        opt_search.fit(tra_df[dm.cl.x_cols],
                       tra_df[dm.cl.y_cols[0]],
                       callback=status_print)

