import numpy as np
import scipy.stats as ss
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm

import preprocessing.config as cfg
from preprocessing.data import DataManager

"""Sources: 
https://www.kaggle.com/yekenot/catboostarter/code
https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm/code
https://www.kaggle.com/pavanraj159/loan-repayers-v-s-loan-defaulters-home-credit
"""
warnings.filterwarnings("ignore")
SEED = 2018
N_FOLDS = 5
np.random.seed(SEED)

col_user_id = 'SK_ID_CURR'
col_y = 'TARGET'

# todo: Train different models for those applicants w/ many NANs and those w/out
# todo: feat idea: Bool -> Applied before 1pm or after
# todo: feat idea: amt_credit²+amt_annuity² plus threshold bool
# todo: feat idea: has low-skill occupation y/n


def main():
    debug_mode = cfg.debug_cfg['DEBUG']

    # FEATURE ENGINEERING

    dm = DataManager()
    dm.factorize_categoricals()
    data_train, data_test = dm.merge_tables()

    data_train_y = data_train.pop(col_y)
    _ = data_train.pop(col_user_id)  # do not predict on user id
    data_test_user_id_col = data_test.pop(col_user_id)

    data_train, data_test = dm.handle_na(data_train, data_test)

    # MODEL PIPELINE

    skfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    y_preds = []
    val_scores = []
    for i, (tr_idcs, va_idcs) in enumerate(skfold.split(data_train,
                                                        data_train_y)):
        x_tr, x_val = data_train.iloc[tr_idcs, :], data_train.iloc[va_idcs, :]
        y_tr, y_val = data_train_y.iloc[tr_idcs], data_train_y[va_idcs]

        print("\nStart LGBM for fold {}".format(i+1))
        model = lightgbm.LGBMClassifier(**cfg.lgbm_cfg['params_found_by_skopt'])

        model.fit(x_tr, y_tr, eval_set=(x_val, y_val),
                  verbose=100, eval_metric='auc', early_stopping_rounds=150)
        score = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
        print('AUC:', score)
        val_scores.append(score)
        y_preds.append(model.predict_proba(data_test)[:, 1])

    print('\nMean AUC on valset: {}'.format(np.mean(val_scores)))
    y_preds = ss.hmean([ss.rankdata(x) for x in y_preds])  # harm. mean of rank
    # y_preds = np.mean(y_preds)  # mean of predictions

    subm = pd.DataFrame({col_user_id: data_test_user_id_col,
                         col_y: y_preds})

    subm_file_name = 'submissions.csv' if not debug_mode else \
        'submissions_debug.csv'

    subm.to_csv('data/out/{}'.format(subm_file_name), index=False)
    # plot_feat_importances(model, data_train.columns)


def plot_feat_importances(_mdl, _cols):
    fea_imp = pd.DataFrame({'imp': _mdl.feature_importances_,
                            'col': _cols})
    fea_imp = fea_imp.sort_values(['imp', 'col'],
                                  ascending=[True, False]).iloc[-30:]
    plt.figure()
    fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))


if __name__ == '__main__':
    main()
