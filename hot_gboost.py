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
    dm.dig_up_features()  # todo: refactor this function
    dm.handle_na()  # todo: when to fill NaNs?

    data_train, data_test = dm.merge_tables()  # pandas.DataFrame

    data_train_y = data_train.pop(col_y)    # LKI: what is my purpose?
    _ = data_train.pop(col_user_id)  # do not predict on user id
    data_test_user_id_col = data_test.pop(col_user_id)


    # MODEL PIPELINE

    # Stratified K-Fold cross-validator.
    # Provides train/test indices to split data in train/test sets.
    # This cross-validation object is a variation of KFold that returns stratified folds.
    # The folds are made by preserving the percentage of samples for each class.
    # WIKI k-fold cross-validation: https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation
    skfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    y_preds = []
    val_scores = []

    # enumerate(iterable, start=0)
    # Return an enumerate object. iterable must support iteration.
    # __next__() returns a tuple containing count and value obtained from iterating over iterable.

    # split(X, y, groups=None)
    # Generate indices to split data into training and test set.
    for i, (tr_idcs, va_idcs) in enumerate(skfold.split(data_train,
                                                        data_train_y)):

        # iloc[<row selection>, <column selection>]
        # "iloc" in pandas is used to select rows and columns by number, in the order that they appear in teh data frame
        # https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/#iloc-selection

        x_tr, x_val = data_train.iloc[tr_idcs, :], data_train.iloc[va_idcs, :]
        y_tr, y_val = data_train_y[tr_idcs], data_train_y[va_idcs]

        print("\nStart LGBM for fold {}".format(i+1))
        model = lightgbm.LGBMClassifier(n_estimators=3500,
                                        **cfg.lgbm_cfg['params'])

        # LightGBM.fit()
        # Build a gradient boosting model from the training set (X,y).
        # http://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.LGBMModel.fit

        # Gradient Boosting from scratch
        # https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d

        # todo: reduce lr during training and cycle it as of some threshold
        model.fit(x_tr, y_tr, eval_set=(x_val, y_val),
                  verbose=-1, eval_metric='auc', early_stopping_rounds=150)

        # sklearn.metrics.roc_auc_score()
        # Compute Area Under the Receiver Operating Characteristics Curve (ROC AUC) from prediction scores.

        # Accuracy is measured by the area under the ROC curve.
        # area = 1.0: perfect
        # area = 0.5: fail

        # LightGBM.predict_proba()
        # Return the predicted probability for each class for each sample.

        score = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
        print('AUC:', score)
        val_scores.append(score)
        y_preds.append(model.predict_proba(data_test)[:, 1])

    print('\nMean AUC on valset: {}'.format(np.mean(val_scores)))
    # todo: Show also prediction distribution. What does that mean at all?

    y_preds_hmean_rank = harm_mean_of_ranks(y_preds)
    y_preds_mean = np.mean(y_preds)

    subm_mean = \
        pd.DataFrame({col_user_id: data_test_user_id_col,
                      col_y: y_preds_mean})
    subm_hmean_rank = \
        pd.DataFrame({col_user_id: data_test_user_id_col,
                      col_y: y_preds_hmean_rank})

    subm_file_name = 'sub.csv' if not debug_mode else 'sub_debug.csv'

    subm_mean.to_csv('data/out/mean_{}'.format(subm_file_name), index=False)
    subm_hmean_rank.to_csv('data/out/hmean_rank_{}'.format(subm_file_name),
                           index=False)
    plot_feat_importances(model, data_train.columns)


def plot_feat_importances(_mdl, _cols):
    fea_imp = pd.DataFrame({'imp': _mdl.feature_importances_,
                            'col': _cols})
    fea_imp = fea_imp.sort_values(['imp', 'col'],
                                  ascending=[False, True]).iloc[:30]
    print("")
    # todo: (LKI) print the mean importances across folds!
    print(fea_imp)
    if cfg.plot_cfg['do_plot']:
        plt.figure()
        fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))


def harm_mean_of_ranks(_preds):
    """calculates the harmonic mean of ranks

    _preds - list of lists of predictions
    """
    tmp = ss.hmean([ss.rankdata(x) for x in _preds])
    return (tmp - np.min(tmp)) / np.ptp(tmp)


if __name__ == '__main__':
    main()
