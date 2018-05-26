import numpy as np
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
"""
warnings.filterwarnings("ignore")
SEED = 2018
N_FOLDS = 10
np.random.seed(SEED)

col_user_id = 'SK_ID_CURR'
col_y = 'TARGET'


def main():

    dm = DataManager()
    dm.factorize_categoricals()
    data_train, data_test = dm.merge_tables()

    data_train_y = data_train.pop(col_y)
    _ = data_train.pop(col_user_id)  # do not predict on user id
    data_test_user_id_col = data_test.pop(col_user_id)

    data_train, data_test = dm.handle_na(data_train, data_test)

    # MODEL PIPELINE

    skfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    y_preds = 0
    for i, (tr_idcs, va_idcs) in enumerate(skfold.split(data_train,
                                                        data_train_y)):
        x_tr, x_val = data_train.iloc[tr_idcs, :], data_train.iloc[va_idcs, :]
        y_tr, y_val = data_train_y.iloc[tr_idcs], data_train_y[va_idcs]

        print("\nStart LGBM for fold {}".format(i))
        model = lightgbm.LGBMClassifier(**cfg.lgbm_cfg['params'])

        model.fit(x_tr, y_tr, eval_set=(x_val, y_val),
                  verbose=100, eval_metric='auc', early_stopping_rounds=150)

        print('AUC:', roc_auc_score(y_val, model.predict_proba(x_val)[:, 1]))
        y_preds = model.predict_proba(data_test)[:, 1] / N_FOLDS

    subm = pd.DataFrame({col_user_id: data_test_user_id_col,
                         col_y: y_preds})

    subm.to_csv('data/out/submission.csv', index=False)
    plot_feat_importances(model, data_train.columns)


def plot_feat_importances(_mdl, _cols):
    fea_imp = pd.DataFrame({'imp': _mdl.feature_importances_,
                            'col': _cols})
    fea_imp = fea_imp.sort_values(['imp', 'col'],
                                  ascending=[True, False]).iloc[-30:]
    _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))


if __name__ == '__main__':
    main()
