import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
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
N_FOLDS = 5
np.random.seed(SEED)

lbl_user_id = 'SK_ID_CURR'
lbl_y = 'TARGET'


def main():
    n_rows_to_read = \
        cfg.debug_cfg['n_debug'] if cfg.debug_cfg['DEBUG'] else None

    print('load data..')
    application_train = pd.read_csv("data/raw/application_train.csv",
                                    nrows=n_rows_to_read)
    application_test = pd.read_csv("data/raw/application_test.csv",
                                   nrows=n_rows_to_read)
    POS_CASH = pd.read_csv('data/raw/POS_CASH_balance.csv',
                           nrows=n_rows_to_read)
    credit_card = pd.read_csv('data/raw/credit_card_balance.csv',
                              nrows=n_rows_to_read)
    bureau = pd.read_csv('data/raw/bureau.csv', nrows=n_rows_to_read)
    previous_app = pd.read_csv('data/raw/previous_application.csv',
                               nrows=n_rows_to_read)

    print("Converting...")
    le = LabelEncoder()
    cat_feat = 'NAME_CONTRACT_STATUS'
    POS_CASH[cat_feat] = \
        le.fit_transform(POS_CASH[cat_feat].astype(str))
    nunique_status = \
        POS_CASH[[lbl_user_id, cat_feat]].groupby(lbl_user_id)\
            .nunique()[[cat_feat]]\
            .rename(columns={cat_feat: 'NUNIQUE_STATUS_POS_CASH'})

    nunique_status.reset_index(inplace=True)
    POS_CASH = POS_CASH.merge(nunique_status, how='left', on=lbl_user_id)
    POS_CASH.drop(['SK_ID_PREV', cat_feat], axis=1, inplace=True)

    credit_card[cat_feat] = \
        le.fit_transform(credit_card[cat_feat].astype(str))
    nunique_status = \
        credit_card[[lbl_user_id, cat_feat]]\
            .groupby(lbl_user_id).nunique()[[cat_feat]]\
            .rename(columns={cat_feat: 'NUNIQUE_STATUS_CREDIT_CARD'})
    nunique_status.reset_index(inplace=True)
    credit_card = credit_card.merge(nunique_status, how='left', on=lbl_user_id)
    credit_card.drop(['SK_ID_PREV', cat_feat], axis=1, inplace=True)

    bureau_cat_features = [f for f in bureau.columns if bureau[f].dtype == 'object']
    for f in bureau_cat_features:
        bureau[f] = le.fit_transform(bureau[f].astype(str))
        nunique = bureau[[lbl_user_id, f]].groupby(lbl_user_id)\
            .nunique()[[f]]\
            .rename(columns={f: 'NUNIQUE_'+f})
        nunique.reset_index(inplace=True)
        bureau = bureau.merge(nunique, how='left', on=lbl_user_id)
        bureau.drop([f], axis=1, inplace=True)  # todo: why is this dropped?
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)

    previous_app_cat_features = [f for f in previous_app.columns if
                                 previous_app[f].dtype == 'object']
    for f in previous_app_cat_features:
        previous_app[f] = le.fit_transform(previous_app[f].astype(str))
        nunique = previous_app[[lbl_user_id, f]].groupby(lbl_user_id)\
            .nunique()[[f]]\
            .rename(columns={f: 'NUNIQUE_'+f})
        nunique.reset_index(inplace=True)
        previous_app = previous_app.merge(nunique, how='left', on=lbl_user_id)
        previous_app.drop([f], axis=1, inplace=True)
    previous_app.drop(['SK_ID_PREV'], axis=1, inplace=True)

    print("Merging...")
    # calc means of all features per user id
    pos_cash_mean_per_id = POS_CASH.groupby(lbl_user_id).mean().reset_index()
    credit_card_mean_per_id = credit_card.groupby(lbl_user_id).mean().reset_index()
    bureau_mean_per_id = bureau.groupby(lbl_user_id).mean().reset_index()
    previous_app_mean_per_id = previous_app.groupby(lbl_user_id).mean().reset_index()

    # merge to dataset
    data_train = \
        application_train.merge(pos_cash_mean_per_id, how='left', on=lbl_user_id)
    data_test = \
        application_test.merge(pos_cash_mean_per_id, how='left', on=lbl_user_id)

    data_train = \
        data_train.merge(credit_card_mean_per_id, how='left', on=lbl_user_id)
    data_test = \
        data_test.merge(credit_card_mean_per_id, how='left', on=lbl_user_id)

    data_train = \
        data_train.merge(bureau_mean_per_id, how='left', on=lbl_user_id)
    data_test = \
        data_test.merge(bureau_mean_per_id, how='left', on=lbl_user_id)

    data_train = \
        data_train.merge(previous_app_mean_per_id, how='left', on=lbl_user_id)
    data_test = \
        data_test.merge(previous_app_mean_per_id, how='left', on=lbl_user_id)

    data_train_y = data_train[lbl_y]
    data_train.drop([lbl_user_id, lbl_y], axis=1, inplace=True)
    data_test_user_id_col = data_test.pop(lbl_user_id)

    cat_features = [f for f in data_train.columns if data_train[f].dtype == 'object']
    print("Cat features are: %s" % [f for f in cat_features])

    for col in cat_features:
        data_train[col] = le.fit_transform(data_train[col].astype(str))
        data_test[col] = le.fit_transform(data_test[col].astype(str))

    # NA value handling
    data_train.fillna(-1, inplace=True)
    data_test.fillna(-1, inplace=True)

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

    subm = pd.DataFrame({lbl_user_id: data_test_user_id_col,
                         lbl_y: y_preds})

    subm.to_csv('data/out/submission.csv', index=False)


def plot_feat_importances(_mdl, _cols):
    fea_imp = pd.DataFrame({'imp': _mdl.feature_importances_,
                            'col': _cols})
    fea_imp = fea_imp.sort_values(['imp', 'col'],
                                  ascending=[True, False]).iloc[-30:]
    _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))


if __name__ == '__main__':
    main()
