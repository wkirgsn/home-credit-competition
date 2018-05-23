import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm


"""Sources: 
https://www.kaggle.com/yekenot/catboostarter/code
https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm/code
"""
warnings.filterwarnings("ignore")
SEED = 2018
np.random.seed(SEED)


def main():

    application_train = pd.read_csv("data/raw/application_train.csv")
    application_test = pd.read_csv("data/raw/application_test.csv")
    POS_CASH = pd.read_csv('data/raw/POS_CASH_balance.csv')
    credit_card = pd.read_csv('data/raw/credit_card_balance.csv')
    bureau = pd.read_csv('data/raw/bureau.csv')
    previous_app = pd.read_csv('data/raw/previous_application.csv')
    subm = pd.read_csv("data/raw/sample_submission.csv")

    print("Converting...")
    le = LabelEncoder()
    POS_CASH['NAME_CONTRACT_STATUS'] = \
        le.fit_transform(POS_CASH['NAME_CONTRACT_STATUS'].astype(str))
    nunique_status = \
        POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
    POS_CASH['NUNIQUE_STATUS'] = \
        nunique_status['NAME_CONTRACT_STATUS']
    POS_CASH.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

    credit_card['NAME_CONTRACT_STATUS'] = \
        le.fit_transform(credit_card['NAME_CONTRACT_STATUS'].astype(str))
    nunique_status = \
        credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']]\
            .groupby('SK_ID_CURR').nunique()
    credit_card['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
    credit_card.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

    bureau_cat_features = [f for f in bureau.columns if bureau[f].dtype == 'object']
    for f in bureau_cat_features:
        bureau[f] = le.fit_transform(bureau[f].astype(str))
        nunique = bureau[['SK_ID_CURR', f]].groupby('SK_ID_CURR').nunique()
        bureau['NUNIQUE_'+f] = nunique[f]
        bureau.drop([f], axis=1, inplace=True)
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)

    previous_app_cat_features = [f for f in previous_app.columns if previous_app[f].dtype == 'object']
    for f in previous_app_cat_features:
        previous_app[f] = le.fit_transform(previous_app[f].astype(str))
        nunique = previous_app[['SK_ID_CURR', f]].groupby('SK_ID_CURR').nunique()
        previous_app['NUNIQUE_'+f] = nunique[f]
        previous_app.drop([f], axis=1, inplace=True)
    previous_app.drop(['SK_ID_PREV'], axis=1, inplace=True)

    print("Merging...")
    data_train = \
        application_train.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(),
                                how='left', on='SK_ID_CURR')
    data_test = \
        application_test.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(),
                               how='left', on='SK_ID_CURR')

    data_train = \
        data_train.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(),
                         how='left', on='SK_ID_CURR')
    data_test = \
        data_test.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(),
                        how='left', on='SK_ID_CURR')

    data_train = \
        data_train.merge(bureau.groupby('SK_ID_CURR').mean().reset_index(),
                         how='left', on='SK_ID_CURR')
    data_test = \
        data_test.merge(bureau.groupby('SK_ID_CURR').mean().reset_index(),
                        how='left', on='SK_ID_CURR')

    data_train = \
        data_train.merge(previous_app.groupby('SK_ID_CURR').mean().reset_index(),
                         how='left', on='SK_ID_CURR')
    data_test = \
        data_test.merge(previous_app.groupby('SK_ID_CURR').mean().reset_index(),
                        how='left', on='SK_ID_CURR')

    target_train = data_train['TARGET']
    data_train.drop(['SK_ID_CURR', 'TARGET'], axis=1, inplace=True)
    data_test.drop(['SK_ID_CURR'], axis=1, inplace=True)

    cat_features = [f for f in data_train.columns if data_train[f].dtype == 'object']
    def column_index(df, query_cols):
        cols = df.columns.values
        sidx = np.argsort(cols)
        return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]
    cat_features_inds = column_index(data_train, cat_features)
    print("Cat features are: %s" % [f for f in cat_features])
    print(cat_features_inds)

    for col in cat_features:
        data_train[col] = le.fit_transform(data_train[col].astype(str))
        data_test[col] = le.fit_transform(data_test[col].astype(str))

    data_train.fillna(-1, inplace=True)
    data_test.fillna(-1, inplace=True)
    cols = data_train.columns

    X_train, X_valid, y_train, y_valid = train_test_split(data_train, target_train,
                                                          test_size=0.1,
                                                          random_state=SEED)
    print(X_train.shape)
    print(X_valid.shape)

    print("\nSetup LGBM...")
    lgbm_params = {
        'n_estimators':4000,
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
    }
    model = lightgbm.LGBMClassifier(**lgbm_params)

    model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
              verbose=100, eval_metric='auc', early_stopping_rounds=150)

    fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': cols})
    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
    #_ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))

    print('AUC:', roc_auc_score(y_valid, model.predict_proba(X_valid)[:,1]))
    y_preds = model.predict_proba(data_test)[:, 1]
    subm['TARGET'] = y_preds
    subm.to_csv('data/out/submission.csv', index=False)


if __name__=='__main__':
    main()