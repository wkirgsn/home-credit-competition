"""
Author: Kirgsn, 2018, https://www.kaggle.com/wkirgsn
"""
import time
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

import preprocessing.config as cfg


col_user_id = 'SK_ID_CURR'
col_y = 'TARGET'


def measure_time(func):
    """time measuring decorator"""
    def wrapped(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        print('took {:.3} seconds'.format(end_time-start_time))
        return ret
    return wrapped


class ColumnManager:
    """Class to keep track of the current input columns and for general
    economy of features"""
    def __init__(self, df, white_list=[]):
        self.original = list(df.columns)
        self._x, self._y = None, cfg.data_cfg['Target_param_names']
        self.white_list = white_list
        self.update(df)

    @property
    def x_cols(self):
        return self._x

    @x_cols.setter
    def x_cols(self, value):
        self._x = value

    @property
    def y_cols(self):
        return self._y

    def update(self, df):
        self.x_cols = list(df.columns)


class DataManager:

    @measure_time
    def __init__(self):
        print('load data...')
        n_rows_to_read = \
            cfg.debug_cfg['n_debug'] if cfg.debug_cfg['DEBUG'] else None

        # original data
        self.application_train = pd.read_csv("data/raw/application_train.csv",
                                             nrows=n_rows_to_read)
        self.application_test = pd.read_csv("data/raw/application_test.csv",
                                            nrows=n_rows_to_read)
        self.POS_CASH = pd.read_csv('data/raw/POS_CASH_balance.csv',
                                    nrows=n_rows_to_read)
        self.credit_card = pd.read_csv('data/raw/credit_card_balance.csv',
                                       nrows=n_rows_to_read)
        self.bureau = pd.read_csv('data/raw/bureau.csv', nrows=n_rows_to_read)
        self.bureau_bal = pd.read_csv('data/raw/bureau_balance.csv',
                                      nrows=n_rows_to_read)
        self.previous_app = pd.read_csv('data/raw/previous_application.csv',
                                        nrows=n_rows_to_read)
        self.installments = pd.read_csv('data/raw/installments_payments.csv',
                                        nrows=n_rows_to_read)

    @measure_time
    def factorize_categoricals(self):
        print('factorize categoricals..')

        def _find_cats_and_factorize(_df, _prll, testset=None):
            cat_features = _df.loc[:, _df.dtypes == object].columns.tolist()
            ret_list = _prll(delayed(self._prll_factorize)
                             (_df.loc[:, col]) for col in cat_features)
            for s, idx, col in ret_list:
                _df[col] = s
                if testset is not None:
                    testset[col] = idx.get_indexer(testset.loc[:, col])
            return _df, testset

        with Parallel(n_jobs=2) as prll:

            self.application_train, self.application_test = \
                _find_cats_and_factorize(self.application_train.copy(), prll,
                                         self.application_test.copy())

            self.POS_CASH, _ = _find_cats_and_factorize(self.POS_CASH, prll)
            self.credit_card, _ = _find_cats_and_factorize(self.credit_card,
                                                           prll)
            self.bureau, _ = _find_cats_and_factorize(self.bureau, prll)
            self.bureau_bal, _ = _find_cats_and_factorize(self.bureau_bal, prll)
            self.previous_app, _ = _find_cats_and_factorize(self.previous_app,
                                                            prll)
            self.installments, _ = _find_cats_and_factorize(self.installments,
                                                            prll)

    def get_special_features(self):
        # todo: add features!!!
        cat_feat = 'NAME_CONTRACT_STATUS'

        nunique_contract_status = (
            self.POS_CASH[[col_user_id, cat_feat]].groupby(col_user_id)
                .nunique()[[cat_feat]]
                .rename(index=str,
                        columns={cat_feat: 'NUNIQUE_STATUS_POS_CASH'})
                .reset_index())

        count_contract_status = (
            self.POS_CASH[[col_user_id, cat_feat]]
                .groupby(col_user_id)
                .count()[[cat_feat]]
                .rename(index=str, columns={cat_feat: 'COUNT_STATUS_POS_CASH'})
                .reset_index()
        )

        self.POS_CASH = (
            self.POS_CASH.merge(nunique_contract_status,
                                how='left', on=col_user_id)
                         .merge(count_contract_status,
                                how='left', on=col_user_id)
        )

        self.POS_CASH.drop(['SK_ID_PREV', cat_feat], axis=1, inplace=True)

        # todo: more more more, see references!
        nunique_status = (
            self.credit_card[[col_user_id, cat_feat]]
                .groupby(col_user_id).nunique()[[cat_feat]]
                .rename(columns={cat_feat: 'NUNIQUE_STATUS_CREDIT_CARD'}))
        nunique_status.reset_index(inplace=True)
        self.credit_card = self.credit_card.merge(nunique_status, how='left',
                                        on=col_user_id)
        self.credit_card.drop(['SK_ID_PREV', cat_feat], axis=1, inplace=True)

        bureau_cat_features = [f for f in self.bureau.columns if
                               self.bureau[f].dtype == 'object']
        for f in bureau_cat_features:
            nunique = self.bureau[[col_user_id, f]].groupby(col_user_id) \
                .nunique()[[f]] \
                .rename(columns={f: 'NUNIQUE_' + f})
            nunique.reset_index(inplace=True)
            self.bureau = self.bureau.merge(nunique, how='left', on=col_user_id)
            self.bureau.drop([f], axis=1, inplace=True)  # todo: why is this dropped?
        self.bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)

        previous_app_cat_features = [f for f in self.previous_app.columns if
                                     self.previous_app[f].dtype == 'object']
        for f in previous_app_cat_features:
            nunique = self.previous_app[[col_user_id, f]].groupby(col_user_id) \
                .nunique()[[f]] \
                .rename(columns={f: 'NUNIQUE_' + f})
            nunique.reset_index(inplace=True)
            self.previous_app = self.previous_app.merge(nunique, how='left',
                                              on=col_user_id)
            self.previous_app.drop([f], axis=1, inplace=True)
        self.previous_app.drop(['SK_ID_PREV'], axis=1, inplace=True)

    @staticmethod
    def _prll_factorize(_s):
        """Parallel factorization"""
        series, indexer = _s.factorize()
        return series, indexer, _s.name

    @measure_time
    def merge_tables(self):
        print("Merging tables...")
        # calc means of all features per user id
        pos_cash_mean_per_id = self.POS_CASH.groupby(
            col_user_id).mean().reset_index()
        credit_card_mean_per_id = self.credit_card.groupby(
            col_user_id).mean().reset_index()
        bureau_mean_per_id = self.bureau.groupby(col_user_id).mean().reset_index()
        previous_app_mean_per_id = self.previous_app.groupby(
            col_user_id).mean().reset_index()

        # merge to dataset
        train_set = (
            self.application_train
                .merge(pos_cash_mean_per_id, how='left', on=col_user_id)
                .merge(credit_card_mean_per_id, how='left', on=col_user_id)
                .merge(bureau_mean_per_id, how='left', on=col_user_id)
                .merge(previous_app_mean_per_id, how='left', on=col_user_id)
        )
        test_set = (
            self.application_test
                .merge(pos_cash_mean_per_id, how='left', on=col_user_id)
                .merge(credit_card_mean_per_id, how='left', on=col_user_id)
                .merge(bureau_mean_per_id, how='left', on=col_user_id)
                .merge(previous_app_mean_per_id, how='left', on=col_user_id)
        )

        return train_set, test_set

    def handle_na(self):
        na_value = -99
        self.application_train.fillna(na_value)
        self.application_test.fillna(na_value)
        self.POS_CASH.fillna(na_value)
        self.credit_card.fillna(na_value)
        self.installments.fillna(na_value)
        self.previous_app.fillna(na_value)
        self.bureau.fillna(na_value)
        self.bureau_bal.fillna(na_value)

    def inverse_prediction(self, pred):
        pass

    def plot(self):
        pass

    def indicate_start_of_profile(self, s):
        """Returns a DataFrame where the first observation of each new profile
        id is indicated with True."""
        assert isinstance(s, pd.DataFrame)
        assert s.columns == self.PROFILE_ID_COL
        return pd.DataFrame(data=~s.duplicated(),
                            columns=[self.START_OF_PROFILE_COL])

    @staticmethod
    def sum_of_squares(df):
        """ Return a DataFrame with a single column that is the sum of all
        columns of the given dataframe squared"""
        assert isinstance(df, pd.DataFrame)
        colnames = ["{}^2".format(c) for c in list(df.columns)]
        return pd.DataFrame(data=np.square(df).sum(axis=1),
                            columns=["+".join(colnames)])





