"""
Author: WKirgsn, 2018, https://www.kaggle.com/wkirgsn
"""
import time
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

import preprocessing.config as cfg
from preprocessing.reducing import Reducer

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


class DataManager:

    @measure_time
    def __init__(self):
        print('load data...')

        # Number of rows to pull in.
        # DEBUG = 10000 rows
        # Else  = all rows
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
        self.cat_encode = {}
        self.nan_table = {}

        if not cfg.debug_cfg['DEBUG']:
            reducer = Reducer()
            self.application_train = reducer.reduce(self.application_train)
            self.application_test = reducer.reduce(self.application_test)
            self.POS_CASH = reducer.reduce(self.POS_CASH)
            self.credit_card = reducer.reduce(self.credit_card)
            self.bureau = reducer.reduce(self.bureau)
            self.bureau_bal = reducer.reduce(self.bureau_bal)
            self.previous_app = reducer.reduce(self.previous_app)
            self.installments = reducer.reduce(self.installments)

    @staticmethod
    def _prll_factorize(_s):
        """Parallel factorization"""
        series, indexer = _s.factorize()
        return series, indexer, _s.name

    @measure_time
    def factorize_categoricals(self):
        print('factorize categoricals..')

        def _find_cats_and_factorize(_df, _prll):
            # todo: Are there NaNs in categorical features??
            cat_features = _df.loc[:, _df.dtypes == object].columns.tolist()
            ret_list = _prll(delayed(self._prll_factorize)
                             (_df.loc[:, col]) for col in cat_features)
            indexers = {}
            for s, idx, col in ret_list:
                _df[col] = s
                indexers[col] = idx
            return _df, indexers

        with Parallel(n_jobs=2) as prll:

            self.application_train, self.cat_encode['app_main'] = \
                _find_cats_and_factorize(self.application_train.copy(), prll)
            for col, indexer in self.cat_encode['app_main'].items():
                self.application_test[col] = \
                    indexer.get_indexer(self.application_test.loc[:, col])

            self.POS_CASH, self.cat_encode['pos_cash'] = \
                _find_cats_and_factorize(self.POS_CASH, prll)
            self.credit_card, self.cat_encode['credit_card'] = \
                _find_cats_and_factorize(self.credit_card, prll)
            self.bureau, self.cat_encode['bureau'] = \
                _find_cats_and_factorize(self.bureau, prll)
            self.bureau_bal, self.cat_encode['bureau_bal'] = \
                _find_cats_and_factorize(self.bureau_bal, prll)
            self.previous_app, self.cat_encode['prev_app'] = \
                _find_cats_and_factorize(self.previous_app, prll)
            self.installments, self.cat_encode['installments'] = \
                _find_cats_and_factorize(self.installments, prll)

    @measure_time
    def get_special_features(self):
        """Heavy Feature Engineering"""

        print('Engineer features...')

        merge_cfg = {'how': 'left',
                     'on': col_user_id}
        # todo: (LKI) add features!!!
        # POS CASH
        more_on_contract_status = (
            self.POS_CASH[[col_user_id, *self.cat_encode['pos_cash'].keys()]]
                .groupby(col_user_id)
                .agg(['count', 'nunique'])
                .reset_index()
                )
        # flatten multiindex
        more_on_contract_status.columns = \
            ['_'.join(col).strip('_') for col in
             more_on_contract_status.columns.values]

        self.POS_CASH = self.POS_CASH.merge(more_on_contract_status,
                                            **merge_cfg)

        self.POS_CASH.drop(['SK_ID_PREV',
                            *self.cat_encode['pos_cash'].keys()],
                           axis=1, inplace=True)

        # CREDIT CARD
        # todo: (LKI) more more more, see references!
        more_on_contract_status = (
            self.credit_card[[col_user_id,
                              *self.cat_encode['credit_card'].keys()]]
                .groupby(col_user_id)
                .agg(['nunique', 'count'])
                .reset_index())
        # flatten multiindex
        more_on_contract_status.columns = \
            ['_'.join(col).strip('_') for col in
             more_on_contract_status.columns.values]
        self.credit_card = self.credit_card.merge(more_on_contract_status,
                                                  **merge_cfg)
        self.credit_card.drop(['SK_ID_PREV',
                               *self.cat_encode['credit_card'].keys()],
                              axis=1, inplace=True)

        # BUREAU
        """Source: https://www.kaggle.com/shanth84/
        home-credit-bureau-data-feature-engineering/notebook
        """
        n_unique_categoricals = (
            self.bureau[[col_user_id, *self.cat_encode['bureau'].keys()]]
                .groupby(col_user_id)
                .agg(['nunique'])
                .reset_index()
        )
        n_unique_categoricals.columns = \
            ['_'.join(col).strip('_') for col in
             n_unique_categoricals.columns.values]

        n_entries = (
            self.bureau[[col_user_id,
                         'CREDIT_ACTIVE']]  # could have been any col
                .groupby(col_user_id)
                .count()
                .reset_index()
                .rename(columns={'CREDIT_ACTIVE': 'entries_in'})
        )

        amt_active_credits = (
            self.bureau.loc[
                self.bureau.CREDIT_ACTIVE ==
                self.cat_encode['bureau']['CREDIT_ACTIVE'].get_loc('Active'),
                [col_user_id, 'CREDIT_ACTIVE']]
                .groupby(col_user_id)
                .count()
                .reset_index()
                .rename(columns={'CREDIT_ACTIVE': 'amt_active_credits'})
        )

        self.bureau = (
            self.bureau
                .merge(n_unique_categoricals, **merge_cfg)
                .merge(n_entries, **merge_cfg)
                .merge(amt_active_credits, **merge_cfg)
        )
        self.bureau.drop(['SK_ID_BUREAU', *self.cat_encode['bureau'].keys()],
                         axis=1, inplace=True)

        # PREV APP
        previous_app_cat_features = [f for f in self.previous_app.columns if
                                     self.previous_app[f].dtype == 'object']
        for f in previous_app_cat_features:
            nunique = self.previous_app[[col_user_id, f]].groupby(col_user_id) \
                .nunique()[[f]] \
                .rename(columns={f: 'NUNIQUE_' + f})
            nunique.reset_index(inplace=True)
            self.previous_app = self.previous_app.merge(nunique, **merge_cfg)
            self.previous_app.drop([f], axis=1, inplace=True)
        self.previous_app.drop(['SK_ID_PREV'], axis=1, inplace=True)

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
                .merge(pos_cash_mean_per_id, how='left', on=col_user_id,
                       suffixes=('', '_POS_CASH'))
                .merge(credit_card_mean_per_id, how='left', on=col_user_id,
                       suffixes=('', '_CC'))
                .merge(bureau_mean_per_id, how='left', on=col_user_id,
                       suffixes=('', '_BUREAU'))
                .merge(previous_app_mean_per_id, how='left', on=col_user_id,
                       suffixes=('', '_PREV_APP'))
        )
        test_set = (
            self.application_test
                .merge(pos_cash_mean_per_id, how='left', on=col_user_id,
                       suffixes=('', '_POS_CASH'))
                .merge(credit_card_mean_per_id, how='left', on=col_user_id,
                       suffixes=('', '_CC'))
                .merge(bureau_mean_per_id, how='left', on=col_user_id,
                       suffixes=('', '_BUREAU'))
                .merge(previous_app_mean_per_id, how='left', on=col_user_id,
                       suffixes=('', '_PREV_APP'))
        )

        return train_set, test_set

    def handle_na(self):
        na_value = -99
        self.application_train.fillna(na_value, inplace=True)
        self.application_test.fillna(na_value, inplace=True)
        self.POS_CASH.fillna(na_value, inplace=True)
        self.credit_card.fillna(na_value, inplace=True)
        self.installments.fillna(na_value, inplace=True)
        self.previous_app.fillna(na_value, inplace=True)

        self.nan_table['bureau'] = {'amt_active_credits': 0}

        self.bureau.fillna(self.nan_table['bureau'], inplace=True)
        self.bureau.fillna(na_value, inplace=True)
        self.bureau_bal.fillna(na_value, inplace=True)

    def inverse_prediction(self, pred):
        pass

    def plot(self):
        pass

    @staticmethod
    def sum_of_squares(df):
        """ Return a DataFrame with a single column that is the sum of all
        columns of the given dataframe squared"""
        assert isinstance(df, pd.DataFrame)
        colnames = ["{}^2".format(c) for c in list(df.columns)]
        return pd.DataFrame(data=np.square(df).sum(axis=1),
                            columns=["+".join(colnames)])





