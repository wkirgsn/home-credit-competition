"""
Author: Kirgsn, 2018, https://www.kaggle.com/wkirgsn
"""
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler,\
    PolynomialFeatures, LabelEncoder
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.base import BaseEstimator, TransformerMixin

import preprocessing.config as cfg

"""
CAUTION!
THIS FILE IS UNDER HEAVY CONSTRUCTION!
"""

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
        self.previous_app = pd.read_csv('data/raw/previous_application.csv',
                                   nrows=n_rows_to_read)

    @measure_time
    def build_pipeline(self):
        print('build pipeline...')
        featurize_union = FeatureUnion([('simple_trans_y',
                                         SimpleTransformer(np.sqrt,
                                                           np.square,
                                                           self.cl.y_cols
                                                           )),
                                        ('identity_x', Router(self.cl.x_cols)),
                                        ('lag_feats_x',
                                         LagFeatures(self.cl.x_cols)),
                                        ('rolling_feats_x',
                                         RollingFeatures(self.cl.x_cols,
                                                         lookback=100)),
                                        ('start_of_profile',
                                         SimpleTransformer(
                                             self.indicate_start_of_profile,
                                             None, [self.PROFILE_ID_COL])),
                                        ('u_q_sqrd+i_q_sqrd',
                                         SimpleTransformer(
                                             self.sum_of_squares,
                                             None, ['i_q', 'u_q'])
                                         ),
                                        ('u_d_sqrd+i_d_sqrd',
                                         SimpleTransformer(
                                             self.sum_of_squares,
                                             None, ['i_d', 'u_d'])
                                         )
                                        ])

        featurize_pipe = FeatureUnionReframer.make_df_retaining(featurize_union)

        col_router_pstart = Router([self.START_OF_PROFILE_COL])
        col_router_y = Router(self.cl.y_cols)
        scaling_union = FeatureUnion([('scaler_x', Scaler(StandardScaler(),
                                                          self.cl,
                                                          select='x')),
                                      ('scaler_y', Scaler(StandardScaler(),
                                                          self.cl, select='y')),

                                      ('start_of_profile', col_router_pstart)
                                      ])
        scaling_pipe = FeatureUnionReframer.make_df_retaining(scaling_union)

        poly_union = make_union(Polynomials(degree=2, colmanager=self.cl),
                                col_router_pstart, col_router_y)
        poly_pipe = FeatureUnionReframer.make_df_retaining(poly_union)

        return Pipeline([
                    ('feat_engineer', featurize_pipe),
                    ('cleaning', DFCleaner()),
                    ('scaler', scaling_pipe),
                    ('poly', poly_pipe),
                    ('ident', None)
                ])

    @measure_time
    def factorize_categoricals(self):
        print('factorize categoricals..')
        le = LabelEncoder()
        cat_feat = 'NAME_CONTRACT_STATUS'
        self.POS_CASH[cat_feat] = \
            le.fit_transform(self.POS_CASH[cat_feat].astype(str))
        nunique_status = \
            self.POS_CASH[[col_user_id, cat_feat]].groupby(col_user_id) \
                .nunique()[[cat_feat]] \
                .rename(columns={cat_feat: 'NUNIQUE_STATUS_POS_CASH'})

        nunique_status.reset_index(inplace=True)
        self.POS_CASH = self.POS_CASH.merge(nunique_status, how='left',
                                      on=col_user_id)
        self.POS_CASH.drop(['SK_ID_PREV', cat_feat], axis=1, inplace=True)

        self.credit_card[cat_feat] = \
            le.fit_transform(self.credit_card[cat_feat].astype(str))
        nunique_status = \
            self.credit_card[[col_user_id, cat_feat]] \
                .groupby(col_user_id).nunique()[[cat_feat]] \
                .rename(columns={cat_feat: 'NUNIQUE_STATUS_CREDIT_CARD'})
        nunique_status.reset_index(inplace=True)
        self.credit_card = self.credit_card.merge(nunique_status, how='left',
                                        on=col_user_id)
        self.credit_card.drop(['SK_ID_PREV', cat_feat], axis=1, inplace=True)

        bureau_cat_features = [f for f in self.bureau.columns if
                               self.bureau[f].dtype == 'object']
        for f in bureau_cat_features:
            self.bureau[f] = le.fit_transform(self.bureau[f].astype(str))
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
            self.previous_app[f] = le.fit_transform(self.previous_app[f].astype(str))
            nunique = self.previous_app[[col_user_id, f]].groupby(col_user_id) \
                .nunique()[[f]] \
                .rename(columns={f: 'NUNIQUE_' + f})
            nunique.reset_index(inplace=True)
            self.previous_app = self.previous_app.merge(nunique, how='left',
                                              on=col_user_id)
            self.previous_app.drop([f], axis=1, inplace=True)
        self.previous_app.drop(['SK_ID_PREV'], axis=1, inplace=True)

        main_set_cat_features = [f for f in self.application_train.columns if
                                 self.application_train[f].dtype == 'object']
        for col in main_set_cat_features:
            self.application_train[col], _ = \
                self.application_train[col].astype(str).factorize()
            self.application_test[col], _ = \
                self.application_test[col].astype(str).factorize()

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
        train_set = \
            self.application_train.merge(pos_cash_mean_per_id, how='left',
                                    on=col_user_id)
        test_set = \
            self.application_test.merge(pos_cash_mean_per_id, how='left',
                                   on=col_user_id)

        train_set = \
            train_set.merge(credit_card_mean_per_id, how='left', on=col_user_id)
        test_set = \
            test_set.merge(credit_card_mean_per_id, how='left', on=col_user_id)

        train_set = \
            train_set.merge(bureau_mean_per_id, how='left', on=col_user_id)
        test_set = \
            test_set.merge(bureau_mean_per_id, how='left', on=col_user_id)

        train_set = \
            train_set.merge(previous_app_mean_per_id, how='left', on=col_user_id)
        test_set = \
            test_set.merge(previous_app_mean_per_id, how='left', on=col_user_id)
        return train_set, test_set

    @staticmethod
    def handle_na(_train, _test):
        return _train.fillna(-1), _test.fillna(-1)

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


class SimpleTransformer(BaseEstimator, TransformerMixin):
    """Apply given transformation."""
    def __init__(self, trans_func, untrans_func, columns):
        self.transform_func = trans_func
        self.inverse_transform_func = untrans_func
        self.cols = columns
        self.out_cols = []

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x = self._get_selection(x)
        ret = self.transform_func(x) if callable(self.transform_func) else x
        self.out_cols = list(ret.columns)
        return ret

    def inverse_transform(self, x):
        return self.inverse_transform_func(x) \
            if callable(self.inverse_transform_func) else x

    def _get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        assert set(self.cols).issubset(set(df.columns)),\
            '{} is not in {}'.format(self.cols, df.columns)
        return df[self.cols]

    def get_feature_names(self):
        return self.out_cols


class Router(SimpleTransformer):
    """SimpleTransformer with transformation functions being None"""
    def __init__(self, columns):
        super().__init__(None, None, columns=columns)


class LagFeatures(BaseEstimator, TransformerMixin):
    """This Transformer adds arithmetic variations between current and lag_x
    observation"""
    def __init__(self, columns, lookback=1):
        self.lookback = lookback
        self.cols = columns
        self.transformed_cols = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self._get_selection(X)
        dfs = []
        for lback in range(1, self.lookback + 1):
            lag_feats = {'lag{}': X.shift(lback),
                         'lag{}_diff': X.diff(periods=lback),
                         }
            lag_feats['lag{}_abs'] = abs(lag_feats['lag{}_diff'])
            lag_feats['lag{}_sum'] = X + lag_feats['lag{}']

            lag_feats = {key.format(lback): value for key, value
                         in lag_feats.items()}
            # update columns
            for k in lag_feats:
                lag_feats[k].columns = ['{}_{}'.format(c, k) for c in
                                        X.columns]

            dfs.append(pd.concat(list(lag_feats.values()), axis=1))
        df = pd.concat(dfs, axis=1)
        self.transformed_cols = list(df.columns)
        return df

    def _get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        return df[self.cols]

    def get_feature_names(self):
        return self.transformed_cols


class RollingFeatures(BaseEstimator, TransformerMixin):
    """This Transformer adds rolling statistics"""
    def __init__(self, columns, lookback=10):
        self.lookback = lookback
        self.cols = columns
        self.transformed_cols = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self._get_selection(X)
        feat_d = {'std': X.rolling(self.lookback).std(),
                  'mean': X.rolling(self.lookback).mean(),
                  # 'sum': X.rolling(self.lookback).sum()
                  }
        for k in feat_d:
            feat_d[k].columns = \
                ['{}_rolling{}_{}'.format(c, self.lookback, k) for
                 c in X.columns]
        df = pd.concat(list(feat_d.values()), axis=1)
        self.transformed_cols = list(df.columns)
        return df

    def _get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        return df[self.cols]

    def get_feature_names(self):
        return self.transformed_cols


class ReSamplerForBatchTraining(BaseEstimator, TransformerMixin):
    """This transformer sorts the samples according to a
    batch size for batch training"""
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.indices, self.columns = [], []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        # cut the tail
        trunc_idx = len(X) % self.batch_size
        if trunc_idx > 0:
            X = X.iloc[:-trunc_idx, :]

        # reorder
        new_idcs = np.tile(np.arange(self.batch_size), len(X) //
                           self.batch_size)
        assert len(X) == new_idcs.shape[0], \
            "{} != {}".format(len(X), new_idcs.shape[0])
        X.loc[:, 'new_idx'] = new_idcs
        X.sort_values(by='new_idx', ascending=True, inplace=True)
        self.indices = X.index
        X.reset_index(drop=True, inplace=True)
        X.drop(['new_idx'], axis=1, inplace=True)
        self.columns = X.columns
        return X

    def inverse_transform(self, X):
        # columns undefined
        inversed = pd.DataFrame(X, index=self.indices).sort_index()
        return inversed


class Polynomials(BaseEstimator, TransformerMixin):

    def __init__(self, degree, colmanager):
        self.poly = PolynomialFeatures(degree=degree)
        self.out_cols = []
        self.cl = colmanager

    def fit(self, X, y=None):
        X = self._get_selection(X)
        assert isinstance(X, pd.DataFrame)
        self.poly.fit(X, y)
        self.out_cols = self.poly.get_feature_names(input_features=X.columns)
        return self

    def transform(self, X):
        """This transform shall only take Input that has the same columns as
        those this transformer had during fit"""
        X = self._get_selection(X, update=False)
        assert isinstance(X, pd.DataFrame)
        X = self.poly.transform(X)
        ret = pd.DataFrame(X, columns=self.out_cols)
        return ret

    def _get_selection(self, df, update=True):
        assert isinstance(df, pd.DataFrame)
        if update:
            self.cl.update(df)
        return df[self.cl.x_cols]

    def get_feature_names(self):
        return self.out_cols


class DFCleaner(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X.dropna(inplace=True)
        X.reset_index(drop=True, inplace=True)
        return X


class IdentityEstimator(BaseEstimator, TransformerMixin):
    """This class is for replacing a basic identity estimator with one that
    returns the full input pandas DataFrame instead of a numpy arr
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class Scaler(BaseEstimator, TransformerMixin):
    """scales selected columns only with given scaler.
    Parameter 'select' is either 'x' or 'y' """
    def __init__(self, scaler, column_manager, select='x'):
        self.cl = column_manager
        self.scaler = scaler
        self.select = select
        self.cols = []

    def fit(self, X, y=None):
        X = self.get_selection(X)
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X = X[self.cols]
        return self.scaler.transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

    def get_selection(self, df):
        assert isinstance(df, pd.DataFrame)
        if self.select.lower() == 'x':
            self.cl.update(df)
            self.cols = self.cl.x_cols
        elif self.select.lower() == 'y':
            self.cols = self.cl.y_cols
        else:
            raise NotImplementedError()
        return df[self.cols]

    def get_feature_names(self):
        return self.cols


class FeatureUnionReframer(BaseEstimator, TransformerMixin):
    """Transforms preceding FeatureUnion's output back into Dataframe"""
    def __init__(self, feat_union, cutoff_transformer_name=True):
        self.union = feat_union
        self.cutoff_transformer_name = cutoff_transformer_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, np.ndarray)
        if self.cutoff_transformer_name:
            # todo Warum haben wir sechs Spalten mehr?
            cols = [c.split('__')[1] for c in self.union.get_feature_names()]
        else:
            cols = self.union.get_feature_names()
        df = pd.DataFrame(data=X, columns=cols)
        return df

    @classmethod
    def make_df_retaining(cls, feature_union):
        """With this method a feature union will be returned as a pipeline
        where the first step is the union and the second is a transformer that
        re-applies the columns to the union's output"""
        return Pipeline([('union', feature_union),
                         ('reframe', cls(feature_union))])


