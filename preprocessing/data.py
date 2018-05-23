"""
Author: Kirgsn, 2017, https://www.kaggle.com/wkirgsn
"""
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler,\
    PolynomialFeatures
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.base import BaseEstimator, TransformerMixin

import preprocessing.config as cfg


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
        x_cols = []
        for col in df.columns:
            for p in cfg.data_cfg['Input_param_names']:
                if p in col:
                    x_cols.append(col)
                    break
            else:
                # col hasn't matched the pattern, check whitelist
                if col in self.white_list:
                    x_cols.append(col)
        self.x_cols = x_cols


class DataManager:

    PROFILE_ID_COL = 'profile_id'
    START_OF_PROFILE_COL = 'p_start'

    def __init__(self, path, create_hold_out=True):
        # original data
        self.dataset = pd.read_csv(path, dtype=np.float32)
        # When using CV, do not create a hold out
        self.has_hold_out = create_hold_out

        # downsample
        #self.dataset = self.dataset.iloc[::2, :]
        # drop profiles
        """drop_p = ['11', ]
        self.dataset.drop(axis=1, inplace=True, index=self.dataset[self.dataset[
            self.PROFILE_ID_COL].isin(drop_p)].index)"""
        # feature engineered dataset
        self.df = self.dataset.copy()

        # column management
        self.cl = ColumnManager(self.df)
        self.cl.x_cols = cfg.data_cfg['Input_param_names']

        # build pipeline building blocks
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

        self.pipe = Pipeline([
            ('feat_engineer', featurize_pipe),
            ('cleaning', DFCleaner()),
            ('scaler', scaling_pipe),
            ('poly', poly_pipe),
            ('ident', None)
        ])

    @property
    def tra_df(self):
        testsets = cfg.data_cfg['testset']
        valsets = cfg.data_cfg['valset']
        profiles_to_exclude = \
            testsets + valsets if self.has_hold_out else testsets
        sub_df = \
            self.df[~self.df[self.PROFILE_ID_COL].isin(profiles_to_exclude)]
        sub_df.reset_index(drop=True, inplace=True)
        self.cl.update(sub_df)
        return sub_df

    @property
    def val_df(self):
        sub_df = \
            self.df[self.df[self.PROFILE_ID_COL].isin(cfg.data_cfg['valset'])]
        sub_df.reset_index(drop=True, inplace=True)
        return sub_df

    @property
    def tst_df(self):
        sub_df = self.df[self.df[self.PROFILE_ID_COL].isin(cfg.data_cfg[
                                                             'testset'])]
        sub_df.reset_index(drop=True, inplace=True)
        return sub_df

    @property
    def actual(self):
        sub_df = self.dataset[self.dataset[self.PROFILE_ID_COL].isin(
            cfg.data_cfg[
                                                            'testset'])]
        return sub_df[self.cl.y_cols].reset_index(drop=True)

    @measure_time
    def get_featurized_sets(self):
        print('build dataset..')
        tra_df = self.tra_df
        tst_df = self.tst_df
        val_df = self.val_df

        tra_df = self.pipe.fit_transform(tra_df)
        tst_df = self.pipe.transform(tst_df)
        if val_df is not None:
            val_df = self.pipe.transform(val_df)

        self.cl.update(tra_df)
        return tra_df, val_df, tst_df

    def inverse_prediction(self, pred):
        simple_transformer = {k: v for k, v in
                              self.pipe
                                  .named_steps['feat_engineer']
                                  .named_steps['union']
                                  .transformer_list}['simple_trans_y']

        scaler = {k: v for k, v in
                  self.pipe
                      .named_steps['scaler']
                      .named_steps['union']
                      .transformer_list}['scaler_y']

        reduced_pipe = make_pipeline(simple_transformer, scaler)

        inversed = pd.DataFrame(reduced_pipe.inverse_transform(pred),
                                columns=self.cl.y_cols)
        return inversed

    def plot(self):
        from pandas.plotting import autocorrelation_plot
        import matplotlib.pyplot as plt
        self.df[[c for c in self.x_cols if 'rolling' in c] + self.y_cols]\
            .plot(subplots=True, sharex=True)
        plt.show()

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
        columns squared"""
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


