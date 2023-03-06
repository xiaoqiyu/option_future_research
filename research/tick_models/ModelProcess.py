#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/29 16:24
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : ModelProcess.py


import math
import time
import uqer
import pprint
import numpy as np
from uqer import DataAPI
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.decomposition import PCA
from scipy.stats import rankdata
import talib as ta
import pandas as pd
import statsmodels.api as sm
from editorconfig import get_properties, EditorConfigError
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.svm import SVR
import logging
import os
import gc
from copy import deepcopy

import utils.define as define
import utils.utils as utils
import research.tick_models.FactorProcess as F

logging.basicConfig(filename='logs/{0}.txt'.format(os.path.split(__file__)[-1].split('.')[0]), level=logging.DEBUG)
logger = logging.getLogger()

try:
    _conf_file = utils.get_path([define.CONF_DIR,
                                 define.CONF_FILE_NAME])
    options = get_properties(_conf_file)
except EditorConfigError:
    logging.warning("Error getting EditorConfig propterties", exc_info=True)
else:
    for key, value in options.items():
        # _config = '{0},{1}:{2}'.format(_config, key, value)
        print("{0}:{1}".format(key, value))

uqer_client = uqer.Client(token=options.get('uqer_token'))


def train_model_reg_0(predict_windows: list = [20],
                      lag_windows: list = [60],
                      start_date: str = '', end_date: str = '',
                      top_k_features: int = 5,
                      train_days: int = 3,
                      product_id: str = 'RB'):
    df = DataAPI.MktFutdGet(endDate=end_date, beginDate=start_date, pandas="1")
    df = df[df.mainCon == 1]
    df = df[df.contractObject == product_id][['ticker', 'tradeDate', 'exchangeCD']]
    train_dates = df.iloc[-train_days:-1, :]
    test_dates = df.iloc[-1, :]
    acc_scores = {}
    factor_lst = []
    cnt = 0
    df_corr_lst = []
    _lin_reg_model = LinearRegression()
    _svr_model = SVR(C=1.0, epsilon=0.2)
    _ev_model = ElasticNet(random_state=0, l1_ratio=0.5)
    _rf_model = RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=0)
    reg_model = _ev_model
    # pca = PCA(n_components=5, svd_solver='full')
    print('start train model 1')
    for instrument_id, date, exchange_cd in train_dates.values:
        print("train for trade date:", date)
        start_ts = time.time()
        df_factor = F.get_factor(trade_date=date,
                                 predict_windows=predict_windows,
                                 lag_windows=lag_windows,
                                 instrument_id=instrument_id,
                                 exchange_cd=exchange_cd)
        end_ts = time.time()
        print('update factor timestamp:{0}'.format(end_ts - start_ts))
        cols = list(df_factor.columns)
        cols.remove('label_clf')
        cols.remove('UpdateTime')
        cols.remove('label_reg')
        df_corr = df_factor[cols].corr()
        factor_lst.append(df_factor[cols])
        df_score = df_corr['label_cumsum'].abs()
        _features = list(df_score.index)
        for idx, item in enumerate(_features):
            if item == 'label_cumsum':
                continue
            if item in acc_scores:
                acc_scores[item] += df_score[idx]
            else:
                acc_scores[item] = df_score[idx]
        top_features = df_corr['label_cumsum'].abs().sort_values(ascending=False)[1:top_k_features]
        print(list(top_features.index), top_features)
        _file_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR, define.TICK_MODEL_DIR,
                                  'corr_reg_{0}_{1}_{2}.csv'.format(date, predict_windows, lag_windows))
        df_corr.to_csv(_file_name)
        df_corr_lst.append(df_corr)
    _scores = [item / (train_days - 1) for item in list(acc_scores.values())]
    _features = list(acc_scores.keys())
    _tmp = list(zip(_features, _scores))
    _tmp = sorted(_tmp, key=lambda x: x[1], reverse=True)
    print("top features", _tmp[:top_k_features])
    top_feature_lst = [item[0] for item in _tmp[:top_k_features]]

    if df_corr_lst:
        df_corr_sum = df_corr_lst[0]
        for _df_corr in df_corr_lst[1:]:
            df_corr_sum = (df_corr_sum.abs() + _df_corr.abs()) / 2

    # for df_factor in factor_lst:
    #     pca.fit(df_factor[top_feature_lst])

    _corr_set = set()
    for _col in list(df_corr_sum.columns):
        index_lst = list(df_corr_sum[df_corr_sum[_col] > 0.8].index)
        for _tmp in index_lst:
            _tmp_lst = sorted([_col, _tmp])
            _v = '{0}-{1}'.format(_tmp_lst[0], _tmp_lst[1])
            _corr_set.add(_v)
    print("corr average:")
    pprint.pprint(_corr_set)

    for df_factor in factor_lst:
        try:
            # evaluate PCA
            # fit_result = sm.OLS(df_factor['label_cumsum'], df_factor[top_feature_lst]).fit()
            # print('before pca', '*' * 40)
            # pprint.pprint(fit_result.summary())
            # x_train = pca.transform(df_factor[top_feature_lst])
            reg_model.fit(df_factor[top_feature_lst], df_factor['label_cumsum'])
            # fit_result = sm.OLS(df_factor['label_cumsum'], x_train).fit()
            # print('after pca', '*' * 40)
            # pprint.pprint(fit_result.summary())
            del df_factor
        except Exception as ex:
            print('train error')

        # print('intercept:', lin_reg_model.intercept_)
        # print('coef_:{0}\n'.format(list(zip(top_feature_lst, lin_reg_model.coef_))))

    print('test for trade date:', test_dates[1])
    df_factor = F.get_factor(trade_date=test_dates[1],
                             predict_windows=predict_windows,
                             lag_windows=lag_windows,
                             instrument_id=test_dates[0],
                             exchange_cd=test_dates[2])
    # cols = list(df_factor.columns)
    # cols.remove('label_clf')
    _update_time = list(df_factor['UpdateTime'])
    # x_test = pca.transform(df_factor[top_feature_lst])
    y_pred = reg_model.predict(df_factor[top_feature_lst])
    ret_str = 'rmse:{0}, r2:{1}, date:{2},predict windows:{3}, lag windows:{4}, instrument_id:{5}\n'.format(
        np.sqrt(metrics.mean_squared_error(df_factor['label_cumsum'], y_pred)),
        metrics.r2_score(df_factor['label_cumsum'], y_pred), test_dates[1], predict_windows, lag_windows, instrument_id)
    r_ret = [item - y_pred[idx] for idx, item in enumerate(list(df_factor['label_cumsum']))]
    # plt.plot(y_pred)
    # plt.plot(df_factor['label_cumsum'])
    # plt.plot(r_ret)
    # plt.show()
    _summary_path = utils.get_path([define.RESULT_DIR, define.TICK_MODEL_DIR,
                                    'summary.txt'])
    _evalute_path = utils.get_path([define.RESULT_DIR, define.TICK_MODEL_DIR,
                                    'model_evaluate.json'])
    with open(_summary_path, 'a') as f:
        f.write(ret_str)
    # print('rmse:', np.sqrt(metrics.mean_squared_error(df_factor['label_cumsum'], y_pred)), 'r2:',
    #       metrics.r2_score(df_factor['label_cumsum'], y_pred))

    _model_evaluate = utils.load_json_file(_evalute_path) or dict()
    _ret_lst = _model_evaluate.get('{0}_{1}'.format(instrument_id, test_dates[1].replace('-', ''))) or list()
    _ret_lst.append([np.sqrt(metrics.mean_squared_error(df_factor['label_cumsum'], y_pred)),
                     metrics.r2_score(df_factor['label_cumsum'], y_pred), test_dates[1], predict_windows,
                     lag_windows])
    _model_evaluate.update({'{0}_{1}'.format(instrument_id, test_dates[1].replace('-', '')): _ret_lst})
    utils.write_json_file(_evalute_path, _model_evaluate)

    df_pred = pd.DataFrame({'UpdateTime': df_factor['UpdateTime'], 'pred': y_pred, 'label': df_factor['label_cumsum']})
    _file_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR, define.TICK_MODEL_DIR,
                              'pred_{0}_{1}_{2}_{3}.csv'.format(instrument_id, test_dates[1].replace('-', ''),
                                                                predict_windows, lag_windows))
    df_pred.to_csv(_file_name, index=False)
    del df_factor
    # plt.plot(y_pred)
    # plt.plot(df_factor['label_cumsum'])
    # plt.show()
    # f.write('rmse:{0}\n r2:{1}\n'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    #                                      metrics.r2_score(y_test, y_pred)))


def train_model_reg_with_feature_preselect(predict_windows: list = [20],
                                           lag_windows: list = [10, 600],
                                           start_date: str = '', end_date: str = '', top_k_features: int = 5,
                                           train_days: int = 3,
                                           product_id: str = 'RB'):
    df = DataAPI.MktFutdGet(endDate=end_date, beginDate=start_date, pandas="1")
    df = df[df.mainCon == 1]
    df = df[df.contractObject == product_id][['ticker', 'tradeDate', 'exchangeCD']]
    train_dates = df.iloc[-train_days:-1, :]
    test_dates = df.iloc[-1, :]
    acc_scores = {}
    # factor_lst = []
    cnt = 0
    df_corr_lst = []
    _lin_reg_model = LinearRegression()
    _svr_model = SVR(C=1.0, epsilon=0.2)
    _ev_model = ElasticNet(random_state=0, l1_ratio=0.5)
    _rf_model = RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=0)
    reg_model = _ev_model

    # pca = PCA(n_components=5, svd_solver='full')
    print('start train model 1')
    final_df_factor = None
    factor_cnt = 0
    for instrument_id, date, exchange_cd in train_dates.values:
        print("train for trade date:", date)
        start_ts = time.time()
        df_factor = F.get_factor(trade_date=date,
                                 predict_windows=predict_windows,
                                 lag_windows=lag_windows,
                                 instrument_id=instrument_id,
                                 exchange_cd=exchange_cd)
        if factor_cnt == 0:
            final_df_factor = deepcopy(df_factor)
        else:
            final_df_factor = final_df_factor.append(df_factor)
        factor_cnt += 1
        end_ts = time.time()
        print('update factor timestamp:{0}'.format(end_ts - start_ts))
        cols = list(df_factor.columns)
        # cols.remove('label_clf')
        cols.remove('UpdateTime')
        df_corr = df_factor[cols].corr()
        df_score = df_corr['label_cumsum'].abs()
        _features = list(df_score.index)
        for idx, item in enumerate(_features):
            if item == 'label_cumsum':
                continue
            if item in acc_scores:
                acc_scores[item] += df_score[idx]
            else:
                acc_scores[item] = df_score[idx]
        top_features = df_corr['label_cumsum'].abs().sort_values(ascending=False)[1:top_k_features]
        print(list(top_features.index), top_features)
        _file_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR, define.TICK_MODEL_DIR,
                                  'corr_reg_{0}_{1}_{2}.csv'.format(date, predict_windows, lag_windows))
        df_corr.to_csv(_file_name)
        df_corr_lst.append(df_corr)
    _scores = [item / (train_days - 1) for item in list(acc_scores.values())]
    _features = list(acc_scores.keys())
    _tmp = list(zip(_features, _scores))
    _tmp = sorted(_tmp, key=lambda x: x[1], reverse=True)
    print("top features", _tmp[:top_k_features])
    top_feature_lst = [item[0] for item in _tmp[:top_k_features]]

    if False:  # cal factor corr
        if df_corr_lst:
            df_corr_sum = df_corr_lst[0]
            for _df_corr in df_corr_lst[1:]:
                df_corr_sum = (df_corr_sum.abs() + _df_corr.abs()) / 2
        _corr_set = set()
        for _col in list(df_corr_sum.columns):
            index_lst = list(df_corr_sum[df_corr_sum[_col] > 0.8 & df_corr_sum[_col] != 1.0].index)
            for _tmp in index_lst:
                _tmp_lst = sorted([_col, _tmp])
                _v = '{0}-{1}'.format(_tmp_lst[0], _tmp_lst[1])
                _corr_set.add(_v)
        logger.debug('corr average', _corr_set)

    # scoring = ('r2', 'neg_mean_squared_error', 'mean_absolute_percentage_error')
    print('train shape:{0}'.format(final_df_factor.shape))
    cv_results = cross_validate(reg_model, final_df_factor[top_feature_lst], final_df_factor['label_cumsum'], cv=3,
                                n_jobs=3,
                                scoring=('r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error'))
    print('train results:', cv_results)

    reg_model.fit(final_df_factor[top_feature_lst], final_df_factor[define.LABEL_NAME])
    # print('intercept:', lin_reg_model.intercept_)
    # print('coef_:{0}\n'.format(list(zip(top_feature_lst, lin_reg_model.coef_))))

    print('test for trade date:', test_dates[1])
    df_factor = F.get_factor(trade_date=test_dates[1],
                             predict_windows=predict_windows,
                             lag_windows=lag_windows,
                             instrument_id=test_dates[0],
                             exchange_cd=test_dates[2])
    # cols = list(df_factor.columns)
    # cols.remove('label_clf')
    _update_time = list(df_factor['UpdateTime'])
    # x_test = pca.transform(df_factor[top_feature_lst])
    y_pred = reg_model.predict(df_factor[top_feature_lst])
    ret_str = 'rmse:{0}, r2:{1}, date:{2},predict windows:{3}, lag windows:{4}, instrument_id:{5}\n'.format(
        np.sqrt(metrics.mean_squared_error(df_factor['label_cumsum'], y_pred)),
        metrics.r2_score(df_factor['label_cumsum'], y_pred), test_dates[1], predict_windows, lag_windows, instrument_id)

    _summary_path = utils.get_path([define.RESULT_DIR, define.TICK_MODEL_DIR,
                                    'summary.txt'])
    _evalute_path = utils.get_path([define.RESULT_DIR, define.TICK_MODEL_DIR,
                                    'model_evaluate.json'])
    with open(_summary_path, 'a') as f:
        f.write(ret_str)

    _model_evaluate = utils.load_json_file(_evalute_path) or dict()
    _ret_lst = _model_evaluate.get('{0}_{1}'.format(instrument_id, test_dates[1].replace('-', ''))) or list()
    _ret_lst.append([np.sqrt(metrics.mean_squared_error(df_factor['label_cumsum'], y_pred)),
                     metrics.r2_score(df_factor['label_cumsum'], y_pred), test_dates[1], predict_windows,
                     lag_windows])
    _model_evaluate.update({'{0}_{1}'.format(instrument_id, test_dates[1].replace('-', '')): _ret_lst})
    utils.write_json_file(_evalute_path, _model_evaluate)

    df_pred = pd.DataFrame({'UpdateTime': df_factor['UpdateTime'], 'pred': y_pred, 'label': df_factor['label_cumsum']})
    _file_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR, define.TICK_MODEL_DIR,
                              'pred_{0}_{1}_{2}_{3}.csv'.format(instrument_id, test_dates[1].replace('-', ''),
                                                                predict_windows, lag_windows))
    df_pred.to_csv(_file_name, index=False)
    del df_factor


def train_model_sta(predict_windows: list = [20],
                    lag_windows: list = [60],
                    start_date: str = '',
                    end_date: str = '',
                    top_k_features: int = 5,
                    train_days: int = 3,
                    product_id: str = 'RB'):
    df = DataAPI.MktFutdGet(endDate=end_date, beginDate=start_date, pandas="1")
    df = df[df.mainCon == 1]
    df = df[df.contractObject == product_id][['ticker', 'tradeDate', 'exchangeCD']]
    train_dates = df.iloc[-train_days:-1, :]
    test_dates = df.iloc[-1, :]
    acc_scores = {}
    factor_lst = []
    cnt = 0
    df_corr_lst = []
    lin_reg_model = LinearRegression()
    # pca = PCA(n_components=5, svd_solver='full')
    logger.info('start train model 1')
    for instrument_id, date, exchange_cd in train_dates.values:
        logger.info("train for trade date:", date)
        df_factor = F.get_factor(trade_date=date,
                                 predict_windows=predict_windows,
                                 lag_windows=lag_windows,
                                 instrument_id=instrument_id,
                                 exchange_cd=exchange_cd)

        cols = list(df_factor.columns)
        # cols.remove('label_clf')
        cols.remove('UpdateTime')
        cols.remove('label_reg')
        df_corr = df_factor[cols].corr()
        factor_lst.append(df_factor[cols])
        df_score = df_corr['label_cumsum'].abs()
        _features = list(df_score.index)
        for idx, item in enumerate(_features):
            if item == 'label_cumsum':
                continue
            if item in acc_scores:
                acc_scores[item] += df_score[idx]
            else:
                acc_scores[item] = df_score[idx]
        top_features = df_corr['label_cumsum'].abs().sort_values(ascending=False)[1:top_k_features]
        logger.info(list(top_features.index), top_features)
        _file_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR, define.TICK_MODEL_DIR,
                                  'corr_reg_{0}_{1}_{2}.csv'.format(date, predict_windows, lag_windows))
        df_corr.to_csv(_file_name)
        df_corr_lst.append(df_corr)
    _scores = [item / (train_days - 1) for item in list(acc_scores.values())]
    _features = list(acc_scores.keys())
    _tmp = list(zip(_features, _scores))
    _tmp = sorted(_tmp, key=lambda x: x[1], reverse=True)
    logger.info("top features", _tmp[:top_k_features])
    top_feature_lst = [item[0] for item in _tmp[:top_k_features]]

    if df_corr_lst:
        df_corr_sum = df_corr_lst[0]
        for _df_corr in df_corr_lst[1:]:
            df_corr_sum = (df_corr_sum.abs() + _df_corr.abs()) / 2

    # for df_factor in factor_lst:
    #     pca.fit(df_factor[top_feature_lst])

    _corr_set = set()
    for _col in list(df_corr_sum.columns):
        index_lst = list(df_corr_sum[df_corr_sum[_col] > 0.8].index)
        for _tmp in index_lst:
            _tmp_lst = sorted([_col, _tmp])
            _v = '{0}-{1}'.format(_tmp_lst[0], _tmp_lst[1])
            _corr_set.add(_v)
    logger.debug("corr average:{0}".format(_corr_set))

    for df_factor in factor_lst:
        try:
            vif = [variance_inflation_factor(df_factor[top_feature_lst].values, i) for i in
                   range(df_factor[top_feature_lst].shape[1])]
            logger.info(list(zip(top_feature_lst, vif)))
            # evaluate PCA
            fit_result = sm.OLS(df_factor['label_cumsum'], df_factor[top_feature_lst]).fit()
            # print('before pca', '*' * 40)
            pprint.pprint(fit_result.summary())
            # fit_result = sm.OLS(df_factor['label_cumsum'], x_train).fit()
            # print('after pca', '*' * 40)
            # pprint.pprint(fit_result.summary())
            del df_factor
        except Exception as ex:
            logger.info('train error with error msg:{0}'.format(ex))


def _feature_preprocess(df_ref_factor, df_factor, cols):
    _tmp = {}
    for col in cols:
        _low, _high = df_ref_factor[col].quantile([0.1, 0.9])
        _max = df_ref_factor[col].max()
        _min = df_ref_factor[col].min()
        ref = np.array([_low if item < _low else _high if item > _high else item for item in df_ref_factor[col]])
        _std = ref.std()
        _mean = ref.mean()
        val = [_min if item < _min else _max if item > _max else item for item in df_factor[col]]
        val = [_low if item < _mean - 10 * _std else _high if item > _mean + 10 * _std else item for item in
               val]
        # _tmp[col] = [(item - _low) / (_high - _low) for item in val]
        # _tmp[col] = [(item - _min) / (_max - _min) for item in val]

        # _tmp[col] = [(item - _min) / (_max - _min) for item in df_factor[col] ]
        _tmp[col] = df_factor[col]
        _tmp[define.REG_LABEL_NAME] = df_factor[define.REG_LABEL_NAME]
    return pd.DataFrame(_tmp)


def _resample(df_factor):
    from random import sample
    df0 = df_factor[df_factor[define.CLF_LABEL_NAME] == 0]
    df1 = df_factor[df_factor[define.CLF_LABEL_NAME] == 1]
    # df2 = df_factor[df_factor[define.LABEL_NAME] == 2]
    df3 = df_factor[df_factor[define.CLF_LABEL_NAME] == -1]
    # df4 = df_factor[df_factor[define.LABEL_NAME] == -2]
    num0 = df0.shape[0]
    sample_0 = sample(list(range(num0)), int(num0 * 1))
    df1 = df1.append(df0.iloc[sample_0])
    # df1 = df1.append(df2)
    df1 = df1.append(df3)
    # df1 = df1.append(df4)
    return df1


def train_model_clf(predict_windows: list = [20],
                    lag_windows: list = [60],
                    start_date: str = '',
                    end_date: str = '',
                    top_k_features: int = 1,
                    train_days: int = 3,
                    product_id: str = 'RB'):
    df = DataAPI.MktFutdGet(endDate=end_date, beginDate=start_date, pandas="1")
    df = df[df.mainCon == 1]
    df = df[df.contractObject == product_id][['ticker', 'tradeDate', 'exchangeCD']]
    train_dates = df.iloc[-train_days:-1, :]
    test_dates = df.iloc[-1, :]
    model1 = SelectKBest(mutual_info_classif, k=top_k_features)

    import pickle

    _model_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR,
                               define.TICK_MODEL_DIR,
                               'clfmodel_{0}.pkl'.format(product_id))

    try:
        # Load from file
        with open(_model_name, 'rb') as file:
            model2 = pickle.load(file)
            print('model loaded====>', model2.coef_, model2.intercept_, model2.classes_)
    except Exception as ex:
        print('load model:{0} fail with error:{1}'.format(_model_name, ex))
        model2 = LogisticRegression(class_weight='balanced')

    acc_scores = {}
    factor_lst = []
    cnt = 0
    related_factors = []
    for instrument_id, date, exchange_cd in train_dates.values:
        print("train for trade date with date:{0}", date)
        df_factor = F.get_factor(trade_date=date,
                                 predict_windows=predict_windows,
                                 lag_windows=lag_windows,
                                 instrument_id=instrument_id,
                                 exchange_cd=exchange_cd)

        cols = list(df_factor.columns)
        # cols = ['open_close_ratio', 'oi', 'oir', 'aoi', 'slope', 'cos', 'macd', 'dif', 'dea', 'bsvol_volume_0',
        #         'trend_1', 'bsvol_volume_1',
        #         'trend_ls_diff', 'trend_ls_ratio', 'volume_ls_ratio', 'turnover_ls_ratio', 'bs_vol_ls_ratio',
        #         'bsvol_volume_ls_ratio']
        cols = define.train_cols
        import copy
        reg_cols = copy.deepcopy(define.train_cols)
        reg_cols.append('label_1')
        # reg_cols = ['open_close_ratio', 'oi', 'oir', 'aoi', 'slope', 'cos', 'macd', 'dif', 'dea', 'bsvol_volume_0',
        #             'trend_1', 'bsvol_volume_1',
        #             'trend_ls_diff', 'trend_ls_ratio', 'volume_ls_ratio', 'turnover_ls_ratio', 'bs_vol_ls_ratio',
        #             'bsvol_volume_ls_ratio', 'label_1']

        df_corr = df_factor[reg_cols].corr()
        df_corr.to_csv('corr_{0}.csv'.format(date))
        for item in reg_cols:
            try:
                _similar_factors = list(df_corr[df_corr[item] > 0.07][item].index)
                for _k in _similar_factors:
                    if not ((item, _k) in related_factors or (_k, item) in related_factors) and _k != item:
                        related_factors.append((item, _k))
            except Exception as ex:
                print(ex, date)
        factor_lst.append(df_factor)

        # model1.fit(df_factor[cols], df_factor['label_clf_1'])

        # logger.info("model selection scores for mutual information:{0}".format(list(zip(cols, model1.scores_))))
        # cnt += 1
        # rank_score = rankdata(model1.scores_)
        # for idx, item in enumerate(rank_score):
        #     try:
        #         if cols[idx] not in acc_scores:
        #             acc_scores[cols[idx]] = item
        #         else:
        #             acc_scores[cols[idx]] += item
        #     except Exception as ex:
        #         logger.info('id:{0}, error:{1}, col:{2}'.format(idx, ex, cols))

    if top_k_features < 0:
        acc_scores = dict(zip(cols, [0.1] * len(cols)))

    for k1, k2 in related_factors:
        if k1 in acc_scores and k2 in acc_scores:
            print("{0}:{1} and {2}:{3} in dict, pop{4}".format(k1, acc_scores.get(k1), k2, acc_scores.get(k2), k2))
            # acc_scores.pop(k2)  # TODO always pop k2, add more logic here if needed

    top_k_features = len(acc_scores) if top_k_features < 0 else top_k_features
    sorted_scores = sorted(acc_scores.items(), reverse=True, key=lambda x: x[1])[-top_k_features:]
    sorted_features = [item[0] for item in sorted_scores]
    logger.info(sorted_scores)
    print("sorted features:", sorted_features)

    df_train = _resample(factor_lst[0])
    for df_factor in factor_lst[1:]:
        df_train = df_train.append(_resample(df_factor))
    model2.fit(df_train[sorted_features], df_train[define.LABEL_NAME])
    _model_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR,
                               define.TICK_MODEL_DIR,
                               'clfmodel_{0}.pkl'.format(product_id))

    with open(_model_name, 'wb') as file:
        pickle.dump(model2, file)

    del factor_lst

    # for df_factor in factor_lst:
    #     df_train = _resample(df_factor)
    #     model2.fit(df_train[sorted_features], df_train['label_clf_1'])
    logger.info('test for trade date:{0}'.format(test_dates[1]))

    df_test = F.get_factor(trade_date=test_dates[1],
                           predict_windows=predict_windows,
                           lag_windows=lag_windows,
                           instrument_id=test_dates[0],
                           exchange_cd=test_dates[2])
    # cols = list(df_factor.columns)
    # cols.remove('label_clf')

    # y_pred = model2.predict(model1.transform(df_factor[cols])) #TODO why this??
    # if there might be class imbalance, then  micro is preferable since micro-average will aggregate the contributions
    # of all classes to compute the average metric, while macro-average will compute the metric independently for each
    # class and then take the average(hence treating all classes equally)

    y_true = list(df_test[define.LABEL_NAME])
    y_pred = model2.predict(df_test[sorted_features])
    _update_time = list(df_test['UpdateTime'])
    _update_time_str = [item.split()[-1] for item in _update_time]

    dict_score = {}

    for _clf_label in model2.classes_:
        _pred = len([item for item in y_pred if item == _clf_label])
        _true = len([item for item in y_true if item == _clf_label])
        _num_prec = len([item for idx, item in enumerate(y_pred) if y_true[idx] == _clf_label and item == _clf_label])
        _prec_ratio = _num_prec / _pred if _pred > 0 else 0
        dict_score.update({'pred_{0}'.format(_clf_label): _pred, 'true_{0}'.format(_clf_label): _true,
                           'n_correct_{0}'.format(_clf_label): _num_prec,
                           'ratio_correct_{0}'.format(_clf_label): _prec_ratio})

    dict_score.update({'macro precision': metrics.precision_score(y_pred, y_true, average='macro')})
    dict_score.update({'micro precisionn': metrics.precision_score(y_pred, y_true, average='micro')})

    df_pred = pd.DataFrame({'pred': y_pred, 'true': y_true, 'UpdateTime': _update_time_str})
    # df_pred.to_csv("df_pred.csv")
    # FIXME double check the file path

    df_coef = pd.DataFrame(model2.coef_, columns=sorted_features)
    df_coef['classes'] = model2.classes_
    _cofe_file_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR,
                                   define.TICK_MODEL_DIR,
                                   'coef_{0}_{1}.csv'.format(test_dates[1].replace('-', ''), instrument_id))
    df_coef.to_csv(_cofe_file_name, index=False)
    _inter_file_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR,
                                    define.TICK_MODEL_DIR,
                                    'inter_{0}_{1}.json'.format(test_dates[1].replace('-', ''), instrument_id))
    utils.write_json_file(_inter_file_name, list(model2.intercept_))
    _file_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR, define.TICK_MODEL_DIR,
                              'df_pred_{0}.csv'.format(test_dates[1]))
    df_pred.to_csv(_file_name)
    return dict_score


def train_model_ols(predict_windows: list = [20],
                    lag_windows: list = [60],
                    start_date: str = '',
                    end_date: str = '',
                    top_k_features: int = 5,
                    train_days: int = 3,
                    product_id: str = 'RB'):
    df = DataAPI.MktFutdGet(endDate=end_date, beginDate=start_date, pandas="1")
    df = df[df.mainCon == 1]
    df = df[df.contractObject == product_id][['ticker', 'tradeDate', 'exchangeCD']]
    train_dates = df.iloc[-train_days:-1, :]
    test_dates = df.iloc[-1, :]
    model1 = SelectKBest(mutual_info_classif, k=top_k_features)
    model2 = LogisticRegression()
    acc_scores = {}
    factor_lst = []
    cnt = 0
    df_corr_lst = []
    # lin_reg_model_ols = LinearRegression()
    for instrument_id, date, exchange_cd in train_dates.values:
        print("train for trade date:", date)
        df_factor = F.get_factor(trade_date=date,
                                 predict_windows=predict_windows,
                                 lag_windows=lag_windows,
                                 instrument_id=instrument_id,
                                 exchange_cd=exchange_cd)

        cols = list(df_factor.columns)
        cols.remove('label_clf')
        cols.remove('UpdateTime')
        cols.remove('label_reg')
        df_corr = df_factor[cols].corr()
        factor_lst.append(df_factor[cols])
        df_score = df_corr['label_cumsum'].abs()
        _features = list(df_score.index)
        for idx, item in enumerate(_features):
            if item == 'label_cumsum':
                continue
            if item in acc_scores:
                acc_scores[item] += df_score[idx]
            else:
                acc_scores[item] = df_score[idx]
        top_features = df_corr['label_cumsum'].abs().sort_values(ascending=False)[1:top_k_features]
        print(list(top_features.index), top_features)
        _file_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR, define.TICK_MODEL_DIR,
                                  'corr_reg_{0}_{1}_{2}.csv'.format(date, predict_windows, lag_windows))
        df_corr.to_csv(_file_name)
        df_corr_lst.append(df_corr)
    _scores = [item / (train_days - 1) for item in list(acc_scores.values())]
    _features = list(acc_scores.keys())
    _tmp = list(zip(_features, _scores))
    _tmp = sorted(_tmp, key=lambda x: x[1], reverse=True)
    print("top features", _tmp[:top_k_features])
    top_feature_lst = [item[0] for item in _tmp[:top_k_features]]

    if df_corr_lst:
        df_corr_sum = df_corr_lst[0]
        for _df_corr in df_corr_lst[1:]:
            df_corr_sum = (df_corr_sum + _df_corr) / 2

    for df_factor in factor_lst:
        # lin_reg_model.fit(df_factor[top_feature_lst], df_factor['label_cumsum'])
        fit_result = sm.OLS(df_factor['label_cumsum'], df_factor[top_feature_lst]).fit()
        pprint.pprint(fit_result.summary())
    # print('test for trade date:', test_dates[1])
    # df_factor = get_factor(trade_date=test_dates[1],
    #                        predict_windows=predict_windows,
    #                        lag_long=lag_windows,
    #                        stop_profit=stop_profit, stop_loss=stop_loss, instrument_id=test_dates[0],
    #                        exchange_cd=test_dates[2])
    # # cols = list(df_factor.columns)
    # # cols.remove('label_clf')
    # _update_time = list(df_factor['UpdateTime'])
    # y_pred = lin_reg_model.predict(df_factor[top_feature_lst])
    # print('rmse:', np.sqrt(metrics.mean_squared_error(df_factor['label_cumsum'], y_pred)), 'r2:',
    #       metrics.r2_score(df_factor['label_cumsum'], y_pred))


def train_model_reg_without_feature_preselect(predict_windows: list = [20],
                                              lag_windows: list = [10, 600],
                                              start_date: str = '', end_date: str = '', top_k_features: int = 5,
                                              train_days: int = 3,
                                              product_id: str = 'RB'):
    df = DataAPI.MktFutdGet(endDate=end_date, beginDate=start_date, pandas="1")
    df = df[df.mainCon == 1]
    df = df[df.contractObject == product_id][['ticker', 'tradeDate', 'exchangeCD']]
    train_dates = df.iloc[-train_days:-1, :]
    test_dates = df.iloc[-1, :]
    acc_scores = {}
    # factor_lst = []
    cnt = 0
    df_corr_lst = []
    _lin_reg_model = LinearRegression()
    _svr_model = SVR(C=1.0, epsilon=0.2)
    reg_model = _lin_reg_model

    # pca = PCA(n_components=5, svd_solver='full')
    print('start train model from date {0} to {1}'.format(start_date, end_date))
    final_df_factor = None
    factor_cnt = 0
    for instrument_id, date, exchange_cd in train_dates.values:
        print("train for trade date:", date)
        start_ts = time.time()
        _factor_path = utils.get_path([define.CACHE_DIR, define.FACTOR_DIR,
                                       'factor_{0}_{1}.csv'.format(instrument_id, date.replace('-', ''))])
        try:
            df_factor = pd.read_csv(_factor_path)
        except Exception as ex:
            print('factor not cache with error:{0}'.format(ex))
            df_factor = F.get_factor(trade_date=date,
                                     predict_windows=predict_windows,
                                     lag_windows=lag_windows,
                                     instrument_id=instrument_id,
                                     exchange_cd=exchange_cd)

        if factor_cnt == 0:
            final_df_factor = deepcopy(df_factor)
        else:
            final_df_factor = final_df_factor.append(df_factor)
        del df_factor
        factor_cnt += 1
        end_ts = time.time()
        print('update factor timestamp:{0}'.format(end_ts - start_ts))
        # TODO remove hardcode of the factor
        cols = ['trend_ls_ratio', 'bs_vol_ls_ratio', 'open_close_ratio', 'slope', 'oir']
        # cols = list(final_df_factor.columns)
        # cols.remove('label_clf')
        # cols.remove('UpdateTime')
    # calculate factor
    #     df_corr = df_factor[cols].corr()
    #     # calculate corr score
    #     df_score = df_corr['label_cumsum'].abs()
    #     _features = list(df_score.index)
    #     for idx, item in enumerate(_features):
    #         if item == 'label_cumsum':
    #             continue
    #         if item in acc_scores:
    #             acc_scores[item] += df_score[idx]
    #         else:
    #             acc_scores[item] = df_score[idx]
    #     top_features = df_corr['label_cumsum'].abs().sort_values(ascending=False)[1:top_k_features]
    #     print(list(top_features.index), top_features)
    #     df_corr_lst.append(df_corr)
    #     _file_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR, define.TICK_MODEL_DIR,
    #                               'corr_reg_{0}_{1}_{2}.csv'.format(date, predict_windows, lag_windows))
    #     df_corr.to_csv(_file_name)
    #
    # _scores = [item / (train_days - 1) for item in list(acc_scores.values())]
    # _features = list(acc_scores.keys())
    # _tmp = list(zip(_features, _scores))
    # _tmp = sorted(_tmp, key=lambda x: x[1], reverse=True)
    # print("top features", _tmp[:top_k_features])
    # top_feature_lst = [item[0] for item in _tmp[:top_k_features]]

    if False:  # cal factor corr
        if df_corr_lst:
            df_corr_sum = df_corr_lst[0]
            for _df_corr in df_corr_lst[1:]:
                df_corr_sum = (df_corr_sum.abs() + _df_corr.abs()) / 2
        _corr_set = set()
        for _col in list(df_corr_sum.columns):
            index_lst = list(df_corr_sum[df_corr_sum[_col] > 0.8 & df_corr_sum[_col] != 1.0].index)
            for _tmp in index_lst:
                _tmp_lst = sorted([_col, _tmp])
                _v = '{0}-{1}'.format(_tmp_lst[0], _tmp_lst[1])
                _corr_set.add(_v)
        logger.debug('corr average', _corr_set)

    selected_cols = deepcopy(cols)
    for item in cols:
        if item == 'rv_0':
            pass
        if item.startswith('label') or item.startswith('rv'):
            selected_cols.remove(item)
    print('train shape:{0}'.format(final_df_factor[selected_cols].shape))
    cv_results = cross_validate(reg_model, final_df_factor[selected_cols], final_df_factor[define.LABEL_NAME], cv=3,
                                n_jobs=3,
                                scoring=('r2', 'neg_mean_squared_error'))
    print('train results:', cv_results)

    reg_model.fit(final_df_factor[selected_cols], final_df_factor[define.LABEL_NAME])

    print('test for trade date:', test_dates[1])
    df_factor = F.get_factor(trade_date=test_dates[1],
                             predict_windows=predict_windows,
                             lag_windows=lag_windows,
                             instrument_id=test_dates[0],
                             exchange_cd=test_dates[2])
    _update_time = list(df_factor['UpdateTime'])
    # x_test = pca.transform(df_factor[top_feature_lst])
    y_pred = reg_model.predict(df_factor[selected_cols])
    ret_str = 'rmse:{0}, r2:{1}, date:{2},predict windows:{3}, lag windows:{4}, instrument_id:{5}\n'.format(
        np.sqrt(metrics.mean_squared_error(df_factor[define.LABEL_NAME], y_pred)),
        metrics.r2_score(df_factor[define.LABEL_NAME], y_pred), test_dates[1], predict_windows, lag_windows,
        instrument_id)

    _summary_path = utils.get_path([define.RESULT_DIR, define.TICK_MODEL_DIR,
                                    'summary.txt'])
    _evalute_path = utils.get_path([define.RESULT_DIR, define.TICK_MODEL_DIR,
                                    'model_evaluate.json'])
    with open(_summary_path, 'a') as f:
        f.write(ret_str)

    _model_evaluate = utils.load_json_file(_evalute_path) or dict()
    _ret_lst = _model_evaluate.get('{0}_{1}'.format(instrument_id, test_dates[1].replace('-', ''))) or list()
    _ret_lst.append([np.sqrt(metrics.mean_squared_error(df_factor[define.LABEL_NAME], y_pred)),
                     metrics.r2_score(df_factor[define.LABEL_NAME], y_pred), test_dates[1], predict_windows,
                     lag_windows])
    _model_evaluate.update({'{0}_{1}'.format(instrument_id, test_dates[1].replace('-', '')): _ret_lst})
    utils.write_json_file(_evalute_path, _model_evaluate)

    df_pred = pd.DataFrame(
        {'UpdateTime': df_factor['UpdateTime'], 'pred': y_pred, 'label': df_factor[define.LABEL_NAME]})
    _file_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR, define.TICK_MODEL_DIR,
                              'pred_{0}_{1}_{2}_{3}_{4}.csv'.format(define.LABEL_NAME, instrument_id,
                                                                    test_dates[1].replace('-', ''),
                                                                    predict_windows, lag_windows))
    df_pred.to_csv(_file_name, index=False)
    del df_factor


def train_model_reg(predict_windows: list = [20],
                    lag_windows: list = [60],
                    start_date: str = '',
                    end_date: str = '',
                    top_k_features: int = 1,
                    train_days: int = 3,
                    product_id: str = 'RB'):
    df = DataAPI.MktFutdGet(endDate=end_date, beginDate=start_date, pandas="1")
    df = df[df.mainCon == 1]
    df = df[df.contractObject == product_id][['ticker', 'tradeDate', 'exchangeCD']]
    train_dates = df.iloc[-train_days:-1, :]
    test_dates = df.iloc[-1, :]

    import pickle

    _model_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR,
                               define.TICK_MODEL_DIR,
                               'regmodel_{0}.pkl'.format(product_id))

    try:
        # Load from file
        with open(_model_name, 'rb') as file:
            model2 = pickle.load(file)
            print('model loaded====>', model2.coef_, model2.intercept_, model2.classes_)
    except Exception as ex:
        print('load model:{0} fail with error:{1}'.format(_model_name, ex))
        model2 = _lin_reg_model = LinearRegression()

    acc_scores = {}
    factor_lst = []
    cnt = 0
    related_factors = []

    for instrument_id, date, exchange_cd in train_dates.values:
        print("train for trade date with date:{0}", date)
        df_factor = F.get_factor(trade_date=date,
                                 predict_windows=predict_windows,
                                 lag_windows=lag_windows,
                                 instrument_id=instrument_id,
                                 exchange_cd=exchange_cd)

        cols = list(df_factor.columns)
        # cols = ['open_close_ratio', 'oi', 'oir', 'aoi', 'slope', 'cos', 'macd', 'dif', 'dea', 'bsvol_volume_0',
        #         'trend_1', 'bsvol_volume_1',
        #         'trend_ls_diff', 'trend_ls_ratio', 'volume_ls_ratio', 'turnover_ls_ratio', 'bs_vol_ls_ratio',
        #         'bsvol_volume_ls_ratio']
        cols = define.train_cols
        import copy
        reg_cols = copy.deepcopy(define.train_cols)
        reg_cols.append('label_1')
        # reg_cols = ['open_close_ratio', 'oi', 'oir', 'aoi', 'slope', 'cos', 'macd', 'dif', 'dea', 'bsvol_volume_0',
        #             'trend_1', 'bsvol_volume_1',
        #             'trend_ls_diff', 'trend_ls_ratio', 'volume_ls_ratio', 'turnover_ls_ratio', 'bs_vol_ls_ratio',
        #             'bsvol_volume_ls_ratio', 'label_1']

        df_corr = df_factor[reg_cols].corr()
        df_corr.to_csv('corr_{0}.csv'.format(date))
        for item in reg_cols:
            try:
                _similar_factors = list(df_corr[df_corr[item] > 0.07][item].index)
                for _k in _similar_factors:
                    if not ((item, _k) in related_factors or (_k, item) in related_factors) and _k != item:
                        related_factors.append((item, _k))
            except Exception as ex:
                print(ex, date)
        factor_lst.append(df_factor)

    if top_k_features < 0:
        acc_scores = dict(zip(cols, [0.1] * len(cols)))

    for k1, k2 in related_factors:
        if k1 in acc_scores and k2 in acc_scores:
            print("{0}:{1} and {2}:{3} in dict, pop{4}".format(k1, acc_scores.get(k1), k2, acc_scores.get(k2), k2))
            # acc_scores.pop(k2)  # TODO always pop k2, add more logic here if needed

    top_k_features = len(acc_scores) if top_k_features < 0 else top_k_features
    sorted_scores = sorted(acc_scores.items(), reverse=True, key=lambda x: x[1])[-top_k_features:]
    sorted_features = [item[0] for item in sorted_scores]
    logger.info(sorted_scores)
    print("sorted features:", sorted_features)

    df_train = None
    if len(factor_lst) > 1:
        for idx in list(range(1, len(factor_lst))):
            _df_train = _feature_preprocess(factor_lst[idx - 1], factor_lst[idx], sorted_features)
            if idx == 1:
                df_train = copy.deepcopy(_df_train)
            else:
                df_train = df_train.append(_df_train)
    df_train.to_csv('train_factor.csv', index=False)
    # df_train = _resample(factor_lst[0])
    # for df_factor in factor_lst[1:]:
    #     df_train = df_train.append(_resample(df_factor))
    model2.fit(df_train[sorted_features], df_train[define.REG_LABEL_NAME])
    _model_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR,
                               define.TICK_MODEL_DIR,
                               'regmodel_{0}.pkl'.format(product_id))

    with open(_model_name, 'wb') as file:
        pickle.dump(model2, file)

    del factor_lst

    # for df_factor in factor_lst:
    #     df_train = _resample(df_factor)
    #     model2.fit(df_train[sorted_features], df_train['label_clf_1'])
    logger.info('test for trade date:{0}'.format(test_dates[1]))

    df_test = F.get_factor(trade_date=test_dates[1],
                           predict_windows=predict_windows,
                           lag_windows=lag_windows,
                           instrument_id=test_dates[0],
                           exchange_cd=test_dates[2])
    # cols = list(df_factor.columns)
    # cols.remove('label_clf')

    # y_pred = model2.predict(model1.transform(df_factor[cols])) #TODO why this??
    # if there might be class imbalance, then  micro is preferable since micro-average will aggregate the contributions
    # of all classes to compute the average metric, while macro-average will compute the metric independently for each
    # class and then take the average(hence treating all classes equally)

    y_true = list(df_test[define.REG_LABEL_NAME])
    y_pred = model2.predict(df_test[sorted_features])
    _update_time = list(df_test['UpdateTime'])
    _update_time_str = [item.split()[-1] for item in _update_time]

    dict_score = {}

    df_pred = pd.DataFrame({'pred': y_pred, 'true': y_true, 'UpdateTime': _update_time_str})

    _file_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR, define.TICK_MODEL_DIR,
                              'df_pred_{0}.csv'.format(test_dates[1]))
    df_pred.to_csv(_file_name)

    ret_str = 'rmse:{0}, r2:{1}, date:{2},predict windows:{3}, lag windows:{4}, instrument_id:{5}\n'.format(
        np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
        metrics.r2_score(y_true, y_pred), test_dates[1], predict_windows, lag_windows, instrument_id)

    _model_evalulate_path = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR,
                                         define.TICK_MODEL_DIR,
                                         'model_evaluate_{0}_{1}.txt'.format(test_dates[1], instrument_id))
    with open(_model_evalulate_path, 'a') as f:
        f.write(ret_str)

    coef_param = dict(zip(sorted_features, list(model2.coef_)))
    reg_params = {'coef': coef_param, 'intercept': model2.intercept_}

    _param_file_name = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR,
                                    define.TICK_MODEL_DIR,
                                    'reg_params_{1}.json'.format(test_dates[1].replace('-', ''), instrument_id))
    utils.write_json_file(_param_file_name, reg_params)

    # r_ret = [item - y_pred[idx] for idx, item in enumerate(y_true)]
    # plt.plot(r_ret)
    print(ret_str)
    # plt.show()
    return dict_score
