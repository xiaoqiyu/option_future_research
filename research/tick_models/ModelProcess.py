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


def _is_trading_time(time_str=''):
    if time_str > '09:00:00.000' and time_str <= '15:00:00.000' or time_str > '21:00:00.000' and time_str < '23:00:01.000':
        return True
    else:
        return False


def get_selected_factor(factor_lst=[]):
    factor_lst.remove('Exchange')
    factor_lst.remove('Instrument')


def train_model_reg(predict_windows=120,
                    lag_windows=60,
                    stop_profit=0.001, stop_loss=0.01, start_date='', end_date='', top_k_features=5, train_days=3,
                    product_id='RB'):
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
                                 lag_long=lag_windows,
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
                             lag_long=lag_windows,
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


def train_model_reg_intraday(predict_windows=120,
                             lag_windows=60,
                             start_date='', end_date='', top_k_features=5,
                             train_days=3,
                             product_id='RB'):
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
    print('start train model 1')
    final_df_factor = None
    factor_cnt = 0
    for instrument_id, date, exchange_cd in train_dates.values:
        print("train for trade date:", date)
        start_ts = time.time()
        df_factor = F.get_factor(trade_date=date,
                                 predict_windows=predict_windows,
                                 lag_long=lag_windows, instrument_id=instrument_id,
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
        # cols.remove('label_reg')
        df_corr = df_factor[cols].corr()
        # factor_lst.append(df_factor[cols])
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
                                n_jobs=3, scoring=('r2', 'neg_mean_absolute_percentage_error','neg_mean_squared_error'))
    print('train results:', cv_results)
    # print('intercept:', lin_reg_model.intercept_)
    # print('coef_:{0}\n'.format(list(zip(top_feature_lst, lin_reg_model.coef_))))

    reg_model.fit(final_df_factor[top_feature_lst], final_df_factor['label_cumsum'])

    print('test for trade date:', test_dates[1])
    df_factor = F.get_factor(trade_date=test_dates[1],
                             predict_windows=predict_windows,
                             lag_long=lag_windows, instrument_id=test_dates[0],
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


def train_model_sta(predict_windows=120,
                    lag_windows=60,
                    start_date='', end_date='', top_k_features=5, train_days=3,
                    product_id='RB'):
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
                                 lag_long=lag_windows,
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


def train_model_clf(predict_windows=120,
                    lag_windows=60,
                    start_date='', end_date='', top_k_features=1, train_days=3):
    df = DataAPI.MktFutdGet(endDate=end_date, beginDate=start_date, pandas="1")
    df = df[df.mainCon == 1]
    df = df[df.contractObject == 'RB'][['ticker', 'tradeDate']]
    train_dates = df.iloc[-train_days:-1, :]
    test_dates = df.iloc[-1, :]
    model1 = SelectKBest(mutual_info_classif, k=top_k_features)
    model2 = LogisticRegression()
    acc_scores = {}
    factor_lst = []
    cnt = 0
    for instrument_id, date in train_dates.values:
        logger.info("train for trade date with date:{0}", date)
        df_factor = F.get_factor(trade_date=date,
                                 predict_windows=predict_windows,
                                 lag_long=lag_windows,
                                 instrument_id=instrument_id)

        cols = list(df_factor.columns)
        cols.remove('label_clf')
        cols.remove('UpdateTime')
        model1.fit(df_factor[cols], df_factor['label_clf'])
        factor_lst.append(df_factor)
        logger.info("model selection scores for mutual information:{0}".format(list(zip(cols, model1.scores_))))
        # acc_scores.append(list(model1.scores_))
        cnt += 1
        rank_score = rankdata(model1.scores_)
        for idx, item in enumerate(rank_score):
            try:
                if cols[idx] not in acc_scores:
                    # acc_scores[cols[idx]] = item * cnt / (train_days - 1)
                    acc_scores[cols[idx]] = item
                else:
                    # acc_scores[cols[idx]] += item * cnt / (train_days - 1)
                    acc_scores[cols[idx]] += item
            except Exception as ex:
                logger.info('id:{0}, error:{1}, col:{2}'.format(idx, ex, cols))
        # print("model selection pvalues for mutual information:", model1.pvalues_)
    sorted_scores = sorted(acc_scores.items(), reverse=True, key=lambda x: x[1])[-top_k_features:]
    sorted_features = [item[0] for item in sorted_scores]
    logger.info(sorted_scores)

    for df_factor in factor_lst:
        model2.fit(df_factor[sorted_features], df_factor['label_clf'])
    logger.info('test for trade date:{0}'.format(test_dates[1]))

    df_factor = F.get_factor(trade_date=test_dates[1],
                             predict_windows=predict_windows,
                             lag_long=lag_windows,
                             instrument_id=test_dates[0])
    # cols = list(df_factor.columns)
    # cols.remove('label_clf')
    _update_time = list(df_factor['UpdateTime'])
    _update_time_str = [item.split()[-1] for item in _update_time]
    y_pred = model2.predict(df_factor[sorted_features])
    y_pred = model2.predict(model1.transform(df_factor[cols]))
    # if there might be class imbalance, then  micro is preferable since micro-average will aggregate the contributions
    # of all classes to compute the average metric, while macro-average will compute the metric independently for each
    # class and then take the average(hence treating all classes equally)
    y_true = list(df_factor['label_clf'])
    pred0 = len([item for item in y_pred if item == 0])
    pred1 = len([item for item in y_pred if item == 1])
    pred2 = len([item for item in y_pred if item == 2])

    true0 = len([item for item in y_true if item == 0])
    true1 = len([item for item in y_true if item == 1])
    true2 = len([item for item in y_true if item == 2])

    num_prec0 = len([item for idx, item in enumerate(y_pred) if y_true[idx] == 0 and item == 0])
    num_prec1 = len([item for idx, item in enumerate(y_pred) if y_true[idx] == 1 and item == 1])
    num_prec2 = len([item for idx, item in enumerate(y_pred) if y_true[idx] == 2 and item == 2])

    logger.info('true vs pred:', '(0,{},{},{})'.format(true0, pred0, num_prec0 / pred0 if pred0 > 0 else 0))
    logger.info('true vs pred:', '(1,{},{},{})'.format(true1, pred1, num_prec1 / pred1 if pred1 > 0 else 0))
    logger.info('true vs pred:', '(2,{},{},{})'.format(true2, pred2, num_prec2 / pred2 if pred2 > 0 else 0))

    # scores for clf models
    # score = metrics.precision_score(y_pred, df_factor['label_clf'], average='macro')
    # score = metrics.precision_score(y_pred, df_factor['label_clf'], average='micro')
    # score = metrics.recall_score(y_pred, df_factor['label_clf'], average='macro')
    # score = metrics.recall_score(y_pred, df_factor['label_clf'], average='micro')
    # score = metrics.f1_score(y_pred, df_factor['label_clf'], average='macro')
    # score = metrics.f1_score(y_pred, df_factor['label_clf'], average='micro')
    df_pred = pd.DataFrame({'pred': y_pred, 'true': df_factor['label_clf'], 'UpdateTime': _update_time_str})
    _file_name = os.path.join(os.path.abspath(os.pardir), define.RESULT_DIR, define.TICK_MODEL_DIR, 'df_pred.csv')
    df_pred.to_csv(_file_name)


def factor_process(product_id='m', trade_date='20210105', instrument_id='m2105', windows_len=1000, stop_profit=3.0):
    _factor_path = os.path.join(os.path.abspath(os.pardir), define.CACHE_DIR, define.FACTOR_DIR,
                                'factor_{0}_{1}.csv'.format(instrument_id, trade_date))
    df = pd.read_csv(_factor_path)
    labels = []
    last_lst = list(df['last_price'])
    vol_lst = list(df['vol'])
    turnover_lst = list(df['turnover'])
    _len = len(last_lst)

    for idx in list(range(_len - windows_len)):
        _max = max(last_lst[idx + 1: idx + windows_len])
        _min = min(last_lst[idx + 1: idx + windows_len])
        # print(last_lst[idx], _max, _max - last_lst[idx])
        if _max - last_lst[idx] > stop_profit:
            labels.append(1)
        elif last_lst[idx] - _min < stop_profit:
            labels.append(2)
        else:
            labels.append(0)

    _in_vol = [0]
    for idx, item in enumerate(vol_lst):
        if idx < 1:
            continue
        _in_vol.append(vol_lst[idx] - vol_lst[idx - 1])
    _in_turnover = [0]
    for idx, item in enumerate(turnover_lst):
        if idx < 1:
            continue
        _in_turnover.append(turnover_lst[idx] - turnover_lst[idx - 1])
    df['in_vol'] = _in_vol
    df['in_turnover'] = _in_turnover
    df['vol_ratio'] = df['in_vol'] / df['vol']
    df['turnover_ratio'] = df['in_turnover'] / df['turnover']
    df['turning_spread'] = df['last_price'] - df['turning']
    df['turning_idx_spread'] = df['last_price'] - df['turning_idx']
    df['in_vwap'] = df['in_turnover'] / df['in_vol']

    l1 = len(labels)
    l2 = len([item for item in labels if item == 1])
    l3 = len([item for item in labels if item == 2])
    # print(len(labels), len([item for item in labels if item == 1]), len([item for item in labels if item == 2]))
    print(l1, l2, l3, l1 - l2 - l3)


def train_model_ols(predict_windows=120,
                    lag_windows=60,
                    start_date='', end_date='', top_k_features=5, train_days=3,
                    product_id='RB'):
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
                                 lag_long=lag_windows,
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


if __name__ == '__main__':
    # factor_process(windows_len=600, stop_profit=5.0, trade_date='20210106')
    # get_factor()
    start_date = '20210701'
    end_date = '20210730'
    trade_date_df = utils.get_trade_dates(start_date=start_date, end_date=end_date)
    trade_date_df = trade_date_df[trade_date_df.exchangeCD == 'XSHE']
    trade_dates = list(trade_date_df['calendarDate'])
    train_days = 3
    num_date = len(trade_dates)
    idx = 0
    # [20, 60, 120, 600, 1200]
    predict_window_lst = [120, 1200]  # 10s, 30s,1min,5min,10min
    lag_window_lst = [60, 120, 600]  # 10s, 30s,1min,5min,
    for predict_win in predict_window_lst:
        for lag_win in lag_window_lst:
            print('train for predict windows:{0} and lag_windows:{1}-----------'.format(predict_win, lag_win))
            while idx < num_date - train_days:
                train_model_reg(predict_windows=predict_win,
                                lag_windows=lag_win,
                                top_k_features=10, start_date=trade_dates[idx],
                                end_date=trade_dates[idx + train_days],
                                train_days=train_days, product_id='RB')
                idx += train_days
                gc.collect()
