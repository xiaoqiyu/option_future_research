#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/3 9:53
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : daily_models.py

import math
import uqer
import datetime
import pprint
import pandas as pd
import numpy as np
from uqer import DataAPI
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

uqer_client = uqer.Client(token="")


# TODO line correlation


def fetch_data(start_date='', end_date=''):
    DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE", beginDate=start_date, endDate=end_date, isOpen=u"1",
                        field=u"", pandas="1")
    df = DataAPI.MktMFutdGet(mainCon=u"1", contractMark=u"", contractObject=u"", tradeDate=u"", startDate=start_date,
                             endDate=end_date, field=u"", pandas="1")
    df = df[df.exchangeCD == 'CCFX']
    df = df[df.mainCon == 1]
    ic_df = df[df.contractObject == 'IC'][['tradeDate', 'ticker']]
    if_df = df[df.contractObject == 'IF'][['tradeDate', 'ticker']]
    ih_df = df[df.contractObject == 'IH'][['tradeDate', 'ticker']]

    df_basis = DataAPI.MktFutIdxBasisGet(secID=u"", ticker=u"", beginDate=start_date, endDate=end_date, field=u"",
                                         pandas="1")
    ic_mkt = ic_df.set_index(['tradeDate', 'ticker']).join(df_basis.set_index(['tradeDate', 'ticker']),
                                                           how='left').reset_index()
    if_mkt = if_df.set_index(['tradeDate', 'ticker']).join(df_basis.set_index(['tradeDate', 'ticker']),
                                                           how='left').reset_index()
    ih_mkt = ih_df.set_index(['tradeDate', 'ticker']).join(df_basis.set_index(['tradeDate', 'ticker']),
                                                           how='left').reset_index()

    icf_mkt = ic_mkt.set_index('tradeDate').join(other=if_mkt.set_index('tradeDate'), on='tradeDate', lsuffix='_ic',
                                                 rsuffix='_if')
    df_mkt = icf_mkt.join(ih_mkt.set_index('tradeDate'))
    df_mkt = df_mkt.sort_values(by='tradeDate', ascending=True)
    df_mkt.reset_index().to_csv('fut_index_mkt_{0}_{1}.csv'.format(start_date, end_date), index=False)


def factor_calculate(predict_windows=20, lag_windows=20):
    df_factor = pd.read_csv('fut_index_mkt.csv')
    df_factor['basis_ic_mean'] = df_factor[['basis_ic']].rolling(lag_windows).mean()
    df_factor['basis_if_mean'] = df_factor[['basis_if']].rolling(lag_windows).mean()
    df_factor['basis_ih_mean'] = df_factor[['basis']].rolling(lag_windows).mean()
    df_factor['basis_ic_pct_annual'] = (df_factor['basis_ic_mean'] / df_factor['closeIndex_ic']) * 12
    df_factor['basis_if_pct_annual'] = (df_factor['basis_if_mean'] / df_factor['closeIndex_if']) * 12
    df_factor['basis_ih_pct_annual'] = (df_factor['basis_ih_mean'] / df_factor['closeIndex']) * 12

    df_factor['index_return_ic'] = df_factor['closeIndex_ic'].rolling(2).apply(
        lambda x: math.log(list(x)[-1] / list(x)[0]))
    df_factor['index_return_if'] = df_factor['closeIndex_if'].rolling(2).apply(
        lambda x: math.log(list(x)[-1] / list(x)[0]))
    df_factor['index_return'] = df_factor['closeIndex'].rolling(2).apply(lambda x: math.log(list(x)[-1] / list(x)[0]))
    df_factor['index_return_ic_win'] = df_factor['index_return_ic'].rolling(predict_windows).sum().shift(
        1 - predict_windows)
    df_factor['index_return_if_win'] = df_factor['index_return_if'].rolling(predict_windows).sum().shift(
        1 - predict_windows)
    df_factor['index_return_win'] = df_factor['index_return'].rolling(predict_windows).sum().shift(1 - predict_windows)
    df_factor['icf_index_return_diff_win'] = df_factor['index_return_ic_win'] - df_factor['index_return_if_win']
    df_factor['icf_basis_annual_diff'] = df_factor['basis_ic_pct_annual'] - df_factor['basis_if_pct_annual']
    lst_rolling_corr = list(
        df_factor[['basis_ic_pct_annual', 'basis_if_pct_annual']].rolling(lag_windows).corr().reset_index()[
            'basis_ic_pct_annual'])

    df_factor['icf_corr_win'] = lst_rolling_corr[1::2]
    df_factor[['basis_ic_pct_annual', 'basis_if_pct_annual', 'icf_corr_win']].rolling(lag_windows).corr()
    print(df_factor[['icf_index_return_diff_win', 'icf_basis_annual_diff', 'icf_corr_win']].corr())
    factor_cols = ['tradeDate', 'basis_ic_mean', 'basis_if_mean', 'basis_ic_pct_annual', 'basis_if_pct_annual',
                   'index_return_ic',
                   'index_return_if', 'index_return_ic_win',
                   'index_return_if_win', 'icf_index_return_diff_win', 'icf_basis_annual_diff', 'icf_corr_win']
    df_factor[factor_cols].to_csv('fut_index_factor_{0}_{1}.csv'.format(predict_windows, lag_windows), index=False)


def model_process(predict_windows=40, lag_windows=20):
    df_factor = pd.read_csv('fut_index_factor_{0}_{1}.csv'.format(predict_windows, lag_windows))
    df_factor = df_factor.dropna(axis=0)
    cols = list(df_factor.columns)
    trade_dates = list(df_factor['tradeDate'])
    cols.remove('index_return_ic_win')
    cols.remove('index_return_if_win')
    cols.remove('tradeDate')
    df_corr = df_factor[cols].corr()
    df_corr.index = list(df_corr.columns)
    df_corr.to_csv('corr_{0}_{1}.csv'.format(predict_windows, lag_windows))
    cols.remove('icf_index_return_diff_win')
    X = df_factor[cols]
    y = df_factor['icf_index_return_diff_win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

    with open('model_evaluation.txt', 'a') as f:
        print('Test results for pred_win:{0}, lag_win:{1}'.format(predict_windows, lag_windows))
        f.write('Test results for pred_win:{0}, lag_win:{1}\n'.format(predict_windows, lag_windows))
        lin_reg_model = LinearRegression()
        lin_reg_model.fit(X_train, y_train)
        print('intercept:', lin_reg_model.intercept_)
        f.write('coef_:{0}\n'.format(list(zip(cols, lin_reg_model.coef_))))
        print('coef_:{0}', list(zip(cols, lin_reg_model.coef_)))
        f.write('coef_:{0}\n'.format(list(zip(cols, lin_reg_model.coef_))))
        y_pred = lin_reg_model.predict(X_test)
        print('rmse:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 'r2:', metrics.r2_score(y_test, y_pred))
        f.write('rmse:{0}\n r2:{1}\n'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
                                             metrics.r2_score(y_test, y_pred)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(df_factor['icf_index_return_diff_win'],
             label="IC IF index log return diff in predict windows {0}".format(predict_windows))
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(df_factor['icf_basis_annual_diff'], 'r',
             label='IC IF annual basis pct diff in lag windows {0}'.format(lag_windows))
    ax2.legend()
    _idx_lst = list(range(df_factor.shape[0]))
    ax1.set_xticks(_idx_lst[::120])
    trade_date_str = [item.replace('-', '') for item in trade_dates]
    ax1.set_xticklabels(trade_date_str[::120], rotation=30)
    plt.savefig('icf return diff.jpg')


if __name__ == '__main__':
    # fetch_data('20150103', '20210902')
    predict_windows_lst = [60]
    lag_windows_lst = [120]
    for pred_win in predict_windows_lst:
        for lag_win in lag_windows_lst:
            factor_calculate(predict_windows=pred_win, lag_windows=lag_win)
            model_process(predict_windows=pred_win, lag_windows=lag_win)
