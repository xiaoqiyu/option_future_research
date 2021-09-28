#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/29 16:24
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : ModelProcess.py


import math
import uqer
import pprint
import numpy as np
from uqer import DataAPI
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from define import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
import talib as ta
import pandas as pd
import statsmodels.api as sm

uqer_client = uqer.Client(token="")


def _is_trading_time(time_str=''):
    if time_str > '09:00:00.000' and time_str <= '15:00:00.000' or time_str > '21:00:00.000' and time_str < '23:00:01.000':
        return True
    else:
        return False


def plot_factor(x=[], y=[], labels=[], tick_step=120):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, label="factor")
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(y, 'r', label='log return')
    ax2.legend()
    if labels:
        _idx_lst = list(range(len(x)))
        ax1.set_xticks(_idx_lst[::120])
        trade_date_str = [item.replace('-', '') for item in labels]
        ax1.set_xticklabels(trade_date_str[::tick_step], rotation=30)
    plt.show()


def get_factor(trade_date="20210701", predict_windows=1200, lag_long=600,
               lag_short=10,
               stop_profit=0.005, stop_loss=0.01, instrument_id='', exchange_cd=''):
    exchange_lst = ['dc', 'ine', 'sc', 'zc']
    _exchange_path = {'XZCE': 'zc', 'XSGE': 'sc', 'XSIE': 'ine', 'XDCE': 'dc'}.get(exchange_cd)
    tick_mkt = pd.read_csv(
        "C:\projects\l2mkt\FutAC_TickKZ_PanKou_Daily_202107\{0}\\{1}_{2}.csv".format(_exchange_path, instrument_id,
                                                                                     trade_date.replace('-', '')),
        encoding='gbk')

    tick_mkt.columns = tb_cols
    _flag = [_is_trading_time(item.split()[-1]) for item in list(tick_mkt['UpdateTime'])]
    tick_mkt = tick_mkt[_flag]
    tick_mkt['log_return'] = tick_mkt['LastPrice'].rolling(2).apply(lambda x: math.log(list(x)[-1] / list(x)[0]))
    tick_mkt['label_reg'] = tick_mkt['log_return'].rolling(predict_windows).sum().shift(1 - predict_windows)
    lst_ret_max = list(tick_mkt['label_reg'].rolling(predict_windows).max())
    lst_ret_min = list(tick_mkt['label_reg'].rolling(predict_windows).min())
    # 1: profit for long;2: profit for short; 0: not profit for the predict_windows, based on stop_profit threshold
    # tick_mkt['label_clf'] = [1 if item > math.log(1 + stop_profit) else 2 if item < math.log(1 - stop_profit) else 0 for
    #                          item in
    #                          tick_mkt['label_reg']]
    sum_max = []
    sum_min = []

    _x, _y = tick_mkt.shape

    label_cumsum = []

    for idx in range(_x):
        _cum_sum = tick_mkt['log_return'][idx:idx + predict_windows].cumsum()
        label_cumsum.append(list(_cum_sum)[-1])
        # label_cumsum.append(_cum_sum.max())
        sum_max.append(_cum_sum.max())
        sum_min.append(_cum_sum.min())

    tick_mkt['label_cumsum'] = label_cumsum
    tick_mkt['label_clf'] = [
        1 if item > math.log(1 + stop_profit) else 2 if sum_min[idx] < math.log(1 - stop_profit) else 0 for
        idx, item in enumerate(sum_max)]

    num_class1 = tick_mkt[tick_mkt.label_clf == 1].shape[0]
    num_class2 = tick_mkt[tick_mkt.label_clf == 2].shape[0]
    print('label num for 0, 1, 2 is:', _x - num_class1 - num_class2, num_class1, num_class2)
    _diff = list(tick_mkt['InterestDiff'])
    _vol = list(tick_mkt['Volume'])

    # factor calculate
    open_close_ratio = []
    for idx, item in enumerate(_diff):
        try:
            open_close_ratio.append(item / (_vol[idx] - item))
        except Exception as ex:
            open_close_ratio.append(open_close_ratio[-1])
            # print(idx, item, _vol[idx], ex)
    tick_mkt['open_close_ratio'] = open_close_ratio
    tick_mkt['buy_sell_spread'] = tick_mkt['BidPrice1'] - tick_mkt['AskPrice1']
    lst_bid_price = list(tick_mkt['BidPrice1'])
    lst_bid_vol = list(tick_mkt['BidVolume1'])
    lst_ask_price = list(tick_mkt['AskPrice1'])
    lst_ask_vol = list(tick_mkt['AskVolume1'])

    v_b = [0]
    v_a = [0]

    for i in range(1, _x):
        v_b.append(
            0 if lst_bid_price[i] < lst_bid_price[i - 1] else lst_bid_vol[i] - lst_bid_vol[i - 1] if lst_bid_price[i] ==
                                                                                                     lst_bid_price[
                                                                                                         i - 1] else
            lst_bid_vol[i])

        v_a.append(
            0 if lst_ask_price[i] < lst_ask_price[i - 1] else lst_ask_vol[i] - lst_ask_vol[i - 1] if lst_ask_price[i] ==
                                                                                                     lst_ask_price[
                                                                                                         i - 1] else
            lst_ask_vol[i])
    lst_oi = []
    lst_oir = []
    lst_aoi = []
    for idx, item in enumerate(v_a):
        lst_oi.append(v_b[idx] - item)
        lst_oir.append((v_b[idx] - item) / (v_b[idx] + item) if v_b[idx] + item != 0 else 0.0)
        lst_aoi.append((v_b[idx] - item) / (lst_ask_price[idx] - lst_bid_price[idx]))
    tick_mkt['oi'] = lst_oi
    tick_mkt['oir'] = lst_oir
    tick_mkt['aoi'] = lst_aoi
    tick_mkt['trend_short'] = tick_mkt['LastPrice'].rolling(lag_short).apply(
        lambda x: (list(x)[-1] - list(x)[0]) / lag_short)
    tick_mkt['trend_long'] = tick_mkt['LastPrice'].rolling(lag_long).apply(
        lambda x: (list(x)[-1] - list(x)[0]) / lag_long)
    tick_mkt['trenddiff'] = tick_mkt['trend_short'] - tick_mkt['trend_long']
    tick_mkt['trend_ls_ratio'] = tick_mkt['trend_short'] / tick_mkt['trend_short']
    tick_mkt['vwap'] = tick_mkt['Turnover'] / tick_mkt['Volume']
    tick_mkt['vol_short'] = tick_mkt['Volume'].rolling(lag_short).sum()
    tick_mkt['vol_long'] = tick_mkt['Volume'].rolling(lag_long).sum()
    tick_mkt['vol_ls_ratio'] = tick_mkt['vol_short'] / tick_mkt['vol_long']
    tick_mkt['turnover_short'] = tick_mkt['Turnover'].rolling(lag_short).sum()
    tick_mkt['turnover_long'] = tick_mkt['Turnover'].rolling(lag_long).sum()
    tick_mkt['turnover_ls_ratio'] = tick_mkt['turnover_short'] / tick_mkt['turnover_long']
    tick_mkt['vwap_short'] = tick_mkt['turnover_short'] / tick_mkt['vol_short']
    tick_mkt['vwap_long'] = tick_mkt['turnover_long'] / tick_mkt['vol_long']
    tick_mkt['vwap_ls_ratio'] = tick_mkt['vwap_short'] / tick_mkt['vwap_long']
    dif, dea, macd = ta.MACD(tick_mkt['LastPrice'], fastperiod=12, slowperiod=26, signalperiod=9)
    tick_mkt['macd'] = macd
    tick_mkt['dif'] = dif
    tick_mkt['dea'] = dea
    # plt.plot(dif[:100], c='green')
    # plt.plot(dea[:100], c='blue')
    # plt.plot(macd[:100], c='red')

    # tick_mkt[['dif', 'dea', 'macd']].plot(kind='bar', grid=True, figsize=(9, 7))
    # plt.show()
    tick_mkt = tick_mkt.dropna()
    # plot_factor(list(tick_mkt['macd'])[:100], list(tick_mkt['LastPrice'])[:100])
    # plot_factor(list(tick_mkt['trenddiff'])[:100], list(tick_mkt['LastPrice'])[:100])
    return tick_mkt[
        ['UpdateTime', 'oi', 'oir', 'aoi', 'label_clf', 'buy_sell_spread', 'trend_short', 'trend_long', 'trenddiff',
         'vol_ls_ratio', 'turnover_ls_ratio', 'vwap_ls_ratio', 'label_reg', 'dif', 'dea', 'macd', 'label_cumsum',
         'open_close_ratio', 'trend_ls_ratio']]


def train_model_reg(predict_windows=120,
                    lag_windows=60,
                    stop_profit=0.001, stop_loss=0.01, start_date='', end_date='', top_k_features=5, train_days=3,
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
    lin_reg_model = LinearRegression()
    print('start train model 1')
    for instrument_id, date, exchange_cd in train_dates.values:
        print("train for trade date:", date)
        df_factor = get_factor(trade_date=date,
                               predict_windows=predict_windows,
                               lag_long=lag_windows,
                               stop_profit=stop_profit, stop_loss=stop_loss, instrument_id=instrument_id,
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
        df_corr.to_csv('corr_reg_{0}.csv'.format(date))
    _scores = [item / (train_days - 1) for item in list(acc_scores.values())]
    _features = list(acc_scores.keys())
    _tmp = list(zip(_features, _scores))
    _tmp = sorted(_tmp, key=lambda x: x[1], reverse=True)
    print("top features", _tmp[:top_k_features])
    top_feature_lst = [item[0] for item in _tmp[:top_k_features]]
    for df_factor in factor_lst:
        lin_reg_model.fit(df_factor[top_feature_lst], df_factor['label_cumsum'])
        del df_factor

        # print('intercept:', lin_reg_model.intercept_)
        # print('coef_:{0}\n'.format(list(zip(top_feature_lst, lin_reg_model.coef_))))

    print('test for trade date:', test_dates[1])
    df_factor = get_factor(trade_date=test_dates[1],
                           predict_windows=predict_windows,
                           lag_long=lag_windows,
                           stop_profit=stop_profit, stop_loss=stop_loss, instrument_id=test_dates[0],
                           exchange_cd=test_dates[2])
    # cols = list(df_factor.columns)
    # cols.remove('label_clf')
    _update_time = list(df_factor['UpdateTime'])
    y_pred = lin_reg_model.predict(df_factor[top_feature_lst])
    ret_str = 'rmse:{0}, r2:{1}, date:{2},predict windows:{3}, lag windows:{4}\n'.format(
        np.sqrt(metrics.mean_squared_error(df_factor['label_cumsum'], y_pred)),
        metrics.r2_score(df_factor['label_cumsum'], y_pred), test_dates[1], predict_windows, lag_windows)
    del df_factor
    with open('summary.txt', 'a') as f:
        f.write(ret_str)
    # print('rmse:', np.sqrt(metrics.mean_squared_error(df_factor['label_cumsum'], y_pred)), 'r2:',
    #       metrics.r2_score(df_factor['label_cumsum'], y_pred))
    df_pred = pd.DataFrame({'UpdateTime': df_factor['UpdateTime'], 'pred': y_pred})
    df_pred.to_csv('df_pred_{0}_{1}_{2}.csv'.format(date, predict_windows, lag_windows), index=False)
    # plt.plot(y_pred)
    # plt.plot(df_factor['label_cumsum'])
    # plt.show()
    # f.write('rmse:{0}\n r2:{1}\n'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    #                                      metrics.r2_score(y_test, y_pred)))


def train_model_clf(predict_windows=120,
                    lag_windows=60,
                    stop_profit=0.001, stop_loss=0.01, start_date='', end_date='', top_k_features=1, train_days=3):
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
        print("train for trade date:", date)
        df_factor = get_factor(trade_date=date,
                               predict_windows=predict_windows,
                               lag_long=lag_windows,
                               stop_profit=stop_profit, stop_loss=stop_loss, instrument_id=instrument_id)

        cols = list(df_factor.columns)
        cols.remove('label_clf')
        cols.remove('UpdateTime')
        model1.fit(df_factor[cols], df_factor['label_clf'])
        factor_lst.append(df_factor)
        print("model selection scores for mutual information:", list(zip(cols, model1.scores_)))
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
                print(idx, ex, cols)
        # print("model selection pvalues for mutual information:", model1.pvalues_)
    sorted_scores = sorted(acc_scores.items(), reverse=True, key=lambda x: x[1])[-top_k_features:]
    sorted_features = [item[0] for item in sorted_scores]
    print(sorted_scores)

    for df_factor in factor_lst:
        model2.fit(df_factor[sorted_features], df_factor['label_clf'])
    print('test for trade date:', test_dates[1])

    df_factor = get_factor(trade_date=test_dates[1],
                           predict_windows=predict_windows,
                           lag_long=lag_windows,
                           stop_profit=stop_profit, stop_loss=stop_loss, instrument_id=test_dates[0])
    # cols = list(df_factor.columns)
    # cols.remove('label_clf')
    _update_time = list(df_factor['UpdateTime'])
    _update_time_str = [item.split()[-1] for item in _update_time]
    y_pred = model2.predict(df_factor[sorted_features])
    # print("model scores:----------", model1.scores_)
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

    print('true vs pred:', '(0,{},{},{})'.format(true0, pred0, num_prec0 / pred0 if pred0 > 0 else 0))
    print('true vs pred:', '(1,{},{},{})'.format(true1, pred1, num_prec1 / pred1 if pred1 > 0 else 0))
    print('true vs pred:', '(2,{},{},{})'.format(true2, pred2, num_prec2 / pred2 if pred2 > 0 else 0))

    # score = metrics.precision_score(y_pred, df_factor['label_clf'], average='macro')
    # print("precision macro score:", score)
    # score = metrics.precision_score(y_pred, df_factor['label_clf'], average='micro')
    # print("precision micro score:", score)
    # score = metrics.recall_score(y_pred, df_factor['label_clf'], average='macro')
    # print("recall macro score:", score)
    # score = metrics.recall_score(y_pred, df_factor['label_clf'], average='micro')
    # print("recall micro score:", score)
    # score = metrics.f1_score(y_pred, df_factor['label_clf'], average='macro')
    # print("f1 macro score", score)
    # score = metrics.f1_score(y_pred, df_factor['label_clf'], average='micro')
    # print("f1 micro score", score)
    df_pred = pd.DataFrame({'pred': y_pred, 'true': df_factor['label_clf'], 'UpdateTime': _update_time_str})
    df_pred.to_csv('df_pred.csv')


def factor_process(product_id='m', trade_date='20210105', instrument_id='m2105', windows_len=1000, stop_profit=3.0):
    df = pd.read_csv('cache/factors/factor_{0}_{1}.csv'.format(instrument_id, trade_date))
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
                    stop_profit=0.001, stop_loss=0.01, start_date='', end_date='', top_k_features=5, train_days=3,
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
    # lin_reg_model_ols = LinearRegression()
    for instrument_id, date, exchange_cd in train_dates.values:
        print("train for trade date:", date)
        df_factor = get_factor(trade_date=date,
                               predict_windows=predict_windows,
                               lag_long=lag_windows,
                               stop_profit=stop_profit, stop_loss=stop_loss, instrument_id=instrument_id,
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
        df_corr.to_csv('corr_reg_{0}_{1}_{2}.csv'.format(date, predict_windows, lag_windows))
    _scores = [item / (train_days - 1) for item in list(acc_scores.values())]
    _features = list(acc_scores.keys())
    _tmp = list(zip(_features, _scores))
    _tmp = sorted(_tmp, key=lambda x: x[1], reverse=True)
    print("top features", _tmp[:top_k_features])
    top_feature_lst = [item[0] for item in _tmp[:top_k_features]]

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


def get_trade_dates(start_date='20110920', end_date='20210921'):
    df = DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE", beginDate=start_date, endDate=end_date, isOpen=u"1", field=u"",
                             pandas="1")
    df = df[df.isOpen == 1]
    return df


if __name__ == '__main__':
    # factor_process(windows_len=600, stop_profit=5.0, trade_date='20210106')
    # get_factor()
    start_date = '20210701'
    end_date = '20210730'
    trade_date_df = get_trade_dates(start_date=start_date, end_date=end_date)
    trade_date_df = trade_date_df[trade_date_df.exchangeCD == 'XSHE']
    trade_dates = list(trade_date_df['calendarDate'])
    train_days = 3
    num_date = len(trade_dates)
    idx = 0
    # [20, 60, 120, 600, 1200]
    predict_window_lst = [20, 60, 120, 600, 1200]  # 10s, 30s,1min,5min,10min
    lag_window_lst = [20, 60, 120, 600]  # 10s, 30s,1min,5min,
    for predict_win in predict_window_lst:
        for lag_win in lag_window_lst:
            print('train for predict windows:{0} and lag_windows:{1}-----------'.format(predict_win, lag_win))
            while idx < num_date - train_days:
                train_model_reg(predict_windows=predict_win,
                                lag_windows=lag_win,
                                stop_profit=0.001, stop_loss=0.01, top_k_features=5, start_date=trade_dates[idx],
                                end_date=trade_dates[idx + train_days],
                                train_days=train_days, product_id='RB')
                idx += train_days
