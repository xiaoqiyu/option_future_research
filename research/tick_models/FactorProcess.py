#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/25 14:11
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : FactorProcess.py

import math
import numpy as np
import talib as ta
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt

import utils.define as define
import utils.utils as utils

logging.basicConfig(filename='logs/{0}.txt'.format(os.path.split(__file__)[-1].split('.')[0]), level=logging.DEBUG)
logger = logging.getLogger()


def _normalized_by_base(x: list, normalized_dict: dict, key: str) -> list:
    """

    :param x:
    :param normalized_dict:
    :param key:
    :return:
    """
    _normalized_base = normalized_dict.get(key)
    if not _normalized_base:
        return x
    else:
        return [item / _normalized_base for item in x]


def _cal_turning_item(x: list) -> tuple:
    if not x:
        return 0, np.nan
    if len(x) == 1:
        return 0, np.nan
    if len(x) == 2:
        return -1, x[0]
    try:
        if (x[-1] - x[-2]) * (x[-2] - x[-3]) > 0:
            return -2, x[0]
        else:
            return -1, x[1]
    except Exception as ex:
        raise ValueError("Error to cal turning with error:{0}".format(ex))


def log_return(x):
    return np.log(x).diff()


def realized_volatility(x):
    return np.sqrt(np.sum(x ** 2))


def count_unique(x):
    return len(np.unique(x))


def cal_turning(x: list) -> tuple:
    idx_lst = []
    val_lst = []
    if len(x) == 1:
        idx_lst.append(0)
        val_lst.append(np.nan)
    elif len(x) == 2:
        _idx, _val = _cal_turning_item(x)
        return [0], [_val]
    else:
        _len = len(x)
        idx_lst.append(0)
        val_lst.append(np.nan)
        idx_lst.append(0)
        val_lst.append(x[0])
        _idx, _val = _cal_turning_item(x)
        idx_lst.append(2 + _idx)
        val_lst.append(_val)
        for idx in range(3, _len, 1):
            _idx, _val = _cal_turning_item(x[idx - 3:idx])
            idx_lst.append(idx + _idx)
            val_lst.append(_val)
    return idx_lst, val_lst


def cal_slope(x: list, turn_idx: list, turn_val: list) -> list:
    assert len(x) == len(turn_idx)
    assert len(x) == len(turn_val)
    ret = []
    for idx, item in enumerate(x):
        if idx == 0:
            ret.append(np.nan)
            continue
        ret.append((x[idx] - turn_val[idx]) / (idx - turn_idx[idx]))
    return ret


def cal_cos(x: list, turn_idx: list, turn_val: list) -> list:
    assert len(x) == len(turn_idx)
    assert len(x) == len(turn_val)
    ret = []
    for idx, item in enumerate(x):
        if idx == 0:
            ret.append(0)
            continue
        a = np.array([idx - turn_idx[idx], item - turn_val[idx]])
        b = np.array([0, 1])
        ret.append(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return ret


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





def get_factor(trade_date: str = "20210701", predict_windows: int = 1200, lag_long: int = 600,
               lag_short: int = 10, instrument_id: str = '', exchange_cd: str = '',
               normalized: dict = {'LastPrice': 0, 'Volume': 0, 'OpenInterest': 0}) -> pd.DataFrame:
    """

    :param trade_date:
    :param predict_windows:
    :param lag_long:
    :param lag_short:
    :param stop_profit:
    :param stop_loss:
    :param instrument_id:
    :param exchange_cd:
    :param normalized:
    :return:
    """
    _mul_num = utils.get_mul_num(instrument_id) or 1
    _tick_mkt_path = os.path.join(define.TICK_MKT_DIR, define.exchange_map.get(exchange_cd),
                                  '{0}_{1}.csv'.format(instrument_id, trade_date.replace('-', '')))
    tick_mkt = pd.read_csv(_tick_mkt_path, encoding='gbk')
    tick_mkt.columns = define.tb_cols

    # filter trading timestamp
    _flag = [utils._is_trading_time(item.split()[-1]) for item in list(tick_mkt['UpdateTime'])]
    tick_mkt = tick_mkt[_flag]

    # for price/volume, if not define the normorlized base or defined as 0, we will ret it as the first tick
    open_dict = tick_mkt.head(1).to_dict('record')[0]
    for k, v in normalized.items():
        if not v:
            normalized.update({k: (open_dict.get(k) or 1.0)})

    tick_mkt['vwap'] = (tick_mkt['Turnover'] / tick_mkt['Volume']) / _mul_num
    tick_mkt['wap'] = (tick_mkt['BidPrice1'] * tick_mkt['AskVolume1'] + tick_mkt['AskPrice1'] * tick_mkt[
        'BidVolume1']) / (tick_mkt['AskVolume1'] + tick_mkt['BidVolume1'])
    tick_mkt['log_return'] = tick_mkt['LastPrice'].rolling(2).apply(lambda x: math.log(list(x)[-1] / list(x)[0]))
    # tick_mkt['log_return'] = tick_mkt['wap'].rolling(2).apply(lambda x: math.log(list(x)[-1] / list(x)[0]))
    tick_mkt['log_return_short'] = tick_mkt['log_return'].rolling(lag_short).sum()
    tick_mkt['log_return_long'] = tick_mkt['log_return'].rolling(lag_long).sum()
    # tick_mkt['label_reg'] = tick_mkt['log_return'].rolling(predict_windows).sum().shift(1 - predict_windows)
    # lst_ret_max = list(tick_mkt['label_reg'].rolling(predict_windows).max())
    # lst_ret_min = list(tick_mkt['label_reg'].rolling(predict_windows).min())
    # 1: profit for long;2: profit for short; 0: not profit for the predict_windows, based on stop_profit threshold
    # tick_mkt['label_clf'] = [1 if item > math.log(1 + stop_profit) else 2 if item < math.log(1 - stop_profit) else 0 for
    #                          item in
    #                          tick_mkt['label_reg']]
    sum_max = []
    sum_min = []

    _x, _y = tick_mkt.shape
    label_cumsum = []

    for idx in range(_x):
        _cum_sum = tick_mkt['log_return'][idx:idx + predict_windows].cumsum()  # idx+1, 避免未来数据
        label_cumsum.append(list(_cum_sum)[-1])
        # label_cumsum.append(_cum_sum.max())
        # label_cumsum.append(_cum_sum.mean())

        sum_max.append(_cum_sum.max())
        sum_min.append(_cum_sum.min())

    tick_mkt['label_cumsum'] = label_cumsum
    # uncomment this if for clf models
    # tick_mkt['label_clf'] = [
    #     1 if item > math.log(1 + stop_profit) else 2 if sum_min[idx] < math.log(1 - stop_profit) else 0 for
    #     idx, item in enumerate(sum_max)]
    #
    # num_class1 = tick_mkt[tick_mkt.label_clf == 1].shape[0]
    # num_class2 = tick_mkt[tick_mkt.label_clf == 2].shape[0]
    # print('label num for 0, 1, 2 is:', _x - num_class1 - num_class2, num_class1, num_class2)
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
    tick_mkt['price_spread'] = (tick_mkt['BidPrice1'] - tick_mkt['AskPrice1']) / (
                tick_mkt['BidPrice1'] + tick_mkt['AskPrice1'] / 2)
    tick_mkt['buy_sell_spread'] = abs(tick_mkt['BidPrice1'] - tick_mkt['AskPrice1'])
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

    _lst_last_price = list(tick_mkt['LastPrice'])
    _lst_turn_idx, _lst_turn_val = cal_turning(_lst_last_price)
    _lst_slope = cal_slope(_lst_last_price, _lst_turn_idx, _lst_turn_val)
    _lst_cos = cal_cos(_lst_last_price, _lst_turn_idx, _lst_turn_val)

    # plot_factor(_lst_turn_val[1:], _lst_last_price[1:])
    # plot_factor(_lst_cos[1:100], list(tick_mkt['log_return'][1:100])[1:])

    tick_mkt['slope'] = _lst_slope
    tick_mkt['cos'] = _lst_cos

    tick_mkt['trend_short'] = tick_mkt['LastPrice'].rolling(lag_short).apply(
        lambda x: (list(x)[-1] - list(x)[0]) / lag_short)
    tick_mkt['trend_long'] = tick_mkt['LastPrice'].rolling(lag_long).apply(
        lambda x: (list(x)[-1] - list(x)[0]) / lag_long)
    tick_mkt['trenddiff'] = tick_mkt['trend_short'] - tick_mkt['trend_long']
    # TODO check calculation of trend_ls_ratio, comment first, update later
    tick_mkt['trend_ls_ratio'] = tick_mkt['trend_short'] / tick_mkt['trend_long']
    # for idx, item in enumerate(list(tick_mkt['trend_ls_ratio'])):
    #     if item == np.inf or item == -np.inf:
    #         print(idx, item)

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
    tick_mkt['bs_tag'] = tick_mkt['LastPrice'].rolling(2).apply(lambda x: 1 if list(x)[-1] > list(x)[0] else -1)
    tick_mkt['bs_vol'] = tick_mkt['bs_tag'] * tick_mkt['Volume']
    tick_mkt['bs_vol_long'] = tick_mkt['bs_vol'].rolling(lag_long).sum()
    tick_mkt['bs_vol_short'] = tick_mkt['bs_vol'].rolling(lag_short).sum()
    tick_mkt['bs_vol_diff'] = tick_mkt['bs_vol_short'] - tick_mkt['bs_vol_long']
    tick_mkt['bs_vol_ls_ratio'] = tick_mkt['bs_vol_short'] / tick_mkt['bs_vol_long']
    tick_mkt['bs2vol_ratio_short'] = tick_mkt['bs_vol_short'] / tick_mkt['vol_short']
    tick_mkt['bs2vol_ratio_long'] = tick_mkt['bs_vol_long'] / tick_mkt['vol_long']
    tick_mkt['bs2vol_ls_ratio'] = tick_mkt['bs2vol_ratio_short'] / tick_mkt['bs2vol_ratio_long']

    for _base_col, _ref_col in define.normalized_cols:
        _arr = _normalized_by_base(tick_mkt[_base_col], normalized, _ref_col)
        tick_mkt['norm_{0}'.format(_base_col)] = _arr
    _factor_lst = list(tick_mkt.columns)
    logger.info(_factor_lst)

    for col in define.skip_raw_cols:
        try:
            _factor_lst.remove(col)
        except Exception as ex:
            logger.info('col:{0} not exist with error:{1}'.format(col, ex))
    tick_mkt = tick_mkt.replace(np.inf, np.nan)
    tick_mkt = tick_mkt.replace(-np.inf, np.nan)
    tick_mkt = tick_mkt.dropna()
    # # FIXME  hardcode some features
    # _factor_lst = ['macd', 'log_return', 'aoi', 'norm_wap', 'trenddiff', 'UpdateTime', 'slope', 'cos', 'label_cumsum']
    return tick_mkt[_factor_lst]
