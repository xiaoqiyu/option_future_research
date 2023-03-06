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
        return []
    else:
        try:
            return [float(item) / _normalized_base for item in x]
        except Exception as ex:
            print('error for norm cal for col:{0} with error:{1}'.format(key, ex))
            return []


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


def cal_oir(bid_price: list, bid_vol: list, ask_price: list, ask_vol: list, n_rows: int) -> tuple:
    """
    calculate order imbalance factors
    :param bid_price:
    :param bid_vol:
    :param ask_price:
    :param ask_vol:
    :param n_rows:
    :return:
    """
    v_b = [0]
    v_a = [0]

    for i in range(1, n_rows):
        v_b.append(
            0 if bid_price[i] < bid_price[i - 1] else bid_vol[i] - bid_vol[i - 1] if bid_price[i] == bid_price[
                i - 1] else bid_vol[i])

        v_a.append(
            0 if ask_price[i] < ask_price[i - 1] else ask_vol[i] - ask_vol[i - 1] if ask_price[i] == ask_price[
                i - 1] else ask_vol[i])
    lst_oi = []
    lst_oir = []
    lst_aoi = []
    for idx, item in enumerate(v_a):
        lst_oi.append(v_b[idx] - item)
        lst_oir.append((v_b[idx] - item) / (v_b[idx] + item) if v_b[idx] + item != 0 else 0.0)
        lst_aoi.append((v_b[idx] - item) / (ask_price[idx] - bid_price[idx]))
    return lst_oi, lst_oir, lst_aoi


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


def get_factor(trade_date: str = "20210701", predict_windows: list = [1200], lag_windows: list = [10, 600],
               instrument_id: str = '', exchange_cd: str = '') -> pd.DataFrame:
    """

    :param trade_date:
    :param predict_windows:
    :param lag_long:
    :param lag_short:
    :param stop_profit:
    :param stop_loss:
    :param instrument_id:
    :param exchange_cd:
    :return:
    """
    _mul_num = utils.get_mul_num(instrument_id) or 1
    print(instrument_id, trade_date, define.exchange_map, exchange_cd, define.TICK_MODEL_DIR)
    _tick_mkt_path = os.path.join(define.TICK_MKT_DIR, define.exchange_map.get(exchange_cd),
                                  '{0}_{1}.csv'.format(instrument_id, trade_date.replace('-', '')))
    tick_mkt = pd.read_csv(_tick_mkt_path, encoding='gbk')
    tick_mkt.columns = define.tb_cols

    # filter trading timestamp
    _flag = [utils._is_trading_time(item.split()[-1]) for item in list(tick_mkt['UpdateTime'])]
    tick_mkt = tick_mkt[_flag]

    # for price/volume, if not define the normorlized base or defined as 0, we will ret it as the first tick
    open_dict = tick_mkt.head(1).to_dict('record')[0]

    tick_mkt['vwap'] = (tick_mkt['Turnover'] / tick_mkt['Volume']) / _mul_num
    tick_mkt['wap'] = (tick_mkt['BidPrice1'] * tick_mkt['AskVolume1'] + tick_mkt['AskPrice1'] * tick_mkt[
        'BidVolume1']) / (tick_mkt['AskVolume1'] + tick_mkt['BidVolume1'])
    # tick_mkt['log_return'] = tick_mkt['LastPrice'].rolling(2).apply(lambda x: math.log(list(x)[-1] / list(x)[0]))
    tick_mkt['log_return'] = np.log(tick_mkt['LastPrice']).diff()
    tick_mkt['wap_log_return'] = np.log(tick_mkt['wap']) - np.log(tick_mkt['LastPrice'])

    # tick_mkt['log_return'] = tick_mkt['wap'].rolling(2).apply(lambda x: math.log(list(x)[-1] / list(x)[0]))

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
    _diff = list(tick_mkt['InterestDiff'])
    _vol = list(tick_mkt['Volume'])

    open_close_ratio = []
    for idx, item in enumerate(_diff):
        try:
            open_close_ratio.append(item / (_vol[idx] - item))
        except Exception as ex:
            open_close_ratio.append(open_close_ratio[-1])

    # cal ori factor(paper factor)
    lst_oi, lst_oir, lst_aoi = cal_oir(list(tick_mkt['BidPrice1']), list(tick_mkt['BidVolume1']),
                                       list(tick_mkt['AskPrice1']), list(tick_mkt['AskVolume1']), n_rows=_x)

    # cal cos factor(market factor)
    _lst_last_price = list(tick_mkt['LastPrice'])
    _lst_turn_idx, _lst_turn_val = cal_turning(_lst_last_price)
    _lst_slope = cal_slope(_lst_last_price, _lst_turn_idx, _lst_turn_val)
    _lst_cos = cal_cos(_lst_last_price, _lst_turn_idx, _lst_turn_val)

    dif, dea, macd = ta.MACD(tick_mkt['LastPrice'], fastperiod=12, slowperiod=26, signalperiod=9)

    tick_mkt['open_close_ratio'] = open_close_ratio
    tick_mkt['price_spread'] = (tick_mkt['BidPrice1'] - tick_mkt['AskPrice1']) / (
            tick_mkt['BidPrice1'] + tick_mkt['AskPrice1'] / 2)
    tick_mkt['buy_sell_spread'] = abs(tick_mkt['BidPrice1'] - tick_mkt['AskPrice1'])
    tick_mkt['oi'] = lst_oi
    tick_mkt['oir'] = lst_oir
    tick_mkt['aoi'] = lst_aoi
    tick_mkt['slope'] = _lst_slope
    tick_mkt['cos'] = _lst_cos
    tick_mkt['macd'] = macd
    tick_mkt['dif'] = dif
    tick_mkt['dea'] = dea
    tick_mkt['bs_tag'] = tick_mkt['LastPrice'].rolling(2).apply(lambda x: 1 if list(x)[-1] > list(x)[0] else -1)
    tick_mkt['bs_vol'] = tick_mkt['bs_tag'] * tick_mkt['Volume']

    def _cal_clf_label(x):
        # FIXME remove clf bc hardcode, log(1.001)= 0.0009988(0.1%), log(0.999)=-0.001(0.1%)
        # return [1 if item > 0.001 else 2 if item < -0.001 else 0 for item in x]
        # return [1 if item > 2 else 2 if item < -2 else 0 for item in x]
        # ret = []
        # for item in x:
        #     if item > 5:
        #         ret.append(2)
        #     elif item < 5 and item >= 2:
        #         ret.append(1)
        #     elif item < 2 and item > -2:
        #         ret.append(0)
        #     elif item <= -2 and item > -5:
        #         ret.append(-1)
        #     else:
        #         ret.append(-2)
        # return ret
        return [1 if item > 2 else -1 if item < -2 else 0 for item in x]

    # agg by windows,agg_dict={factor_name:(executor, new_factor_name)}, if None for new_factor_name, then use the
    # raw name by default
    windows_agg_dict = {
        'LastPrice': [(lambda x: (list(x)[-1] - list(x)[0]) / len(x), 'trend')],
        'Volume': [(np.sum, 'volume')],
        'Turnover': [(np.sum, 'turnover')],
        'bs_vol': [(np.sum, None)],
        'log_return': [(np.sum, None)],
    }
    for idx, _lag in enumerate(lag_windows):
        for key, val in windows_agg_dict.items():
            for _func, rename in val:
                factor_name = rename or key
                tick_mkt['{0}_{1}'.format(factor_name, idx)] = list(tick_mkt[key].rolling(_lag).apply(_func))

        tick_mkt['bsvol_volume_{0}'.format(idx)] = tick_mkt['bs_vol_{0}'.format(idx)] / tick_mkt[
            'volume_{0}'.format(idx)]

    for idx, _pred in enumerate(predict_windows):
        tick_mkt['label_{0}'.format(idx)] = tick_mkt['log_return'].rolling(_pred).sum().shift(1 - _pred)
        tick_mkt['rv_{0}'.format(idx)] = tick_mkt['log_return'].rolling(_pred).apply(realized_volatility).shift(
            1 - _pred)
        # tick_mkt['label_clf_{0}'.format(idx)] = _cal_clf_label(tick_mkt['label_{0}'.format(idx)])

        tick_mkt['price_chg_{0}'.format(idx)] = tick_mkt['LastPrice'].rolling(_pred).apply(
            lambda x: list(x)[-1] - list(x)[0]).shift(1 - _pred)
        tick_mkt['vwap_chg_{0}'.format(idx)] = tick_mkt['LastPrice'].rolling(_pred).apply(
            lambda x: x.mean() - list(x)[0]).shift(1 - _pred)
        tick_mkt['label_clf_{0}'.format(idx)] = _cal_clf_label(tick_mkt['vwap_chg_{0}'.format(idx)])

    # tick_mkt['log_return_short'] = tick_mkt['log_return'].rolling(lag_short).sum()
    # tick_mkt['log_return_long'] = tick_mkt['log_return'].rolling(lag_long).sum()
    # tick_mkt['trend_short'] = tick_mkt['LastPrice'].rolling(lag_short).apply(
    #     lambda x: (list(x)[-1] - list(x)[0]) / lag_short)
    # tick_mkt['trend_long'] = tick_mkt['LastPrice'].rolling(lag_long).apply(
    #     lambda x: (list(x)[-1] - list(x)[0]) / lag_long)
    # tick_mkt['vol_short'] = tick_mkt['Volume'].rolling(lag_short).sum()
    # tick_mkt['vol_long'] = tick_mkt['Volume'].rolling(lag_long).sum()
    # tick_mkt['turnover_short'] = tick_mkt['Turnover'].rolling(lag_short).sum()
    # tick_mkt['turnover_long'] = tick_mkt['Turnover'].rolling(lag_long).sum()
    # tick_mkt['bs_vol_long'] = tick_mkt['bs_vol'].rolling(lag_long).sum()
    # tick_mkt['bs_vol_short'] = tick_mkt['bs_vol'].rolling(lag_short).sum()

    # agg by factor,remove vwap
    factor_agg_dict = {'trend': ['minus', 'divide'], 'volume': ['minus', 'divide'], 'turnover': ['minus', 'divide'],
                       'bs_vol': ['minus', 'divide'], 'bsvol_volume': ['divide']}
    lag_win_size = len(lag_windows)
    if lag_win_size >= 2:
        for key, val in factor_agg_dict.items():
            for _func_name in val:
                if _func_name == 'minus':
                    tick_mkt['{0}_ls_diff'.format(key)] = tick_mkt['{0}_{1}'.format(key, 0)] - tick_mkt[
                        '{0}_{1}'.format(key, lag_win_size - 1)]
                if _func_name == 'divide':
                    tick_mkt['{0}_ls_ratio'.format(key)] = tick_mkt['{0}_{1}'.format(key, 0)] / tick_mkt[
                        '{0}_{1}'.format(key, lag_win_size - 1)]

    # tick_mkt['trenddiff'] = tick_mkt['trend_short'] - tick_mkt['trend_long']
    # tick_mkt['trend_ls_ratio'] = tick_mkt['trend_short'] / tick_mkt['trend_long']
    # tick_mkt['vol_ls_ratio'] = tick_mkt['vol_short'] / tick_mkt['vol_long']
    # tick_mkt['turnover_ls_ratio'] = tick_mkt['turnover_short'] / tick_mkt['turnover_long']
    # tick_mkt['vwap_short'] = tick_mkt['turnover_short'] / tick_mkt['vol_short']
    # tick_mkt['vwap_long'] = tick_mkt['turnover_long'] / tick_mkt['vol_long']
    # tick_mkt['vwap_ls_ratio'] = tick_mkt['vwap_short'] / tick_mkt['vwap_long']
    # tick_mkt['bs_vol_diff'] = tick_mkt['bs_vol_short'] - tick_mkt['bs_vol_long']
    # tick_mkt['bs_vol_ls_ratio'] = tick_mkt['bs_vol_short'] / tick_mkt['bs_vol_long']

    # TODO ADD norm factor
    # norm factor processing, calculate the norm here or in model processing. norm with the value of the current date
    # or summary of the prev date
    normalized = define.normalized_vals
    # TODO open_dict is the open value of the current date, this could be normalized by other ref,e.g. the prev date
    for k, v in normalized.items():
        if not v:
            normalized.update({k: (open_dict.get(k) or 1.0)})
    for _base_col, _ref_col in define.normalized_refs:
        _arr = _normalized_by_base(tick_mkt[_base_col], normalized, _ref_col)
        if not _arr:  # FIXME
            continue
        tick_mkt['norm_{0}'.format(_base_col.lower())] = _arr

    _factor_lst = list(tick_mkt.columns)
    print(_factor_lst)

    for col in define.skip_raw_cols:
        try:
            _factor_lst.remove(col)
        except Exception as ex:
            print('col:{0} not exist with error:{1}'.format(col, ex))

    # FIXME replace this with better solution
    for idx, _lag in enumerate(lag_windows):
        _factor_lst.remove('volume_{0}'.format(idx))
        _factor_lst.remove('turnover_{0}'.format(idx))
        _factor_lst.remove('bs_vol_{0}'.format(idx))

    # handle exception values
    tick_mkt = tick_mkt.replace(np.inf, np.nan)
    tick_mkt = tick_mkt.replace(-np.inf, np.nan)
    tick_mkt = tick_mkt.dropna()
    _factor_path = utils.get_path([define.CACHE_DIR, define.FACTOR_DIR,
                                   'factor_{0}_{1}.csv'.format(instrument_id, trade_date.replace('-', ''))])
    tick_mkt[_factor_lst].to_csv(_factor_path, index=False)
    return tick_mkt[_factor_lst]
