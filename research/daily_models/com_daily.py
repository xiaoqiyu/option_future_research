#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/14 14:19
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : com_daily.py

import math
import uqer
import datetime
import pprint
import pandas as pd
import numpy as np
from uqer import DataAPI
import matplotlib.pyplot as plt
import backtester.Account as Account
import backtester.Position as Position
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import utils.define as define

uqer_client = uqer.Client(token="6aa0df8d4eec296e0c25fac407b332449112aad6c717b1ada315560e9aa0a311")

margin_overwrite = {'p': 11}

['P', 'rb', 'm', 'eg', 'fu']

open_fee_overwrite = {'p': 2.5, 'rb': 5.08, 'm': 2.5}
close_fee_overwrite = {'p': 2.5, 'rb': 5.08, 'm': 2.5}  # rb is close today


def select_product(product_lst: list = ['rb', 'm'], start_date: str = '', end_date: str = '',
                   trade_date: str = '') -> pd.DataFrame:
    sum_lst = []
    for pid in product_lst:
        df = DataAPI.MktMFutdGet(mainCon=u"1", contractMark=u"", contractObject=pid, tradeDate=u"",
                                 startDate=start_date,
                                 endDate=end_date, field=u"", pandas="1")
        trade_dates = list(df['tradeDate'])[-5:]
        vol_std_lst = []
        instrument_id_lst = list(df['ticker'])[-5:]

        for idx, trade_date in enumerate(trade_dates):
            df_intraday = DataAPI.FutBarHistOneDay2Get(instrumentID=instrument_id_lst[idx], date=trade_date, unit=u"1",
                                                       field=u"",
                                                       pandas="1")
            # print(instrument_id_lst[idx], trade_date)
            vol_std_lst.append(df_intraday['closePrice'].std())
        # print(vol_std_lst)
        _liquid = df['turnoverVol'].mean()
        _vol = sum(vol_std_lst) / len(vol_std_lst)
        # print(pid, _liquid, _vol)
        _fee = (open_fee_overwrite.get(pid.lower()) or 0.0) + (close_fee_overwrite.get(pid.lower()) or 0.0)
        sum_lst.append([pid, _liquid, _vol, _fee])
    df_sum = pd.DataFrame(sum_lst, columns=['product_id', 'liquid', 'vol', 'fee'])
    df_sum['liquid_rank'] = df_sum['liquid'].rank()
    df_sum['vol_rank'] = df_sum['vol'].rank()
    df_sum['score'] = df_sum['liquid_rank'] + df_sum['vol_rank']
    df_sum = df_sum.sort_values(by='score', ascending=False)
    print(df_sum)


def get_trend_signal(df_mkt: pd.DataFrame, k1: float = 0.2, k2: float = 0.2, std_weight=0.2, sod: bool = True):
    # ['secID', 'ticker', 'exchangeCD', 'secShortName', 'secShortNameEN',
    #  'tradeDate', 'contractObject', 'contractMark', 'preSettlePrice',
    #  'preClosePrice', 'openPrice', 'highestPrice', 'lowestPrice',
    #  'settlePrice', 'closePrice', 'turnoverVol', 'turnoverValue', 'openInt',
    #  'chg', 'chg1', 'chgPct', 'mainCon', 'smainCon']
    # df_mkt = df_mkt.dropna()
    sig = []
    upbound = []
    lowerbound = []
    ref_price = 'preClosePrice' if sod else 'closePrice'  # FIXME should be open price
    for item in df_mkt[[ref_price, 'hh', 'lc', 'hc', 'll', 'accstd']].values:
        _range = max(item[1] - item[2], item[3] - item[4])
        upbound.append(item[0] + k1 * _range + std_weight * item[5])
        lowerbound.append(item[0] - k2 * _range - std_weight * item[5])
        if item[-1] > item[0] + k1 * _range:
            sig.append(define.LONG)
        elif item[-1] < item[0] - k2 * _range:
            sig.append(define.SHORT)
        else:
            sig.append(define.NO_SIGNAL)
    df_mkt['upbound'] = upbound
    df_mkt['lowbound'] = lowerbound
    df_mkt['signal'] = sig
    return df_mkt


def get_revert_signal(df_mkt: pd.DataFrame, k1: float = 0.2, k2: float = 0.2, sod: bool = True):
    pass


def get_signal(start_date: str = '', end_date: str = '', product_id: str = '', options: dict = {},
               sod: bool = True) -> pd.DataFrame:
    '''

    :param start_date:
    :param end_date:
    :param product_id:
    :param options:
    :param sod: True, the signal is available start of the day(for backtest mode);False:eod of the day(for live )
    :return:
    '''
    df = DataAPI.MktMFutdGet(mainCon=u"1", contractMark=u"", contractObject=product_id, tradeDate=u"",
                             startDate=start_date,
                             endDate=end_date, field=u"", pandas="1")
    # ['secID', 'ticker', 'exchangeCD', 'secShortName', 'secShortNameEN',
    #  'tradeDate', 'contractObject', 'contractMark', 'preSettlePrice',
    #  'preClosePrice', 'openPrice', 'highestPrice', 'lowestPrice',
    #  'settlePrice', 'closePrice', 'turnoverVol', 'turnoverValue', 'openInt',
    #  'chg', 'chg1', 'chgPct', 'mainCon', 'smainCon']
    lag_window = options.get('lag_window') or 5
    k1 = options.get('k1') or 0.2
    k2 = options.get('k2') or 0.2
    if sod:  # ignore the latest eod signal, for backtest; for the last date, got the previous market data
        df['hh'] = df['highestPrice'].rolling(lag_window).max().shift()
        df['lc'] = df['closePrice'].rolling(lag_window).min().shift()
        df['hc'] = df['closePrice'].rolling(lag_window).max().shift()
        df['ll'] = df['lowestPrice'].rolling(lag_window).min().shift()
        df['accstd'] = df['settlePrice'].rolling(lag_window).std().shift()
    else:  # the the signal when the date end, signal for the next day, for live trade
        df['hh'] = df['highestPrice'].rolling(lag_window).max()
        df['lc'] = df['closePrice'].rolling(lag_window).min()
        df['hc'] = df['closePrice'].rolling(lag_window).max()
        df['ll'] = df['lowestPrice'].rolling(lag_window).min()
        df['accstd'] = df['settlePrice'].rolling(lag_window).std()

    df = df.dropna()
    df = get_trend_signal(df_mkt=df, k1=k1, k2=k2, sod=sod)
    # ls_tag = []
    # upbound = []
    # lowbound = []
    # for item in df.values:
    #     _max_min_range = item[-3] - item[-2]  # acchigh-acclow
    #     _std = item[-1]  # acc std
    #     _min_mix_weight = options.get('min_max_weight') or 0.2
    #     _std_weight = options.get('std_weight') or 0.2
    #     _up_range = item[8] + _min_mix_weight * _max_min_range + _std_weight * _std
    #     _low_range = item[8] - _min_mix_weight * _max_min_range - _std_weight * _std
    #     upbound.append(_up_range)
    #     lowbound.append(_low_range)
    #     if item[9] > _up_range:  # open > acchigh
    #         ls_tag.append(define.SHORT)  # short
    #     elif item[9] < _low_range:  # open < acclow
    #         ls_tag.append(define.LONG)  # long
    #     else:
    #         ls_tag.append(define.NO_SIGNAL)
    #
    # df['upbound'] = upbound
    # df['lowbound'] = lowbound
    # df['signal'] = ls_tag
    return df


def handle_bar(signal_df: pd.DataFrame, strategy_option: dict, strategy_name: str = '', save_cache: bool = False):
    init_cash = strategy_option.get('init_cash') or 100000
    stop_profit = strategy_option.get('stop_profit') or 0.2
    stop_loss = strategy_option.get('stop_loss') or 0.2
    account = Account.Account(init_cash)
    pos = Position.Position()
    init_risk_ratio = strategy_option.get('inti_risk_ratio') or 0.3

    mkt_val1 = []
    mkt_val2 = []

    product_id = list(signal_df['contractObject'])[0]

    df_contract = DataAPI.FutuGet(secID=u"", ticker=u"", exchangeCD=u"", contractStatus="", contractObject=product_id,
                                  prodID="", field=u"", pandas="1")

    max_drawdown = []
    risk_ratio = []
    pos_vol_lst = []
    transac_lst = []
    for item in signal_df.values:
        instrument_id = item[1]
        open_price = item[10]
        close_price = item[14]
        settle_price = item[13]
        chg = item[19]  # close price
        chg1 = item[20]  # settle price
        pre_settle_price = item[8]
        pre_close = item[9]
        _contract_info = df_contract[df_contract.ticker == instrument_id].to_dict('record')
        margin_val = 0.0
        pos_vol = 0
        transac_vol = 0
        if _contract_info:
            open_fee = open_fee_overwrite.get(product_id.lower()) or _contract_info[0].get('tradeCommiNum')  # 元
            close_fee = close_fee_overwrite.get(product_id.lower()) or _contract_info[0].get('tradeCommiNum')  # %
            multiplier = _contract_info[0].get('contMultNum')
            tick_val = _contract_info[0].get('minChgPriceNum')
            margin_num = margin_overwrite.get(product_id.lower()) or _contract_info[0].get('tradeMarginRatio')
        else:
            open_fee = 2
            close_fee = 2
            multiplier = 10
            tick_val = 1
            margin_num = 12
        _pos = pos.get_position(product_id)
        # position.update_position(instrument_id=instrument_id, long_short=define.SHORT, price=_fill_price,
        #                          timestamp=_update_time, vol=_fill_lot, order_type=define.LONG_CLOSE)
        # update_lst.append([long_short, price, timestamp, vol])
        # print('start of check pos', item[5], _pos)
        if _pos:  # has position
            for long_short, price, ts, vol in _pos:
                if item[-1] == long_short or item[
                    -1] == define.NO_SIGNAL:  # hv pos and same signal direction or no signal, pos not change,update account fee and mv
                    if long_short == define.LONG:
                        _profit = (pre_close / price - 1) * (100 / margin_num)
                        _profit1 = (pre_settle_price / price - 1) * (100 / margin_num)
                        if _profit > stop_profit or _profit1 < -stop_loss:  # TODO  use close price, close pos, stop profit/loss
                            print("stop profit/loss for long, profit=>", _profit)
                            _fee = close_fee * vol
                            _val = (open_price - pre_close - tick_val) * vol * multiplier
                            _val1 = (open_price - pre_settle_price - tick_val) * vol * multiplier
                            account.update_fee(_fee)
                            account.update_market_value(_val, _val1, _fee)
                            pos.update_position(instrument_id=product_id, long_short=define.SHORT, price=open_price,
                                                timestamp='', vol=vol, order_type=define.LONG_CLOSE)
                            margin_val = 0.0
                            pos_vol = 0
                            transac_vol = -vol
                            # print('pos vol:', pos_vol, 'get_position vol:', pos.get_position(instrument_id))

                        else:
                            _val = chg * vol * multiplier
                            _val1 = chg1 * vol * multiplier
                            _val = _val if long_short == define.LONG else -_val
                            _val1 = _val1 if long_short == define.LONG else -_val1
                            account.update_fee(_fee)
                            account.update_market_value(_val, _val1, _fee)
                            margin_val = settle_price * vol * multiplier * margin_num / 100
                            pos_vol = vol
                            transac_vol = 0
                            # print('pos vol:', pos_vol, 'get_position vol:', pos.get_position(instrument_id))
                    elif long_short == define.SHORT:
                        _profit = (price / pre_close - 1) * (100 / margin_num)
                        _profit1 = (price / pre_close - 1) * (100 / margin_num)
                        if _profit > stop_profit or _profit1 < -stop_loss:  # TODO  use close price, close pos, stop profit/loss
                            print("stop profit/loss for short=>", _profit)
                            _fee = close_fee * vol
                            _val = (pre_close - open_price - tick_val) * vol * multiplier
                            _val1 = (pre_settle_price - open_price - tick_val) * vol * multiplier
                            account.update_fee(_fee)
                            account.update_market_value(_val, _val1, _fee)
                            pos.update_position(instrument_id=product_id, long_short=define.LONG, price=open_price,
                                                timestamp='', vol=vol, order_type=define.SHORT_CLOSE)
                            margin_val = 0.0
                            pos_vol = 0
                            transac_vol = -vol
                            # print(item[5], 'pos vol:', pos_vol, 'get_position vol:', pos.get_position(instrument_id))
                        else:
                            _val = chg * vol * multiplier
                            _val1 = chg1 * vol * multiplier
                            _val = _val if long_short == define.LONG else -_val
                            _val1 = _val1 if long_short == define.LONG else -_val1
                            account.update_fee(_fee)
                            account.update_market_value(_val, _val1, _fee)
                            margin_val = settle_price * vol * multiplier * margin_num / 100
                            pos_vol = -vol
                            transac_vol = 0
                            # print('pos vol:', pos_vol, 'get_position vol:', pos.get_position(instrument_id))
                elif long_short == define.LONG:  # long pos and short signal, close long；remarks要不要反手？？
                    _fee = close_fee * vol
                    _val = (open_price - pre_close - tick_val) * vol * multiplier
                    _val1 = (open_price - pre_settle_price - tick_val) * vol * multiplier
                    account.update_fee(_fee)
                    account.update_market_value(_val, _val1, _fee)
                    pos.update_position(instrument_id=product_id, long_short=define.SHORT, price=open_price,
                                        timestamp='', vol=vol, order_type=define.LONG_CLOSE)

                    margin_val = 0.0
                    pos_vol = 0
                    transac_vol = -vol
                    # print('close pos vol:', pos_vol, 'get_position vol:', pos.get_position(instrument_id))
                elif long_short == define.SHORT:  # short pos and long signal, close short，remarks 要不要反手？
                    _fee = close_fee * vol
                    _val = (pre_close - open_price - tick_val) * vol * multiplier
                    _val1 = (pre_settle_price - open_price - tick_val) * vol * multiplier
                    account.update_fee(_fee)
                    account.update_market_value(_val, _val1, _fee)
                    pos.update_position(instrument_id=product_id, long_short=define.LONG, price=open_price,
                                        timestamp='', vol=vol, order_type=define.SHORT_CLOSE)

                    margin_val = 0.0
                    pos_vol = 0
                    transac_vol = -vol
                    # print('pos vol:', pos_vol, 'get_position vol:', pos.get_position(instrument_id))
                else:
                    pass
        else:  # no position
            _mkt_val1 = account.get_trade_market_values()
            available_cash = _mkt_val1[-1] if _mkt_val1 else init_cash
            order_vol = int(available_cash * init_risk_ratio / (open_price * multiplier * margin_num / 100))
            if item[-1] == define.LONG:  # no pos and long signal,open long
                _fee = open_fee * order_vol
                _val = (
                               close_price - open_price - tick_val) * order_vol * multiplier  # assume open price is open_price - 1tick
                _val1 = (
                                settle_price - open_price - tick_val) * order_vol * multiplier  # assume open price is open_price - 1tick
                account.update_fee(_fee)
                account.update_market_value(_val, _val1, _fee)
                pos.update_position(instrument_id=product_id, long_short=define.LONG, price=open_price + tick_val,
                                    timestamp='',
                                    vol=order_vol, order_type=define.LONG_OPEN)

                margin_val = settle_price * order_vol * multiplier * margin_num / 100
                pos_vol = order_vol
                transac_vol = order_vol
                # print('pos vol:', pos_vol, 'get_position vol:', pos.get_position(instrument_id))
            elif item[-1] == define.SHORT:  # no pos and short signal, open short
                _fee = open_fee * order_vol
                _val = (
                               open_price - close_price - tick_val) * order_vol * multiplier  # assume open price is open_price - 1tick
                _val1 = (
                                open_price - settle_price - tick_val) * order_vol * multiplier  # assume open price is open_price - 1tick
                account.update_fee(_fee)
                account.update_market_value(_val, _val1, _fee)
                pos.update_position(instrument_id=product_id, long_short=define.SHORT, price=open_price - tick_val,
                                    timestamp='',
                                    vol=order_vol, order_type=define.SHORT_OPEN)

                margin_val = settle_price * order_vol * multiplier * margin_num / 100
                pos_vol = -order_vol
                transac_vol = order_vol
                # print('pos vol:', pos_vol, 'get_position vol:', pos.get_position(instrument_id))
            else:  # no actions needed
                pass
        # print(pos.get_position(product_id), item[5], "??????")
        mkt_val1.append(account.get_trade_market_values()[-1])
        mkt_val2.append(account.get_settle_market_values()[-1])
        _max_his_mv = max(account.get_trade_market_values())
        _drawdown_val = max(_max_his_mv - account.get_trade_market_values()[-1], 0)
        # print(_max_his_mv, account.get_trade_market_values()[-1], _drawdown_val)
        _tmp_drawdown_ratio = _drawdown_val / _max_his_mv
        max_drawdown.append(_tmp_drawdown_ratio)
        risk_ratio.append(margin_val / mkt_val1[-1])
        pos_vol_lst.append(pos_vol)
        transac_lst.append(transac_vol)
        # print(pos.get_position(instrument_id), item[5], "??????")

    signal_df['mkt_val1'] = mkt_val1
    signal_df['mkt_val2'] = mkt_val2
    signal_df['max_drawdown'] = max_drawdown
    signal_df['pos'] = pos_vol_lst
    signal_df['trans'] = transac_lst
    signal_df['risk_ratio'] = risk_ratio
    bc_start = list(signal_df['preClosePrice'])[0]
    bc_return = [item / bc_start - 1 for item in signal_df['closePrice']]

    if save_cache:
        signal_df.to_csv('{0}.csv'.format(strategy_name), index=False)
    signal_lst = [1 if item == define.LONG else -1 if item == define.SHORT else 0 for item in signal_df['signal']]
    evaluate_ret = []

    # trade_mkt_values = account.get_trade_market_values()
    # settle_mkt_values = account.get_settle_market_values()
    net_val1 = [item / init_cash for item in signal_df['mkt_val1']]
    net_val2 = [item / init_cash for item in signal_df['mkt_val2']]
    _idx_lst = list(range(signal_df.shape[0]))
    trade_date_str = [item.replace('-', '') for item in signal_df['tradeDate']]

    return_lst = np.array([item - 1 for item in net_val1])
    sharp_ratio = (net_val1[-1] / net_val1[0] - 1) / return_lst.std()
    long_sig = len([item for item in signal_lst if item == 1])
    short_sig = len([item for item in signal_lst if item == -1])
    long_holding = len([item for item in pos_vol_lst if item > 0])
    short_holding = len([item for item in pos_vol_lst if item < 0])
    open_trans = len([item for item in transac_lst if item > 0])
    close_trans = len([item for item in transac_lst if item < 0])
    bc_sharp = bc_return[-1] / np.array(bc_return).std()
    evaluate_ret.append(net_val1[-1] / net_val1[0] - 1)
    evaluate_ret.append(return_lst.std())
    evaluate_ret.append(sharp_ratio)
    evaluate_ret.append(max(max_drawdown))
    evaluate_ret.append(max(risk_ratio))
    evaluate_ret.append(long_sig)
    evaluate_ret.append(short_sig)
    evaluate_ret.append(long_holding)
    evaluate_ret.append(short_holding)
    evaluate_ret.append(open_trans)
    evaluate_ret.append(close_trans)
    evaluate_ret.append(bc_return[-1])
    evaluate_ret.append(bc_sharp)
    tick_interval = 30
    fig = plt.figure()
    fig.tight_layout()
    plt.title('{0}'.format(strategy_name))
    plt.axis('off')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    ax1 = fig.add_subplot(221)
    ax1.plot(net_val1, label='net value(by close)')
    # ax1.plot(df['acclow'], label='acclow')
    ax1.legend()

    # ax2 = ax1.twinx()
    # # ax2.plot(net_val2, 'r', label='nv2')
    # ax2.legend()
    ax1.set_xticks(_idx_lst[::tick_interval])
    ax1.set_xticklabels(trade_date_str[::tick_interval], rotation=45, fontsize=5)

    ax3 = fig.add_subplot(222)
    ax3.plot(signal_df['closePrice'], 'r', label='closePrice')
    ax3.legend()
    ax3.set_xticks(_idx_lst[::tick_interval])
    ax3.set_xticklabels(trade_date_str[::tick_interval], rotation=45, fontsize=5)

    ax8 = ax3.twinx()
    ax8.bar(_idx_lst, transac_lst, label='pos_vol')

    # ax4 = ax3.twinx()
    ax4 = fig.add_subplot(223)
    ax4.bar(_idx_lst, signal_lst, label='signal')
    ax4.legend()
    ax4.set_xticks(_idx_lst[::tick_interval])
    ax4.set_xticklabels(trade_date_str[::tick_interval], rotation=45, fontsize=5)

    ax6 = ax4.twinx()
    ax6.plot(signal_df['upbound'], 'r', label='upbound')
    ax6.plot(signal_df['lowbound'], 'r', label='lowbound')
    ax6.plot(signal_df['preClosePrice'], 'g', label='pre_close')

    ax5 = fig.add_subplot(224)
    ax5.plot(max_drawdown, label='max_drawdown')
    ax5.legend()
    ax5.set_xticks(_idx_lst[::tick_interval])
    ax5.set_xticklabels(trade_date_str[::tick_interval], rotation=45, fontsize=5)

    ax7 = ax5.twinx()
    ax7.plot(risk_ratio, 'r', label='risk ratio')
    if save_cache:
        plt.savefig('{0}.jpg'.format(strategy_name))
    # plt.show()
    return evaluate_ret


def backtest(start_date: str = '20210403', end_date: str = '20220419', save_factor: bool = False):
    min_max_weight_lst = [0.2, 0.3, 0.5]
    # min_max_weight_lst = [0.2]
    std_weight_lst = [-0.1, 0, 0.2, 0.3, 0.5]
    # std_weight_lst = [0.2]
    lag_windows_lst = [5, 12, 26]
    # lag_windows_lst = [5]
    # product_id_lst = ['P', 'rb', 'm', 'eg', 'fu']
    product_id_lst = ['P', 'rb']
    k1_lst = [0.2]
    k2_lst = [0.2]
    back_test_ret = []
    strategy_option = {'stop_profit': 0.3, 'stop_loss': 0.5, 'init_cash': 100000, 'inti_risk_ratio': 0.3}
    for w1 in k1_lst:
        for w2 in k2_lst:
            for std_weight in std_weight_lst:
                for product_id in product_id_lst:
                    for lag_window in lag_windows_lst:
                        strategy_name = "{0}_{1}_{2}_{3}".format(product_id, lag_window, w1, w2)
                        signal_option = {'k1': w1, 'std_weight': std_weight, 'k2': w2,
                                         'lag_window': lag_window}
                        df = get_signal(start_date=start_date, end_date=end_date, product_id=product_id,
                                        options=signal_option, sod=True)
                        ret = handle_bar(signal_df=df, strategy_option=strategy_option, strategy_name=strategy_name,
                                         save_cache=save_factor)
                        # select_product(start_date='20220401', end_date='20220415', product_lst=['rb', 'm', ])
                        ret.append(product_id)
                        ret.append(start_date)
                        ret.append(end_date)
                        ret.append(strategy_name)
                        back_test_ret.append(ret)
    bt_df = pd.DataFrame(back_test_ret,
                         columns=['acc_return', 'std', 'sharp_ratio', 'max_drawdown', 'max_risk_ratio', 'long_sig',
                                  'short_sig', 'long_holding', 'short_holding', 'open_trans', 'close_trans',
                                  'bc_return',
                                  'bc_sharp_ratio', 'product_id',
                                  'start_date', 'end_date', 'strategy_name'])
    bt_df.to_csv('back_test_result.csv', index=False)


def get_eod_signal():
    pass


if __name__ == '__main__':
    backtest(start_date='20210403', end_date='20220419', save_factor=False)
