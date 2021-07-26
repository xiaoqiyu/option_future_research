#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 14:05
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : BackTester.py


from define import *
from Factor import Factor
from Position import Position
from Account import Account
from editorconfig import get_properties, EditorConfigError
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np


def get_signal(factor=None, position=None, *args, **kwargs):
    volitility = kwargs.get('volitility') or 20.0
    k1 = kwargs.get('k1') or 1.0
    k2 = kwargs.get('k2') or 1.0
    instrument_id = kwargs.get('instrument_id')
    # adjust by multiplier
    stop_profit = kwargs.get('stop_profit') or 5.0
    stop_loss = kwargs.get('stop_loss') or 20.0

    # fee = kwargs.get('fee') or 3.0
    open_fee = kwargs.get('open_fee') or 1.51
    close_to_fee = kwargs.get('close_t0_fee') or 0.0
    fee = open_fee + close_to_fee
    # start_tick must be greater than 2
    start_tick = kwargs.get('start_tick') or 2
    long_lots_limit = kwargs.get('long_lots_limit') or 1
    short_lots_limit = kwargs.get('short_lots_limit') or 1
    slope_upper = kwargs.get('slope_upper') or 1.0
    slope_lower = kwargs.get('slope_lower') or -1.0
    _position = position.get_position(instrument_id)

    assert isinstance(factor, Factor)
    assert isinstance(position, Position)
    if len(factor.last_price) < start_tick:
        return NO_SIGNAL
    _long, _short, long_price, short_price = 0, 0, 0.0, 0.0

    if _position:
        for item in _position:
            if item[0] == LONG:
                long_price = item[1]
                _long += 1
            elif item[0] == SHORT:
                short_price = item[1]
                _short += 1
    # if factor.last_price[-1] > factor.last_price[-2] and factor.last_price[-1] > factor.vwap[
    #     -1] + k1 * volitility and not _long:
    if (factor.slope[-1] > factor.slope[-2]) and (factor.slope[-2] < factor.slope[-3]) and abs(
            factor.slope[-1]) > slope_upper and _short < short_lots_limit:
        # if (factor.slope[-1] > factor.slope[-2]) and abs(
        #             factor.slope[-1]) > 0.8 and not _long:
        # print(factor.last_price, factor.slope)
        # print(_short, short_lots_limit)
        return SHORT_OPEN
    # if factor.last_price[-1] < factor.last_price[-2] and factor.last_price[-1] < factor.vwap[
    #     -1] + k2 * volitility and not _short:
    if (factor.slope[-1] < factor.slope[-2]) and (factor.slope[-2] > factor.slope[-3]) and abs(
            factor.slope[-1]) > slope_upper and _long < long_lots_limit:
        # if (factor.slope[-1] < factor.slope[-2]) and abs(
        #             factor.slope[-1]) > 0.8 and not _short:

        # print(factor.last_price,factor.slope)
        return LONG_OPEN
    # if factor.last_price[-1] < factor.last_price[-2] and _long and factor.last_price[-1] > long_price + stop_profit:
    if _long and (factor.last_price[-1] > long_price + stop_profit + fee or
                  factor.last_price[-1] < long_price - stop_loss - fee):
        return LONG_CLOSE
    # if factor.last_price[-1] > factor.last_price[-2] and _short and short_price > factor.last_price[-1] + stop_profit:
    if _short and (
            short_price > factor.last_price[-1] + stop_profit + fee or factor.last_price[
        -1] > short_price + stop_loss + fee):
        return SHORT_CLOSE
    return NO_SIGNAL


def on_tick(factor=None, position=None, tick=[], *args, **kwargs):
    assert isinstance(factor, Factor)
    factor.update_factor(tick)
    return get_signal(factor, position, args, kwargs)


def backtesting(product_id='m', trade_date='20210401', prev_date='2021-03-31'):
    file_name = 'C:\projects\pycharm\option_future_research\.editorconfig'
    backtesting_config = "{0}_{1}".format(product_id, trade_date)
    try:
        options = get_properties(file_name)
    except EditorConfigError:
        logging.warning("Error getting EditorConfig propterties", exc_info=True)
    else:
        for key, value in options.items():
            backtesting_config = '{0},{1}:{2}'.format(backtesting_config, key, value)
            # print("{0}:{1}".format(key, value))
    # load daily market data
    df = pd.read_csv('cache/future_20210331_20210630.csv')
    df = df[df.mainCon == 1]
    df = df[df.contractObject == product_id.upper()]
    df = df[df.tradeDate == prev_date]
    row = df.values[0]
    open_price, high, low, close, settle = row[9: 14]
    instrument_id = row[1]
    df = pd.read_csv('cache/{0}/{1}.csv'.format(trade_date, product_id))
    df.columns = cols

    # tick format
    m_df = df[df.InstrumentID == instrument_id][
        ["InstrumentID", "LastPrice", "OpenPrice", "HighestPrice", "LowestPrice", "Volume", "Turnover", "OpenInterest",
         "UpperLimitPrice", "LowerLimitPrice", "UpdateTime",
         "UpdateMillisec", "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1"]]
    m_df = m_df.sort_values(by='UpdateTime', ascending=True)
    values = m_df.values
    factor = Factor()
    position = Position()
    account = Account()
    # kwargs.update({'instrument_id': instrument_id})
    #
    volitility = float(options.get('volitility')) or 20.0
    k1 = float(options.get('k1')) or 0.2
    k2 = float(options.get('k2')) or 0.2
    # # instrument_id = kwargs.get('instrument_id')
    # # adjust by multiplier
    stop_profit = float(options.get('stop_profit')) or 5.0
    stop_loss = float(options.get('stop_loss')) or 20.0
    print("target return", stop_profit)
    open_fee = float(options.get('open_fee')) or 3.0
    close_t0_fee = float(options.get('close_t0_fee')) or 0.0
    fee = open_fee + close_t0_fee
    start_tick = int(options.get('start_tick')) or 2
    long_lots_limit = int(options.get('long_lots_limit')) or 1
    short_lots_limit = int(options.get('short_lots_limit')) or 1
    slope_upper = float(options.get('slope_upper')) or 1.0
    slope_lower = float(options.get('slope_lower')) or -1.0
    total_return = 0.0
    # transaction_lst = []
    _text_lst = ['00', '01', '10', '11']
    for idx, item in enumerate(values):

        _last = item[1]
        _update_time = item[10]
        # TODO idx will be some cnt in real trade
        # TODO check the hardcode
        if idx < 5:
            continue
        factor.update_factor(item, idx=idx)
        # TODO hardcode timestamp
        if _update_time >= '22:50:00':
            continue
        _signal = get_signal(factor, position, volitility=volitility, k1=k1, k2=k2, instrument_id=instrument_id,
                             stop_profit=stop_profit, fee=fee, start_tick=start_tick,
                             long_lots_limit=long_lots_limit, short_lots_limit=short_lots_limit, slope_upper=1.0,
                             slope_lower=-1.0, stop_loss=stop_loss)
        # _signal = on_tick(factor, position, item, args, kwargs)
        if _signal == LONG_OPEN:
            position.open_position(instrument_id, LONG, _last, _update_time)
            account.add_transaction(
                [idx, instrument_id, _text_lst[LONG_OPEN], _last, _last, open_fee, _update_time, 0.0])
            account.update_fee(open_fee)

        elif _signal == SHORT_OPEN:
            position.open_position(instrument_id, SHORT, _last, _update_time)
            account.add_transaction(
                [idx, instrument_id, _text_lst[SHORT_OPEN], _last, _last, open_fee, _update_time, 0.0])
            account.update_fee(open_fee)

        elif _signal == LONG_CLOSE:
            _pos = position.get_position_side(instrument_id, LONG)
            if _pos:
                position.close_position(instrument_id, LONG, _last, _update_time)
                total_return += (_last - _pos[1]) * dict_multiplier.get(product_id) - fee
                account.add_transaction(
                    [idx, instrument_id, _text_lst[LONG_CLOSE], _last, _pos[1], close_t0_fee, _update_time,
                     (_last - _pos[1]) * dict_multiplier.get(product_id) - fee])
                account.update_fee(close_t0_fee)
        elif _signal == SHORT_CLOSE:
            _pos = position.get_position_side(instrument_id, SHORT)
            if _pos:
                position.close_position(instrument_id, SHORT, _last, _update_time)
                total_return += (_pos[1] - _last) * dict_multiplier.get(product_id) - fee
                account.add_transaction(
                    [idx, instrument_id, _text_lst[SHORT_CLOSE], _last, _pos[1], close_t0_fee, _update_time,
                     (_pos[1] - _last) * dict_multiplier.get(product_id) - fee])
                account.update_fee(close_t0_fee)
        else:  # NO_SIGNAL
            pass
            # _pos = position.get_position(instrument_id)
            # if _pos:
            #     for item in _pos:
            #         if item[0] == LONG:
            #             if _last

    _last_price = item[1]
    _pos = position.get_position(instrument_id)
    if _pos:
        for item in _pos:
            if item[0] == LONG:
                total_return += ((_last_price - item[1]) * dict_multiplier.get(product_id) - fee)
                account.add_transaction(
                    [idx, instrument_id, _text_lst[LONG_CLOSE], _last_price, item[1], close_t0_fee, _update_time,
                     (_last_price - item[1]) * dict_multiplier.get(product_id) - fee])
                account.update_fee(close_t0_fee)
            else:
                total_return += ((item[1] - _last_price) * dict_multiplier.get(product_id) - fee)
                account.add_transaction(
                    [idx, instrument_id, _text_lst[SHORT_CLOSE], _last_price, item[1], close_t0_fee, _update_time,
                     (item[1] - _last_price) * dict_multiplier.get(product_id) - fee])
                account.update_fee(close_t0_fee)

    _idx_lst = list(range(len(factor.last_price)))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(_idx_lst[PLT_START:PLT_END], factor.last_price[PLT_START:PLT_END])
    ax1.plot(_idx_lst[PLT_START:PLT_END], factor.vwap[PLT_START:PLT_END])
    ax1.plot(_idx_lst[PLT_START:PLT_END], factor.turning[PLT_START:PLT_END])
    # ax1.plot(_idx_lst[:-10], factor.upper_bound[:-10], 'r')
    # ax1.plot(_idx_lst[:-10], factor.lower_bound[:-10], 'g')
    ts_idx = 0
    for idx, item in enumerate(factor.update_time):
        if item >= '21:00:00':
            ts_idx = idx
            break
    # ts_idx = _ts_lst.index('21:00:00')
    print(np.array(factor.last_price).std())
    ax1.axvline(ts_idx)
    # ax.plot(_idx_lst, _vwap_spread_lst)
    # for item in trans_lst:
    #     plt.text(item[0], item[1], s='({0},{1})'.format(item[3], item[4]))
    ax1.grid(True)
    ax1.set_title('{0}_{1}'.format(instrument_id, trade_date))
    x_idx = range(0, len(_idx_lst), 600)
    xtick_labels = []
    for _idx in x_idx:
        xtick_labels.append(factor.update_time[_idx])
    # ax1.xticks(x_idx, xtick_labels, rotation=60, FontSize=6)
    for item in account.transaction:
        _t_lst = ['lo', 'lc', 'so', 'sc']
        ax1.text(item[0], item[3], s='{0}'.format(item[2]))

    ax2 = ax1.twinx()
    # ax2.plot(_idx_lst[PLT_START:PLT_END], factor.slope[PLT_START:PLT_END], 'r')

    plt.savefig('results/{0}_{1}.jpg'.format(instrument_id, trade_date))
    # plt.show()
    trans_df = pd.DataFrame(account.transaction,
                            columns=['idx', 'instrument_id', 'direction', 'price', 'open_price', 'fee', 'timestamp',
                                     'return'])
    trans_df.to_csv('results/trans_{0}_{1}.csv'.format(instrument_id, trade_date), index=False)
    long_open, short_open, correct_long_open, wrong_long_open, correct_short_open, wrong_short_open = 0, 0, 0, 0, 0, 0
    total_fee = 0.0

    for item in account.transaction:
        if item[2] == '00':
            long_open += 1
        elif item[2] == '10':
            short_open += 1
        elif item[2] == '01':
            if item[5] > 0:
                correct_long_open += 1
            else:
                wrong_long_open += 1
        else:
            if item[5] > 0:
                correct_short_open += 1
            else:
                wrong_short_open += 1

    print(long_open, short_open, correct_long_open, wrong_long_open, correct_short_open, wrong_short_open)
    print('total return:', total_return)
    print('total fee:', account.fee)
    precision = (correct_long_open + correct_short_open) / (long_open + correct_short_open)
    f = open("results/results_{0}.txt".format(product_id), "a")
    _key = '{0}_{1}'.format(trade_date, instrument_id)
    f.write(
        "${0}\ntotal_return:{1},long_open:{2},short_open:{3},correct_long_open:{4},wrong_long_open:{5},correct_short_open:{6},"
        "wrong_short_open:{7},precision:{8},total_fee:{9}\n".format(
            backtesting_config, total_return, long_open, short_open, correct_long_open, wrong_long_open,
            correct_short_open,
            wrong_short_open, precision, account.fee))
