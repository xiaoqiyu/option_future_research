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
from T0Signal import TOSignal
from editorconfig import get_properties, EditorConfigError
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np
import hashlib


def backtesting(product_id='m', trade_date='20210401', prev_date='2021-03-31'):
    # file_name = 'C:\projects\pycharm\option_future_research\.editorconfig'
    # backtesting_config = "{0}_{1}".format(product_id, trade_date)
    backtesting_config = ''
    try:
        options = get_properties(file_name)
    except EditorConfigError:
        logging.warning("Error getting EditorConfig propterties", exc_info=True)
    else:
        for key, value in options.items():
            backtesting_config = '{0},{1}:{2}'.format(backtesting_config, key, value)
            # print("{0}:{1}".format(key, value))
    # load daily market data
    df = pd.read_csv(daily_cache_name)
    df = df[df.mainCon == 1]
    df = df[df.contractObject == product_id.upper()]
    date_lst = [item.replace('-', '') for item in list(df.tradeDate)]
    df['tradeDate1'] = date_lst
    df = df[df.tradeDate1 == prev_date]
    try:
        row = df.values[0]
    except Exception as ex:
        print('missing daily mkt for:', trade_date)
        return
    open_price, high, low, close, settle = row[9: 14]
    instrument_id = row[1]
    # backtesting_config = "{0},{1}".format(instrument_id, backtesting_config)
    try:
        df = pd.read_csv('cache/{0}/{1}.csv'.format(trade_date, product_id))
        df.columns = cols
    except Exception as ex:
        print('read mkt:{0},{1} with error:{2}'.format(trade_date, product_id, ex))
        return

    # tick format
    m_df = df[df.InstrumentID == instrument_id][
        ["InstrumentID", "LastPrice", "OpenPrice", "HighestPrice", "LowestPrice", "Volume", "Turnover", "OpenInterest",
         "UpperLimitPrice", "LowerLimitPrice", "UpdateTime",
         "UpdateMillisec", "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1"]]
    m_df = m_df.sort_values(by='UpdateTime', ascending=True)
    values = m_df.values
    factor = Factor(product_id=product_id, instrument_id=instrument_id, trade_date=trade_date)
    position = Position()
    account = Account()
    signal = TOSignal(factor, position)
    # kwargs.update({'instrument_id': instrument_id})
    #
    volitility = float(options.get('volitility')) or 20.0
    k1 = float(options.get('k1')) or 0.2
    k2 = float(options.get('k2')) or 0.2
    # # instrument_id = kwargs.get('instrument_id')
    # # adjust by multiplier
    stop_profit = float(options.get('stop_profit')) or 5.0
    stop_loss = float(options.get('stop_loss')) or 20.0
    print('trade_date:{0}-----------------'.format(trade_date))
    print("target return", stop_profit)
    open_fee = float(options.get('open_fee')) or 3.0
    close_t0_fee = float(options.get('close_t0_fee')) or 0.0
    fee = open_fee + close_t0_fee
    start_tick = int(options.get('start_tick')) or 2
    long_lots_limit = int(options.get('long_lots_limit')) or 1
    short_lots_limit = int(options.get('short_lots_limit')) or 1
    slope_upper = float(options.get('slope_upper')) or 1.0
    slope_lower = float(options.get('slope_lower')) or -1.0
    start_timestamp = options.get('start_timestamp') or '09:05:00'
    end_timestamp = options.get('end_timestamp') or '22:50:00'
    total_return = 0.0
    close_price = 0.0  # not the true close price, to close the position
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
        if _update_time >= end_timestamp:
            continue
        close_price = _last
        _signal = signal.get_signal(volitility=volitility, k1=k1, k2=k2, instrument_id=instrument_id,
                                    stop_profit=stop_profit, fee=fee, start_tick=start_tick,
                                    long_lots_limit=long_lots_limit, short_lots_limit=short_lots_limit,
                                    slope_upper=slope_upper,
                                    slope_lower=slope_lower, stop_loss=stop_loss,
                                    multiplier=dict_multiplier.get(product_id))
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
            #             if _last    _last_price = item[1]
    from copy import deepcopy
    _pos = position.get_position(instrument_id)
    total_return_risk = total_return
    total_risk = 0.0
    if _pos:
        _tmp_pos = deepcopy(_pos)
        for item in _tmp_pos:
            if item[0] == LONG:
                _return = (close_price - item[1]) * dict_multiplier.get(product_id) - fee
                # TODO position control
                # if _return < stop_loss:
                #     continue
                total_return += _return
                total_risk += item[1]
                print('final long close with return:{0},total return after:{1}'.format(_return, total_return))
                account.add_transaction(
                    [idx, instrument_id, _text_lst[LONG_CLOSE], close_price, item[1], close_t0_fee, _update_time,
                     _return])
                account.update_fee(close_t0_fee)
                position.close_position(instrument_id=instrument_id, long_short=LONG, price=close_price,
                                        timestamp=_update_time)
            else:
                _return = ((item[1] - close_price) * dict_multiplier.get(product_id) - fee)
                # TODO position control
                # if _return < stop_loss:
                #     continue
                total_risk += item[1]
                total_return += _return
                print('final short close with return:{0},total return after:{1}'.format(_return, total_return))
                account.add_transaction(
                    [idx, instrument_id, _text_lst[SHORT_CLOSE], close_price, item[1], close_t0_fee, _update_time,
                     _return])
                account.update_fee(close_t0_fee)
                position.close_position(instrument_id=instrument_id, long_short=SHORT, price=close_price,
                                        timestamp=_update_time)

    # _pos = position.get_position(instrument_id)
    # if _pos:
    #     for item in _pos:
    #         total_risk += item[1]
    factor.cache_factor()
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
            if item[7] > 0:
                correct_long_open += 1
            else:
                wrong_long_open += 1
        else:
            if item[7] > 0:
                correct_short_open += 1
            else:
                wrong_short_open += 1
    print('trade date', trade_date)
    print(long_open, short_open, correct_long_open, wrong_long_open, correct_short_open, wrong_short_open)
    print('total return:', total_return)
    print('total fee:', account.fee)
    print('total risk:', total_risk)
    print('-------------------------------------------')

    precision = (correct_long_open + correct_short_open) / (
            long_open + short_open) if long_open + short_open > 0 else 0.0
    # f = open("results/results_{0}.txt".format(product_id), "a")
    # _key = '{0}_{1}'.format(trade_date, instrument_id)
    result_fname_digest = hashlib.sha256(bytes(backtesting_config, encoding='utf-8')).hexdigest()
    # f.write("{0}:{1}\n".format(backtesting_config, result_fname_digest))
    try:
        result_df = pd.read_csv('results/{0}.csv'.format(result_fname_digest))
    except Exception as ex:
        result_df = pd.DataFrame(
            {'trade_date': [], 'product_id': [], 'instrument_id': [], 'total_return': [], 'total_return_risk': [],
             'total_fee': [],
             'total_risk': [], 'precision': [], 'long_open': [], 'short_open': [],
             'correct_long_open': [], 'wrong_long_open': [], 'correct_short_open': [], 'wrong_short_open': [],
             })

    result_df = result_df.append(
        {'trade_date': trade_date, 'product_id': product_id, 'instrument_id': instrument_id,
         'total_return': total_return, 'total_return_risk': total_return_risk, 'total_fee': account.fee,
         'total_risk': total_risk, 'precision': precision,
         'long_open': long_open, 'short_open': short_open, 'correct_long_open': correct_long_open,
         'wrong_long_open': wrong_long_open, 'correct_short_open': correct_short_open,
         'wrong_short_open': wrong_short_open,
         }, ignore_index=True)
    result_df.to_csv('results/{0}.csv'.format(result_fname_digest), index=False)
    # f.write(
    #     "${0}\ntotal_return:{1},total_fee:{2},precision:{3},correct_long_open:{4},wrong_long_open:{5},correct_short_open:{6},"
    #     "wrong_short_open:{7},short_open:{8},long_open:{9},total_risk:{10}\n".format(
    #         backtesting_config, round(total_return, 2), round(account.fee, 2), round(precision, 2), correct_long_open,
    #         wrong_long_open,
    #         correct_short_open,
    #         wrong_short_open, short_open, long_open, total_risk))
    ret = (total_return, account.fee, precision)
    return ret
