#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 14:05
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : BackTester.py
import time
from .Factor import Factor
from .Position import Position
from backtester.Account import Account
from strategy.T0Signal import TOSignal
from strategy.RegSignal import RegSignal
from editorconfig import get_properties, EditorConfigError
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from datetime import datetime
import os
import utils.utils as utils
import utils.define as define
from copy import deepcopy


def get_fill_ret(order=[], tick=1, mkt=[]):
    _order_type, _price, _lot = order
    bid_price1, ask_price1, bid_vol1, ask_vol1 = mkt[-4:]
    _last_price = mkt[3]
    if _order_type == define.LONG:
        if _lot <= ask_vol1 and (not _price or _price >= ask_price1):
            return [ask_price1, _lot]
        else:
            return [0, 0]
    if _order_type == define.SHORT:
        if _lot <= bid_vol1 and (not _price or _price <= bid_price1):
            return [bid_price1, _lot]
        else:
            return [0, 0]
    return [0, 0]


def backtesting(product_id='m', trade_date='20210401', signal_name='RegSignal', result_fname_digest='', options={},
                plot_mkt=False):
    # load back test config
    backtesting_config = ''
    if not options:
        try:
            _conf_file = utils.get_path([define.CONF_DIR,
                                         define.CONF_FILE_NAME])
            options = get_properties(_conf_file)
        except EditorConfigError:
            logging.warning("Error getting EditorConfig propterties", exc_info=True)

    for key, value in options.items():
        backtesting_config = '{0},{1}:{2}'.format(backtesting_config, key, value)

    # get instrument id and exchange id
    instrument_id_df = utils.get_instrument_ids(start_date=trade_date, end_date=trade_date, product_id=product_id)
    instrument_id, trade_date, exchange_cd = instrument_id_df.values[0]

    _mul_num = utils.get_mul_num(instrument_id=instrument_id)
    _tick_mkt_path = os.path.join(define.TICK_MKT_DIR, define.exchange_map.get(exchange_cd),
                                  '{0}_{1}.csv'.format(instrument_id, trade_date.replace('-', '')))

    tick_mkt = pd.read_csv(_tick_mkt_path, encoding='gbk')
    tick_mkt.columns = define.tb_cols
    logging.info('trade_date:{0}, instrument_id:{1}, shape:{2}'.format(trade_date, instrument_id, tick_mkt.shape))
    # m_df = df[df.InstrumentID == instrument_id][define.selected_cols]
    # m_df = m_df.sort_values(by='UpdateTime', ascending=True)

    values = tick_mkt.values

    # init factor, position account signal
    factor = Factor(product_id=product_id, instrument_id=instrument_id, trade_date=trade_date)
    position = Position()
    account = Account()
    signal_map = {'T0Signal': TOSignal(factor=factor, position=position),
                  'RegSignal': RegSignal(factor=factor, position=position, instrument_id=instrument_id,
                                         trade_date=trade_date)}
    # signal = TOSignal(factor, position)
    signal = signal_map.get(signal_name)
    print('is trade available------------------------', signal.is_available)
    if not signal.is_available:
        return

    # process backtest parameter
    stop_profit = float(options.get('stop_profit')) or 5.0
    stop_loss = float(options.get('stop_loss')) or 20.0
    print('trade_date:{0}-----------------'.format(trade_date))
    print("target return", stop_profit)
    open_fee = float(options.get('open_fee')) or 3.0
    close_t0_fee = float(options.get('close_t0_fee')) or 0.0
    fee = open_fee + close_t0_fee
    start_timestamp = options.get('start_timestamp') or '09:05:00'
    start_datetime = '{0} {1}'.format(trade_date, start_timestamp)
    end_timestamp = options.get('end_timestamp') or '22:50:00'
    end_datetime = '{0} {1}'.format(trade_date, end_timestamp)
    delay_sec = int(options.get('delay_sec')) or 5
    total_return = 0.0
    close_price = 0.0  # not the true close price, to close the position
    _text_lst = ['00', '01', '10', '11']
    update_factor_time = 0.0
    get_signal_time = 0.0

    # robust handling, skip dates with records missing more than threshold(e.g. 0.3 here)
    if len(values) < define.TICK_SIZE * define.MKT_MISSING_SKIP:
        logging.warning("miss mkt for trade_date:{0} with len:{1} and tick size:{2}".format(trade_date, len(values),
                                                                                            define.TICK_SIZE
                                                                                            ))
        return
    _size = len(values)
    print('size is:', _size)
    for idx, item in enumerate(values):
        _last = item[3]
        _update_time = item[2]

        start_ts = time.time()
        factor.update_factor(item, idx=idx, multiplier=_mul_num)
        end_ts = time.time()
        update_factor_time += (end_ts - start_ts)
        # print("update factor time:{0}".format(update_factor_time))

        if not utils.is_trade(start_timestamp, end_timestamp, _update_time):
            # print(_update_time, 'not trade time--------------')
            continue

        close_price = _last
        close_timestamp = _update_time
        start_ts = time.time()

        options.update({'instrument_id': instrument_id})
        options.update({'multiplier': _mul_num})
        options.update({'fee': fee})
        options.update({'tick': item})
        options.update({'trade_date': trade_date})
        # get signal
        s1 = time.time()
        _signal = signal(params=options)
        e1 = time.time()
        # print('get signal time is:', e1-s1)
        # print(idx, _update_time, _signal)
        # print("Signal is:{0}".format(_signal))
        end_ts = time.time()
        get_signal_time += (end_ts - start_ts)
        _pos = position.get_position(instrument_id)
        _last_trans_time = _pos[-1][2] if _pos else start_datetime
        dt_curr_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
        try:
            dt_last_trans_time = datetime.strptime(_last_trans_time, '%H:%M:%S')
        except Exception as ex:
            dt_last_trans_time = datetime.strptime(_last_trans_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
        else:
            pass
        # _trans_gap_time = (dt_curr_time.hour * 3600 + dt_curr_time.minute * 60 + dt_curr_time.second) - (
        #         dt_last_trans_time.hour * 3600 + dt_last_trans_time.minute * 60 + dt_last_trans_time.second)
        _trans_gap_time = (dt_curr_time - dt_last_trans_time).seconds
        # order gap, to be robust
        if _trans_gap_time < delay_sec:
            # print('skip here--------------------------', _trans_gap_time, delay_sec, _update_time)
            continue
        s2 = time.time()
        # handle signal, update account and position
        if _signal == define.LONG_OPEN:
            _fill_price, _fill_lot = get_fill_ret(order=[define.LONG, None, 1], tick=1, mkt=item)
            if _fill_lot > 0:
                position.open_position(instrument_id, define.LONG, _fill_price, _update_time)
                account.add_transaction(
                    [idx, instrument_id, _text_lst[define.LONG_OPEN], _last, _fill_price, _fill_price, open_fee,
                     _update_time,
                     _update_time,
                     0.0, 0.0])
                account.update_fee(open_fee)

        elif _signal == define.SHORT_OPEN:
            _fill_price, _fill_lot = get_fill_ret(order=[define.SHORT, None, 1], tick=1, mkt=item)
            if _fill_lot > 0:
                position.open_position(instrument_id, define.SHORT, _fill_price, _update_time)
                account.add_transaction(
                    [idx, instrument_id, _text_lst[define.SHORT_OPEN], _last, _fill_price, _fill_price, open_fee,
                     _update_time,
                     _update_time,
                     0.0, 0.0])
                account.update_fee(open_fee)

        elif _signal == define.LONG_CLOSE:
            _pos = position.get_position_side(instrument_id, define.LONG)
            if _pos:
                dt_curr_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(_pos[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(_pos[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds

                _fill_price, _fill_lot = get_fill_ret(order=[define.SHORT, None, 1], tick=1, mkt=item)
                if _fill_lot > 0:
                    position.close_position(instrument_id, define.LONG, _fill_price, _update_time)
                    total_return += (_fill_price - _pos[1]) * _mul_num - fee

                    account.add_transaction(
                        [idx, instrument_id, _text_lst[define.LONG_CLOSE], _last, _fill_price, _pos[1], close_t0_fee,
                         _pos[2],
                         _update_time,
                         holding_time,
                         (_fill_price - _pos[1]) * _mul_num - fee])
                    account.update_fee(close_t0_fee)
        elif _signal == define.SHORT_CLOSE:
            _pos = position.get_position_side(instrument_id, define.SHORT)
            if _pos:
                dt_curr_time = datetime.strptime(_update_time.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(_pos[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(_pos[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds

                _fill_price, _fill_lot = get_fill_ret(order=[define.LONG, None, 1], tick=1, mkt=item)
                if _fill_lot > 0:
                    position.close_position(instrument_id, define.SHORT, _fill_price, _update_time)
                    total_return += (_pos[1] - _fill_price) * _mul_num - fee
                    account.add_transaction(
                        [idx, instrument_id, _text_lst[define.SHORT_CLOSE], _last, _fill_price, _pos[1], close_t0_fee,
                         _pos[2],
                         _update_time,
                         holding_time,
                         (_pos[1] - _fill_price) * _mul_num - fee])
                    account.update_fee(close_t0_fee)
        else:  # NO_SIGNAL
            pass
        e2 = time.time()
        # print('handle signal time:', idx, idx/_size, e2-s2)
    _pos = position.get_position(instrument_id)
    total_return_risk = total_return
    total_risk = 0.0
    print("complete tick fedding....")
    if _pos:
        _tmp_pos = deepcopy(_pos)
        for item in _tmp_pos:
            if item[0] == define.LONG:
                dt_curr_time = datetime.strptime(close_timestamp.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(item[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(item[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds
                # TODO to apply  fill ??now assume all fill with latest price with one tick down
                _return = (close_price - define.TICK - item[1]) * _mul_num - fee

                total_return += _return
                total_risk += item[1]
                print('final long close with return:{0},total return after:{1} for trade_date:{2}'.format(_return,
                                                                                                          total_return,
                                                                                                          trade_date))

                account.add_transaction(
                    [idx, instrument_id, _text_lst[define.LONG_CLOSE], close_price, close_price - define.TICK, item[1],
                     close_t0_fee,
                     item[2],
                     close_timestamp, holding_time, _return])
                account.update_fee(close_t0_fee)
                position.close_position(instrument_id=instrument_id, long_short=define.LONG, price=close_price,
                                        timestamp=close_timestamp)
            else:

                dt_curr_time = datetime.strptime(close_timestamp.split('.')[0], '%Y-%m-%d %H:%M:%S')
                try:
                    dt_last_trans_time = datetime.strptime(item[2], '%H:%M:%S')
                except Exception as ex:
                    dt_last_trans_time = datetime.strptime(item[2].split('.')[0], '%Y-%m-%d %H:%M:%S')
                else:
                    pass
                holding_time = (dt_curr_time - dt_last_trans_time).seconds
                # TODO to apply  fill ??now assume all fill with latest price with one tick up, tick hardcode
                _return = ((item[1] - close_price - define.TICK) * _mul_num - fee)
                total_risk += item[1]
                total_return += _return
                print('final short close with return:{0},total return after:{1}'.format(_return, total_return))

                account.add_transaction(
                    [idx, instrument_id, _text_lst[define.SHORT_CLOSE], close_price, close_price + define.TICK, item[1],
                     close_t0_fee, item[2],
                     close_timestamp, holding_time, _return])
                account.update_fee(close_t0_fee)
                position.close_position(instrument_id=instrument_id, long_short=define.SHORT, price=close_price,
                                        timestamp=close_timestamp)

    if options.get('cache_factor') == '1':
        logging.info('Start cache factor')
        factor.cache_factor()
        logging.info('Complete cache factor')
    else:
        logging.info('Stip cache factor')
    if plot_mkt:
        _idx_lst = list(range(len(factor.last_price)))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(_idx_lst[define.PLT_START:define.PLT_END], factor.last_price[define.PLT_START:define.PLT_END])
        ax1.plot(_idx_lst[define.PLT_START:define.PLT_END], factor.vwap[define.PLT_START:define.PLT_END])
        ax1.plot(_idx_lst[define.PLT_START:define.PLT_END], factor.turning[define.PLT_START:define.PLT_END])
        print(np.array(factor.last_price).std())
        ax1.grid(True)
        ax1.set_title('{0}_{1}'.format(instrument_id, trade_date))
        xtick_labels = [item[:-3] for item in factor.update_time]
        ax1.set_xticks(_idx_lst[::3600])
        min_lst = []
        ax1.set_xticklabels(xtick_labels[::3600])
        # ax1.set_xticks(factor.update_time[::3600])
        # ax1.set_xticks(x_idx, xtick_labels, rotation=60, FontSize=6)
        for item in account.transaction:
            _t_lst = ['lo', 'lc', 'so', 'sc']
            ax1.text(item[0], item[3], s='{0}'.format(item[2]))

        ax2 = ax1.twinx()
        # ax2.plot(_idx_lst[PLT_START:PLT_END], factor.slope[PLT_START:PLT_END], 'r')

        _ret_path = utils.get_path([define.RESULT_DIR, define.BT_DIR, '{0}_{1}.jpg'.format(instrument_id, trade_date)])
        plt.savefig(_ret_path)
    trans_df = pd.DataFrame(account.transaction,
                            columns=['idx', 'instrument_id', 'direction', 'last_price', 'fill_price', 'open_price',
                                     'fee', 'open_ts',
                                     'close_ts', 'holding_time',
                                     'return'])
    _trans_path = utils.get_path(
        [define.RESULT_DIR, define.BT_DIR, 'trans_{0}_{1}.csv'.format(instrument_id, trade_date)])
    trans_df.to_csv(_trans_path, index=False)
    long_open, short_open, correct_long_open, wrong_long_open, correct_short_open, wrong_short_open = 0, 0, 0, 0, 0, 0
    total_fee = 0.0
    total_holding_time = 0.0
    max_holding_time = -np.inf
    min_holding_time = np.inf
    for item in account.transaction:
        if item[2] == '00':
            long_open += 1
        elif item[2] == '10':
            short_open += 1
        elif item[2] == '01':
            total_holding_time += item[-2]
            max_holding_time = max(max_holding_time, item[-2])
            min_holding_time = min(min_holding_time, item[-2])
            if item[-1] > 0:
                correct_long_open += 1
            else:
                wrong_long_open += 1
        else:  # 11
            total_holding_time += item[-2]
            max_holding_time = max(max_holding_time, item[-2])
            min_holding_time = min(min_holding_time, item[-2])
            if item[-1] > 0:
                correct_short_open += 1
            else:
                wrong_short_open += 1
    average_holding_time = total_holding_time / (long_open + short_open) if long_open + short_open > 0 else 0.0
    print('trade date', trade_date)
    print(long_open, short_open, correct_long_open, wrong_long_open, correct_short_open, wrong_short_open)
    print('total return:', total_return)
    print('total fee:', account.fee)
    print('total risk:', total_risk)
    print('update factor time:', update_factor_time)
    print('get signal time:', get_signal_time)
    print("average_holding_time:", average_holding_time)
    print("max_holding_time:", max_holding_time)
    print("min_holding_time:", min_holding_time)
    print('-------------------------------------------')

    precision = (correct_long_open + correct_short_open) / (
            long_open + short_open) if long_open + short_open > 0 else 0.0
    # f = open("results/results_{0}.txt".format(product_id), "a")
    # _key = '{0}_{1}'.format(trade_date, instrument_id)
    # result_fname_digest = hashlib.sha256(bytes(backtesting_config, encoding='utf-8')).hexdigest()
    # f.write("{0}:{1}\n".format(backtesting_config, result_fname_digest))
    _ret_path = utils.get_path([define.RESULT_DIR, define.BT_DIR, '{0}.csv'.format(result_fname_digest)])
    try:
        result_df = pd.read_csv(_ret_path)
    except Exception as ex:
        result_df = pd.DataFrame(
            {'trade_date': [], 'product_id': [], 'instrument_id': [], 'total_return_after_fee': [],
             'total_return_risk': [],
             'total_fee': [],
             'total_risk': [], 'precision': [], 'long_open': [], 'short_open': [],
             'correct_long_open': [], 'wrong_long_open': [], 'correct_short_open': [], 'wrong_short_open': [],
             'average_holding_time': [], 'max_holding_time': [], 'min_holding_time': []
             })

    result_df = result_df.append(
        {'trade_date': trade_date, 'product_id': product_id, 'instrument_id': instrument_id,
         'total_return_after_fee': total_return, 'total_return_risk': total_return_risk, 'total_fee': account.fee,
         'total_risk': total_risk, 'precision': precision,
         'long_open': long_open, 'short_open': short_open, 'correct_long_open': correct_long_open,
         'wrong_long_open': wrong_long_open, 'correct_short_open': correct_short_open,
         'wrong_short_open': wrong_short_open, 'average_holding_time': average_holding_time,
         'max_holding_time': max_holding_time, 'min_holding_time': min_holding_time
         }, ignore_index=True)

    result_df.to_csv(_ret_path, index=False)
    ret = (total_return, account.fee, precision)
    return ret
