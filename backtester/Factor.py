#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 14:03
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : Factor.py

import numpy as np
import pandas
import math
import utils.utils as utils
import utils.define as define
import collections
import copy


class Factor(object):
    def __init__(self, product_id='m', instrument_id='', trade_date='', long_windows=600, short_windows=120):
        # same variables in live
        self.last_price = collections.deque(long_windows)  # xxx_circular_ptr
        self.mid_price = collections.deque(long_windows)
        self.spread = collections.deque(long_windows)
        self.buy_vol = collections.deque(long_windows)
        self.sell_vol = collections.deque(long_windows)
        self.turnover = collections.deque(long_windows)
        self.vol = collections.deque(long_windows)
        self.long_windows = long_windows
        self.short_windows = short_windows
        self.curr_factor = []
        self.last_factor = []
        self.last_mkt = []

        self.product_id = product_id
        self.instrument_id = instrument_id
        self.trade_date = trade_date

        self.vwap = []
        self.upper_bound = []
        self.lower_bound = []
        self.open_turnover = -np.inf
        self.open_vol = -np.inf
        self.product_id = product_id
        self.update_time = []
        self.turning = []
        self.turning_idx = []
        self.slope = []
        # this only calculate by time stamp

        self.b2s_vol = []
        self.b2s_turnover = []
        self.turnover = []
        self.price_highest = []
        self.price_lowest = []
        self.ma10 = []
        self.ma20 = []
        self.ma60 = []
        self.ma120 = []
        self.mid_price = []
        self.trend_ls_ratio = []
        self.cos = []
        self.trend_long = []
        self.trend_short = []
        self.log_return = []
        self.log_return10 = []
        self.log_return20 = []
        self.log_return40 = []
        self.log_return60 = []
        self.mid_log_return = []
        self.vwap_ls_diff = []

    def update_factor(self, tick=[], *args, **kwa):
        '''

        :param tick: live的depth market
        :param args:
        :param kwa:
        :return:
        '''
        last_price = tick[3]
        vol = tick[7]  # 当前成交量，和live 不同
        turnover = tick[6]  # 当前成交额，和live不同
        _update_time = tick[2]
        bid_price1, ask_price1, bid_vol1, ask_vol1 = tick[-4:]

        # same var with live
        _curr_last = tick[3]
        _curr_vol = tick[7]  # 当前成交量，和CTP live不同
        _curr_interest = tick[4]  # 当前持仓，和CTP live 不同
        _curr_turnover = tick[6]  # 当前成交额，和CTP live不同
        _prev_max = _curr_last
        _prev_min = _curr_last
        _prev_vwap = _curr_last
        _log_return = 0.0
        _mid_log_return = 0.0
        _log_return_long = 0.0
        _log_return_short = 0.0
        _mid_log_return_short = 0.0
        _mid_log_return_long = 0.0
        _ma_short = 0.0
        _ma_long = 0.0
        _ma_ls_diff_last = 0.0
        _ma_ls_diff_curr = 0.0
        # already included in the hist data
        tick_type = tick[10]
        tick_direction = tick[11]
        _vol_buy = 0.0
        _vol_sell = 0.0
        _interest_open = 0.0
        _interest_close = 0.0
        _turnover_long = 0.0
        _turnover_short = 0.0
        _vol_long = 0.0
        _vol_short = 0.0

        _curr_spread = ask_price1 - bid_price1
        _curr_vwap = turnover / vol  # FIXME check the calculation with live
        _curr_mid = (ask_price1 * bid_vol1 + bid_price1 * ask_vol1) / (ask_vol1 + bid_vol1)

        # update from last factor, start from the 2nd tick
        if len(self.last_factor) > 0:
            _prev_max = self.last_factor[1]
            _prev_min = self.last_factor[2]
            _prev_vwap = self.last_factor[5]
            _log_return = math.log(_curr_last) - math.log(self.last_factor[0])
            _mid_log_return = math.log(_curr_mid) - math.log(self.last_factor[4])

        cir_size = len(self.last_price)
        _curr_max = max(_prev_max, _curr_last)
        _curr_min = min(_prev_min, _curr_last)

        if len(self.last_mkt) > 0:
            pass

        if cir_size >= self.long_windows:
            _log_return_long = self.last_factor[8] + _log_return - math.log(self.last_price[1]) - math.log(
                self.last_price[1])
            _mid_log_return_long = self.last_factor[10] + _mid_log_return - math.log(self.mid_price[1]) - math.log(
                self.mid_price[1])
            _log_return_short = self.last_factor[7] + _log_return - math.log(self.last_price[1]) - math.log(
                self.last_price[0])
            _mid_log_return_short = self.last_factor[9] + _mid_log_return - math.log(self.mid_price[1]) - math.log(
                self.mid_price[0])
            _turnover_long = sum(self.turnover[-self.long_windows:])  # TODO different from live
            _turnover_short = sum(self.turnover[-self.short_windows:])  # TODO
            _vol_long = sum(self.vol[-self.long_windows:])
            _vol_short = sum(self.vol[-self.short_windows:])
            _ma_long = _turnover_long / _vol_long
            _ma_short = _turnover_short / _vol_short
            _ma_ls_diff_curr = _ma_short - _ma_long
            _ma_ls_diff_last = self.last_factor[11] - self.last_factor[12]
        elif cir_size >= self.short_windows:
            _ma_long = _curr_vwap
            _log_return_short = self.last_factor[7] + _log_return - math.log(self.last_price[1]) - math.log(
                self.last_price[0])

            _mid_log_return_short = self.last_factor[9] + _mid_log_return - math.log(self.mid_price[1]) - math.log(
                self.mid_price[0])
            _ma_short = _curr_vwap
        else:
            _ma_long = _curr_vwap
            _ma_short = _curr_vwap

        # time series cached factor
        self.last_price.append(_curr_last)
        self.spread.append(_curr_spread)
        self.mid_price.append(_curr_mid)
        self.turnover.append(_curr_turnover)
        self.vol.append(_curr_vol)

        ret_factor = copy.deepcopy(self.curr_factor)
        # currentfactorupdate
        self.curr_factor.clear()
        self.curr_factor.append(_curr_last)  # factor vector:0
        self.curr_factor.append(_curr_max)  # vector: 1;factor: 5
        self.curr_factor.append(_curr_min)  # vector: 2;factor: 6
        self.curr_factor.append(_curr_spread)  # vector: 3;factor: 7
        self.curr_factor.append(_curr_mid)  # vector: 4;factor: 8
        self.curr_factor.append(_curr_vwap)  # vector: 5;factor: 9
        self.curr_factor.append(_log_return)  # vector: 6;factor: 10
        self.curr_factor.append(_log_return_short)  # vector: 7;factor: 11
        self.curr_factor.append(_log_return_long)  # vector: 8;factor: 12
        self.curr_factor.append(_mid_log_return_short)  # vector: 9;factor: 13
        self.curr_factor.append(_mid_log_return_long)  # vector: 10;factor: 14
        self.curr_factor.append(_ma_short)  # vector: 11; factor: 15
        self.curr_factor.append(_ma_long)  # // vector: 12;        factor: 16
        self.curr_factor.append(_curr_vol)  # vector: 13;factor: 17
        self.curr_factor.append(_curr_interest)  # vector14; factor: 18

        self.last_factor.clear()
        for item in self.curr_factor:
            self.last_factor.append(item)
        self.last_mkt.clear()
        self.last_mkt.append(ask_price1)  # 0: askprice 1
        self.last_mkt.append(bid_price1)  # 1:bidprice 1
        self.last_mkt.append(_curr_vol)  # 2 vol TODO  VOL is different
        self.last_mkt.append(_curr_interest)  # 3, TODO interest is diff
        self.last_mkt.append(_vol_buy)
        self.last_mkt.append(_vol_sell)
        self.last_mkt.append(_curr_interest)  # TODO interest is diff
        return ret_factor

    # @timeit
    def update_factor_bt(self, tick=[], *args, **kwargs):
        # values from tick
        last_price = tick[3]
        vol = tick[7]
        turnover = tick[6]
        _update_time = tick[2]
        bid_price1, ask_price1, bid_vol1, ask_vol1 = tick[-4:]

        _mid = (bid_price1 * ask_vol1 + ask_price1 * bid_vol1) / (ask_vol1 + bid_vol1)
        self.mid_price.append(_mid)
        _range = kwargs.get('range') or 20
        k1 = kwargs.get('k1') or 0.2
        k2 = kwargs.get('k2') or 0.2
        idx = kwargs.get('idx') or 0
        lag_long = kwargs.get('lag_long') or 60
        lag_short = kwargs.get('lag_short') or 20
        _mul_num = kwargs.get('multiplier') or 1.0
        self.open_turnover = turnover if self.open_turnover < 0 else self.open_turnover
        self.open_vol = vol if self.open_vol < 0 else self.open_vol
        self.last_price.append(last_price)
        self.update_time.append(_update_time)
        # "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1",

        self.spread.append(ask_price1 - bid_price1)
        self.b2s_vol.append(bid_vol1 / ask_vol1 if ask_vol1 > 0 else 1.0)
        self.b2s_turnover.append(
            ((bid_vol1 * bid_price1) / (ask_vol1 * ask_price1) if ask_vol1 * ask_price1 > 0 else 1.0))
        # self.price_highest.append(max(self.last_price))
        # self.price_lowest.append(min(self.last_price))
        _vwap_val = None
        try:
            # if _update_time >= '21:00:00':
            _tmp_vol = vol
            _tmp_turnover = turnover
            self.vol.append(_tmp_vol)
            self.turnover.append(_tmp_turnover)
            _vwap_val = (_tmp_turnover) / (_mul_num * (_tmp_vol))

        except Exception as ex:
            if self.vwap:
                _vwap_val = self.vwap[-1]
            else:
                _vwap_val = last_price
        self.vwap.append(_vwap_val)
        self.upper_bound.append(_vwap_val + k1 * _range)
        self.lower_bound.append(_vwap_val - k2 * _range)

        # ma for fixed length
        self.ma10.append(sum(self.last_price[-10:]) / len(self.last_price[-10:]))
        self.ma20.append(sum(self.last_price[-20:]) / len(self.last_price[-20:]))
        self.ma60.append(sum(self.last_price[-60:]) / len(self.last_price[-60:]))
        self.ma120.append(sum(self.last_price[-120:]) / len(self.last_price[-120:]))

        if len(self.turning) <= 1:
            self.turning.append(last_price)
            self.turning_idx.append(idx)
        else:
            _last_turn = self.turning[-1]
            _last_turn1 = self.turning[-2]
            _last_price = self.last_price[-2]
            _last_price1 = self.last_price[-3]
            if (last_price - _last_price) * (_last_price - _last_price1) <= 0:  # new turning
                self.turning.append(_last_price)
                self.turning_idx.append(idx - 1)
            else:  # last turning
                self.turning.append(_last_turn)
                self.turning_idx.append(self.turning_idx[-1])
        # print(self.turning)
        if not self.slope:
            self.slope.append(0.5)
        else:
            if idx == self.turning_idx[-1]:
                self.slope.append(self.slope[-1])
            else:
                self.slope.append((last_price - self.turning[-1]) / (idx - self.turning_idx[-1]))

        # factor for lag windows,
        try:
            if len(self.last_price) >= lag_long:  # long windows
                _trend_long = (self.last_price[-1] - self.last_price[-lag_long]) / lag_long
                _trend_short = (self.last_price[-1] - self.last_price[-lag_short]) / lag_short
                self.trend_long.append(_trend_long)
                self.trend_short.append(_trend_short)
                self.trend_ls_ratio.append(
                    self.trend_short[-1] / self.trend_long[-1] if self.trend_long[-1] != 0 else np.nan)

                _ls_diff = sum(self.turnover[lag_long:]) / sum(self.vol[lag_long:]) - sum(
                    self.turnover[lag_short:]) / sum(
                    self.vol[lag_short:])
                # _ls_diff = np.array(self.last_price[lag_long:]).mean() - np.array(self.last_price[lag_short:]).mean()
                self.vwap_ls_diff.append(_ls_diff)


            elif len(self.last_price) >= lag_short:  # short windows
                _trend_short = (self.last_price[-1] - self.last_price[-lag_short]) / lag_short
                self.trend_short.append(_trend_short)
                self.trend_long.append(np.nan)
                self.trend_ls_ratio.append(np.nan)
                self.vwap_ls_diff.append(np.nan)

            else:
                self.trend_long.append(np.nan)
                self.trend_short.append(np.nan)
                self.trend_ls_ratio.append(np.nan)
                self.vwap_ls_diff.append(np.nan)

        except Exception as ex:
            print(ex)

        a = np.array([idx - self.turning_idx[-1], self.last_price[-1] - self.last_price[self.turning_idx[-1]]])
        b = np.array([0, 1])
        self.cos.append(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        if len(self.last_price) >= 2:
            self.log_return.append(math.log(self.last_price[-1]) - math.log(self.last_price[-2]))
        else:
            self.log_return.append(np.nan)
        if len(self.mid_price) >= 2:
            self.mid_log_return.append(math.log(self.mid_price[-1]) - math.log(self.mid_price[-2]))
        else:
            self.mid_log_return.append(np.nan)
        if len(self.log_return) > 10:
            self.log_return10.append(sum(self.log_return[-10:]))
        else:
            self.log_return10.append(np.nan)
        if len(self.last_price) != len(self.trend_long):
            print('check')

    def load_factor(self):
        pass

    def get_factor(self):
        return {'slope': self.slope[-1],
                'vwap': self.vwap[-1],
                'spread': self.spread[-1],
                'b2s_vol': self.b2s_vol[-1],
                'cos': self.cos[-1],
                'trend_ls_ratio': self.trend_ls_ratio[-1],
                'log_return': self.log_return[-1],
                'log_return_0': self.log_return10[-1],
                'wap_log_return': self.mid_log_return[-1],

                }

    def cache_factor(self):
        # factor_df = pandas.DataFrame({'last_price': self.last_price, 'vwap': self.vwap, 'upper_bound': self.upper_bound,
        #                               'lower_bound': self.lower_bound, 'update_time': self.update_time,
        #                               'turning': self.turning,
        #                               'slope': self.slope, 'turning_idx': self.turning_idx, 'spread': self.spread,
        #                               'vol_ratio': self.vol_ratio, 'price_highest': self.price_highest,
        #                               'price_lowest': self.price_lowest,
        #                               'curr_vol': self.curr_vol, 'curr_turnover': self.curr_turnover})
        factor_df = pandas.DataFrame({'last_price': self.last_price, 'vwap': self.vwap, 'upper_bound': self.upper_bound,
                                      'lower_bound': self.lower_bound, 'update_time': self.update_time,
                                      'turning': self.turning,
                                      'slope': self.slope, 'turning_idx': self.turning_idx, 'spread': self.spread,
                                      'b2s_vol': self.b2s_vol,
                                      'b2s_turnover': self.b2s_turnover, 'vol': self.vol,
                                      'turnover': self.turnover,
                                      # 'price_high': self.price_highest,
                                      # 'price_low': self.price_lowest
                                      })
        _factor_path = utils.get_path([define.CACHE_DIR, define.FACTOR_DIR,
                                       'factor_{0}_{1}.csv'.format(self.instrument_id, self.trade_date)])
        factor_df.to_csv(_factor_path, index=False)
