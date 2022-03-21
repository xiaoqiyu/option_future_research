#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 14:03
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : Factor.py

import numpy as np
import pandas
import utils.utils as utils
import utils.define as define
import os


class Factor(object):
    def __init__(self, product_id='m', instrument_id='', trade_date=''):
        self.product_id = product_id
        self.instrument_id = instrument_id
        self.trade_date = trade_date
        self.last_price = []
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

        self.spread = []
        self.b2s_vol = []
        self.b2s_turnover = []
        self.vol = []
        self.turnover = []
        self.price_highest = []
        self.price_lowest = []
        # self.curr_vol = []
        # self.curr_turnover = []
        self.ma10 = []
        self.ma20 = []
        self.ma60 = []
        self.ma120 = []
        self.mid_price = []
        self.trend_ls_ratio = []
        self.cos = []
        self.trend_long = []
        self.trend_short = []

    # @timeit
    def update_factor(self, tick=[], *args, **kwargs):
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

        try:
            if len(self.last_price) >= lag_long:
                _trend_long = (self.last_price[-1] - self.last_price[-lag_long]) / lag_long
                _trend_short = (self.last_price[-1] - self.last_price[-lag_short]) / lag_short
                self.trend_long.append(_trend_long)
                self.trend_short.append(_trend_short)
                self.trend_ls_ratio.append(
                    self.trend_short[-1] / self.trend_long[-1] if self.trend_long[-1] != 0 else np.nan)
            elif len(self.last_price) >= lag_short:
                _trend_short = self.last_price[-1] - self.last_price[-lag_short]
                self.trend_short.append(_trend_short)
                self.trend_ls_ratio.append(np.nan)
            else:
                self.trend_ls_ratio.append(np.nan)
        except Exception as ex:
            print(ex)

        a = np.array([idx - self.turning_idx[-1], self.last_price[-1] - self.last_price[self.turning_idx[-1]]])
        b = np.array([0, 1])
        self.cos.append(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def load_factor(self):
        pass

    def get_factor(self):
        return {'slope': self.slope[-1],
                'vwap': self.vwap[-1],
                'spread': self.spread[-1],
                'b2s_vol': self.b2s_vol[-1],
                'cos': self.cos[-1],
                'trend_ls_ratio': self.trend_ls_ratio[-1]
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
