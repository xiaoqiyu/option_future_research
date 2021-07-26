#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 14:03
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : Factor.py



class Factor(object):
    def __init__(self, product_id='m'):
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
        self.vol_ratio = []
        self.price_highest = []
        self.price_lowest = []
        self.curr_vol = []
        self.curr_turnover = []

    def update_factor(self, tick=[], *args, **kwargs):
        last_price = tick[1]
        vol = tick[5]
        turnover = tick[6]
        _update_time = tick[10]
        _range = kwargs.get('range') or 20
        k1 = kwargs.get('k1') or 0.2
        k2 = kwargs.get('k2') or 0.2
        idx = kwargs.get('idx') or 0
        self.open_turnover = turnover if self.open_turnover < 0 else self.open_turnover
        self.open_vol = vol if self.open_vol < 0 else self.open_vol
        self.last_price.append(last_price)
        self.update_time.append(tick[10])

        _vwap_val = None
        try:
            if _update_time >= '21:00:00':
                _vwap_val = (turnover) / (dict_multiplier.get(self.product_id) * (vol))
            else:
                _vwap_val = (turnover - self.open_turnover) / (
                        dict_multiplier.get(self.product_id) * (vol - self.open_vol))
        except Exception as ex:
            if self.vwap:
                _vwap_val = self.vwap[-1]
            else:
                _vwap_val = last_price
        self.vwap.append(_vwap_val)
        self.upper_bound.append(_vwap_val + k1 * _range)
        self.lower_bound.append(_vwap_val - k2 * _range)

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

    def load_factor(self):
        pass

    def cache_factor(self):
        pass