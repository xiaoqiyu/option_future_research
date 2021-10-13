#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/28 16:15
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : RegSignal.py

import numpy as np
import pandas as pd
from strategy.Signal import Signal
import utils.define as define
import utils.utils as utils
from backtester.Factor import Factor
from backtester.Position import Position


class RegSignal(Signal):
    def __init__(self, factor, position):
        super().__init__(factor, position)
        _path = utils.get_path(
            [define.RESULT_DIR, define.TICK_MODEL_DIR, 'df_pred_2021-07-26_120_60.csv'])
        df = pd.read_csv(_path)
        self.map_dict = dict(zip([item.split()[1].split('.')[0] for item in df['UpdateTime']], df['pred']))

    def get_signal(self, params={}):
        _k = params.get('tick')[2].split()[1].split('.')[0]
        _v = self.map_dict.get(_k) or 0.0
        if _v >= 0.001:
            return define.LONG_OPEN
        elif _v <= -0.001:
            return define.SHORT_OPEN
        instrument_id = params.get('instrument_id')
        start_tick = int(params.get('start_tick')) or 2
        stop_profit = float(params.get('stop_profit')) or 5.0
        stop_loss = float(params.get('stop_loss')) or 20.0
        multiplier = int(params.get('multiplier')) or 10

        open_fee = float(params.get('open_fee')) or 1.51
        close_to_fee = float(params.get('close_t0_fee')) or 0.0
        fee = (open_fee + close_to_fee) / multiplier
        _position = self.position.get_position(instrument_id)

        # if params.get('tick')[2].split()[1] < '21:00:00.000':
        #     print('check')

        if len(self.factor.last_price) < start_tick:
            return define.NO_SIGNAL
        _long, _short, long_price, short_price = 0, 0, 0.0, 0.0

        if _position:
            for item in _position:
                if item[0] == define.LONG:
                    long_price = item[1]
                    _long += 1
                elif item[0] == define.SHORT:
                    short_price = item[1]
                    _short += 1

        if _long and (self.factor.last_price[-1] > long_price + stop_profit + fee or
                      self.factor.last_price[-1] < long_price - stop_loss - fee):
            return define.LONG_CLOSE
        if _short and (
                short_price > self.factor.last_price[-1] + stop_profit + fee or self.factor.last_price[
            -1] > short_price + stop_loss + fee):
            return define.SHORT_CLOSE
        return define.NO_SIGNAL
