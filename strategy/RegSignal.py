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
import os


class RegSignal(Signal):
    def __init__(self, factor, position, instrument_id=None, trade_date=None):
        super().__init__(factor, position)
        trade_date_df = utils.get_trade_dates(start_date='20210101', end_date=trade_date)
        trade_date_df = trade_date_df[trade_date_df.exchangeCD == 'XSHE']
        trade_dates = list(trade_date_df['calendarDate'])
        prev_date = trade_dates[-2]
        _evalute_path = utils.get_path([define.RESULT_DIR, define.TICK_MODEL_DIR,
                                        'model_evaluate.json'])
        _model_evaluate = utils.load_json_file(_evalute_path)
        _ret_lst = _model_evaluate.get('{0}_{1}'.format(instrument_id, trade_date.replace('-', ''))) or []
        _ret_lst.sort(key=lambda x: x[1], reverse=True)
        mse, r2, date, predict_window, lag_window = _ret_lst[0]
        _path = utils.get_path(
            [define.RESULT_DIR, define.TICK_MODEL_DIR,
             'pred_{0}_{1}_{2}_{3}.csv'.format(instrument_id, date.replace('-', ''), predict_window, lag_window)])
        df = pd.read_csv(_path)
        _pred_lst = sorted(list(df['pred']))
        self.map_dict = dict(zip([item.split()[1].split('.')[0] for item in df['UpdateTime']], df['pred']))
        _prev_path = utils.get_path(
            [define.RESULT_DIR, define.TICK_MODEL_DIR,
             'pred_{0}_{1}_{2}_{3}.csv'.format(instrument_id, prev_date.replace('-', ''), predict_window, lag_window)])
        self.df_prev = pd.read_csv(_prev_path)
        _label_lst = sorted(list(self.df_prev['pred']))
        # self._ret_up = self.df_prev['label'].quantile(1 - 0.0001)
        # self._ret_down = self.df_prev['label'].quantile(0.002)
        self._ret_up = _label_lst[-50]
        self._ret_down = _label_lst[20]
        print('up ret:{0}, down ret:{1},up pred:{2}, down pred:{3}'.format(self._ret_up, _pred_lst[-50], self._ret_down,
                                                                           _pred_lst[50]))

    def get_signal(self, params={}):
        _k = params.get('tick')[2].split()[1].split('.')[0]
        _v = self.map_dict.get(_k) or 0.0
        _up_ratio = float(params.get('ret_up_ratio')) or 0.0015
        _down_ratio = float(params.get('ret_down_ratio')) or 0.001

        # _ret_down = float(params.get('ret_down_ratio')) or 0.001
        _long, _short, long_price, short_price = 0, 0, 0.0, 0.0
        instrument_id = params.get('instrument_id')
        start_tick = int(params.get('start_tick')) or 2
        stop_profit = float(params.get('stop_profit')) or 5.0
        stop_loss = float(params.get('stop_loss')) or 20.0
        multiplier = int(params.get('multiplier')) or 10
        long_lots_limit = int(params.get('long_lots_limit')) or 1
        short_lots_limit = int(params.get('short_lots_limit')) or 1

        open_fee = float(params.get('open_fee')) or 1.51
        close_to_fee = float(params.get('close_t0_fee')) or 0.0
        fee = (open_fee + close_to_fee) / multiplier
        _position = self.position.get_position(instrument_id)
        if _position:
            for item in _position:
                if item[0] == define.LONG:
                    long_price = item[1]
                    _long += 1
                elif item[0] == define.SHORT:
                    short_price = item[1]
                    _short += 1
        if _v >= self._ret_up and _long < long_lots_limit:
            return define.LONG_OPEN
        elif _v <= self._ret_down and _short < short_lots_limit and _v < 0:
            return define.SHORT_OPEN

        if len(self.factor.last_price) < start_tick:
            return define.NO_SIGNAL

        if _long and (self.factor.last_price[-1] > long_price + stop_profit + fee or
                      self.factor.last_price[-1] < long_price - stop_loss - fee):
            return define.LONG_CLOSE
        if _short and (
                short_price > self.factor.last_price[-1] + stop_profit + fee or self.factor.last_price[
            -1] > short_price + stop_loss + fee):
            return define.SHORT_CLOSE
        return define.NO_SIGNAL
