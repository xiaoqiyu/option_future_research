#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 16:32
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : T0Signal.py

from strategy.Signal import Signal
from define import *
from backtester.Factor import Factor
from backtester.Position import Position


class TOSignal(Signal):
    def __init__(self, factor, position):
        super().__init__(factor, position)

    def get_signal(self, *args, **kwargs):
        assert isinstance(self.factor, Factor)
        assert isinstance(self.position, Position)
        volitility = kwargs.get('volitility') or 20.0
        k1 = kwargs.get('k1') or 1.0
        k2 = kwargs.get('k2') or 1.0
        instrument_id = kwargs.get('instrument_id')
        # adjust by multiplier
        stop_profit = kwargs.get('stop_profit') or 5.0
        stop_loss = kwargs.get('stop_loss') or 20.0
        multiplier = kwargs.get('multiplier') or 10

        open_fee = kwargs.get('open_fee') or 1.51
        close_to_fee = kwargs.get('close_t0_fee') or 0.0
        fee = (open_fee + close_to_fee)/multiplier
        start_tick = kwargs.get('start_tick') or 2
        long_lots_limit = kwargs.get('long_lots_limit') or 1
        short_lots_limit = kwargs.get('short_lots_limit') or 1
        slope_upper = kwargs.get('slope_upper') or 1.0
        slope_lower = kwargs.get('slope_lower') or -1.0
        _position = self.position.get_position(instrument_id)

        if len(self.factor.last_price) < start_tick:
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
        if (self.factor.slope[-1] > self.factor.slope[-2]) and (self.factor.slope[-2] < self.factor.slope[-3]) and abs(
                self.factor.slope[-1]) > slope_upper and _short < short_lots_limit:
            # if (factor.slope[-1] > factor.slope[-2]) and abs(
            #             factor.slope[-1]) > 0.8 and not _long:
            # print(factor.last_price, factor.slope)
            # print(_short, short_lots_limit)
            return SHORT_OPEN
        # if factor.last_price[-1] < factor.last_price[-2] and factor.last_price[-1] < factor.vwap[
        #     -1] + k2 * volitility and not _short:
        if (self.factor.slope[-1] < self.factor.slope[-2]) and (self.factor.slope[-2] > self.factor.slope[-3]) and abs(
                self.factor.slope[-1]) > slope_upper and _long < long_lots_limit:
            # if (factor.slope[-1] < factor.slope[-2]) and abs(
            #             factor.slope[-1]) > 0.8 and not _short:

            # print(factor.last_price,factor.slope)
            return LONG_OPEN
        # if factor.last_price[-1] < factor.last_price[-2] and _long and factor.last_price[-1] > long_price + stop_profit:
        if _long and (self.factor.last_price[-1] > long_price + stop_profit + fee or
                      self.factor.last_price[-1] < long_price - stop_loss - fee):
            return LONG_CLOSE
        # if factor.last_price[-1] > factor.last_price[-2] and _short and short_price > factor.last_price[-1] + stop_profit:
        if _short and (
                short_price > self.factor.last_price[-1] + stop_profit + fee or self.factor.last_price[
            -1] > short_price + stop_loss + fee):
            return SHORT_CLOSE
        return NO_SIGNAL
