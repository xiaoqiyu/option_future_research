#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 16:28
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : Signal.py

import abc
import utils.define as define


class SignalField(object):
    def __init__(self):
        self._signal_type = define.NO_SIGNAL
        self._vol = 0
        self._price = 0
        self._direction = define.LONG

    @property
    def signal_type(self):
        return self._signal_type

    @signal_type.setter
    def signal_type(self, val):
        self._signal_type = val

    @property
    def vol(self):
        return self._vol

    @vol.setter
    def vol(self, val):
        self._vol = val

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, val):
        self._price = val

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, val):
        self._direction = val


class Signal(object):
    def __init__(self, factor, position):
        self.factor = factor
        self.position = position

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass
