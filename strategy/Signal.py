#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 16:28
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : Signal.py

import abc


class Signal(object):
    def __init__(self, factor, position):
        self.factor = factor
        self.position = position

    @abc.abstractmethod
    def get_signal(self, *args, **kwargs):
        pass
