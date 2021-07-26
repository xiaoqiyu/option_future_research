#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 14:03
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : Position.py

from collections import defaultdict


class Position(object):
    def __init__(self):
        self.position = defaultdict(list)

    def open_position(self, instrument_id, long_short, price, timestamp):
        self.position[instrument_id].append([long_short, price])

    def close_position(self, instrument_id, long_short, price, timestamp):
        _lst = self.position.get(instrument_id) or []
        for item in _lst:
            if item[0] == long_short:
                _lst.remove(item)

    def get_position(self, instrument_id):
        return self.position.get(instrument_id)

    def get_position_side(self, instrument_id, side):
        _pos_lst = self.position.get(instrument_id)
        for item in _pos_lst:
            if item[0] == side:
                return item
        return
