#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 14:03
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : Position.py
import copyreg
from collections import defaultdict
import utils.define as define
import copy


class Position(object):
    def __init__(self):
        self.position = defaultdict(list)

    def open_position(self, instrument_id, long_short, price, timestamp, vol):
        self.position[instrument_id].append([long_short, price, timestamp, vol])

    def close_position(self, instrument_id, long_short, price, timestamp):
        _lst = self.position.get(instrument_id) or []
        for item in _lst:
            if item[0] == long_short:
                _lst.remove(item)

    def update_position(self, instrument_id, long_short, price, timestamp, vol, order_type):
        print('update position:long_short=>', long_short, 'price=>', price, 'ts=>', timestamp, 'vol=>', vol,
              'order type=>', order_type)
        _lst = self.position.get(instrument_id) or []
        update_lst = []

        print("before update position:", self.position[instrument_id])
        if not _lst:
            print('pos not exist,add possition')
            # self.open_position(instrument_id, long_short, price, timestamp, vol)
            update_lst.append([long_short, price, timestamp, vol])
        else:
            for item in _lst:
                print("before update position:", item, instrument_id, long_short, price, vol, order_type)
                _direction, _price, _ts, _vol = item
                if _direction == define.LONG and order_type == define.LONG_OPEN:
                    item[0] = define.LONG
                    item[1] = (_price * _vol + price * vol) / (_vol + vol)
                    item[2] = timestamp
                    item[3] = _vol + vol
                    update_lst.append(item)
                elif _direction == define.LONG and order_type == define.LONG_CLOSE:
                    assert _vol >= vol
                    if _vol == vol:
                        continue
                    item[1] = 0.0 if _vol == vol else (_price * _vol - price * vol) / (_vol - vol)
                    item[2] = timestamp
                    item[3] = _vol - vol
                elif _direction == define.SHORT and order_type == define.SHORT_OPEN:
                    item[0] = define.SHORT
                    item[1] = (_price * _vol + price * vol) / (_vol + vol)
                    item[2] = timestamp
                    item[3] = _vol + vol
                    update_lst.append(item)
                elif _direction == define.SHORT and order_type == define.SHORT_CLOSE:
                    assert _vol >= vol
                    if _vol == vol:
                        continue
                    item[1] = 0.0 if _vol == vol else (_price * _vol - price * vol) / (_vol - vol)
                    item[2] = timestamp
                    item[3] = _vol - vol
                else:
                    update_lst.append(item)
        self.position[instrument_id] = update_lst
        print("after update position:", self.position[instrument_id])

    def get_position(self, instrument_id):
        _lst = self.position.get(instrument_id) or []
        ret = []
        for item in _lst:
            if item[-1] > 0:
                ret.append(item)
        return ret

    def get_position_side(self, instrument_id, side):
        _pos_lst = self.position.get(instrument_id)
        if _pos_lst:
            for item in _pos_lst:
                if item[0] == side:
                    return item
        return
