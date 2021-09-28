#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 14:04
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : define.py


import os
import re

NO_SIGNAL = -1
LONG_OPEN = 0
LONG_CLOSE = 1
SHORT_OPEN = 2
SHORT_CLOSE = 3
LONG = 4
SHORT = 5
PLT_START = 3
PLT_END = -10
TICK_SIZE = 41400

cols = ["InstrumentID", "LastPrice", "OpenPrice", "HighestPrice", "LowestPrice", "Volume", "Turnover", "OpenInterest",
        "UpperLimitPrice", "LowerLimitPrice", "UpdateTime",
        "UpdateMillisec",
        "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1", "BidPrice2", "BidVolume2", "AskPrice2", "AskVolume2",
        "BidPrice3", "BidVolume3",
        "AskPrice3", "AskVolume3", "BidPrice4", "BidVolume4", "AskPrice4", "AskVolume4", "BidPrice5", "BidVolume5",
        "AskPrice5", "AskVolume5"]

selected_cols = ['InstrumentID', 'UpdateTime', 'Turnover', 'Volume', 'LastPrice', 'AskPrice1', 'AskVolume1',
                 'BidPrice1', 'BidVolume1']

tb_cols = ["Exchange", "InstrumentID", "UpdateTime", "LastPrice", "OpenInterest", "InterestDiff", "Turnover",
           "Volume", "OpenVolume", "CloseVolume", "TransactionType", "Direction", "BidPrice1", "AskPrice1",
           "BidVolume1",
           "AskVolume1"]

dict_multiplier = {'m': 10, 'i': 100, 'TA': 1, 'ru': 10, }
file_name = '/.editorconfig'
factor_file_path = 'C:\projects\pycharm\option_future_research\cache\\factors'
daily_cache_name = 'cache/future_20210101_20210804.csv'


def get_trade_dates():
    with open('../cache/dates.txt', 'rb') as f:
        dates = f.readlines()
        lst = os.listdir('C:\projects\pycharm\option_future_research\cache')
        dates = []
        for d in lst:
            _s, _e = re.match(r'\d*', d).span()
            if _s != _e:
                dates.append(d)

    return sorted(dates)
