#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 14:04
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : define.py


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
TICK = 1

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

skip_raw_cols = ['Exchange', 'InstrumentID', 'LastPrice', 'OpenInterest', 'InterestDiff',
                 'Turnover', 'Volume', 'OpenVolume', 'CloseVolume', 'TransactionType', 'Direction',
                 'BidPrice1', 'AskPrice1', 'BidVolume1', 'AskVolume1', 'vwap', 'vol_short', 'vol_long',
                 'turnover_short', 'turnover_long', 'vwap_short', 'vwap_long', 'bs_vol', 'bs_vol_long',
                 'bs_vol_short', 'bs_vol_diff', 'bs_tag', 'wap',
                 'trenddiff', 'trend_long', 'dif', 'dea', 'turnover_ls_ratio',
                 'vwap_ls_ratio', 'aoi', 'oi']
# remove from VIF, to be added with other calculation


skip_raw_cols_normalized = ['Exchange', 'InstrumentID', 'OpenInterest', 'InterestDiff',
                            'Turnover', 'Volume', 'OpenVolume', 'CloseVolume', 'TransactionType', 'Direction',
                            'BidPrice1', 'AskPrice1', 'BidVolume1', 'AskVolume1', 'vwap', 'vol_short', 'vol_long',
                            'turnover_short', 'turnover_long', 'vwap_short', 'vwap_long', 'bs_vol', 'bs_vol_long',
                            'bs_vol_short', 'bs_vol_diff', 'bs_tag']

normalized_cols = [('LastPrice', 'LastPrice'), ('wap', 'LastPrice'), ('OpenInterest', 'OpenInterest'),
                   ('vwap', 'LastPrice'), ('Volume', 'Volume'), ('BidPrice1', 'LastPrice'), ('AskPrice1', 'LastPrice')]

# normalized_cols = []

BASE_DIR = 'option_future_research'
RESULT_DIR = 'results'
CONF_DIR = 'conf'
FACTOR_DIR = 'factors'
CONF_FILE_NAME = '.editorconfig'
TICK_MODEL_DIR = 'tickmodels'
TICK_MKT_DIR = 'C:\projects\l2mkt\FutAC_TickKZ_PanKou_Daily_202107'
FACTOR_DIR = 'factors'
CACHE_DIR = 'cache'
BT_DIR = 't0backtest'
daily_cache_name = 'cache/future_20210101_20210804.csv'

MKT_MISSING_SKIP = 0.3

exchange_map = {'XZCE': 'zc', 'XSGE': 'sc', 'XSIE': 'ine', 'XDCE': 'dc'}
