#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/29 15:50
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : utils.py

import time
import json
import math
import uqer
import pprint
import numpy as np
from uqer import DataAPI
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import utils.define as define
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
import talib as ta
import pandas as pd
import statsmodels.api as sm
from editorconfig import get_properties, EditorConfigError
import logging
import os
import gc

try:
    print("current path in utils:", os.getcwd())
    # _conf_file = os.path.join(os.path.abspath(os.pardir), define.CONF_DIR,
    #                           define.CONF_FILE_NAME)
    _conf_file = os.path.join(os.path.abspath(os.getcwd()), define.CONF_DIR, define.CONF_FILE_NAME)
    options = get_properties(_conf_file)
except EditorConfigError:
    logging.warning("Error getting EditorConfig propterties", exc_info=True)
else:
    for key, value in options.items():
        # _config = '{0},{1}:{2}'.format(_config, key, value)
        print("{0}:{1}".format(key, value))
print(options.get('uqer_token'))
uqer_client = uqer.Client(token=options.get('uqer_token'))


def get_instrument_ids(start_date='20210901', end_date='20210930', product_id='RB'):
    df = DataAPI.MktFutdGet(endDate=end_date, beginDate=start_date, pandas="1")
    df = df[df.mainCon == 1]
    df = df[df.contractObject == product_id.upper()][['ticker', 'tradeDate', 'exchangeCD']]
    return df


def get_trade_dates(start_date='20110920', end_date='20210921'):
    df = DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE", beginDate=start_date, endDate=end_date, isOpen=u"1",
                             field=u"",
                             pandas="1")
    df = df[df.isOpen == 1]
    return df


def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        # logger.info('%r (%r, %r) %2.2f sec' % (func.__name__, args, kwargs, te - ts))
        print('%r (%r, %r) %2.2f sec' % (func.__name__, args, kwargs, te - ts))
        return result

    return


def test_time():
    for item in range(10):
        time.sleep(1)


def is_trade(start_timestamp=None, end_timestamp=None, update_time=None):
    update_time = update_time.split()[1].split('.')[0]
    if update_time > start_timestamp and update_time < '22:59:30':
        return True
    elif update_time > '09:00:30' and update_time < end_timestamp:
        return True
    else:
        return False
    return False


def get_path(ref_path_lst=[]):
    _path = os.path.join(os.path.abspath(os.getcwd()))
    # print(_path)
    for p in ref_path_lst:
        _path = os.path.join(_path, p)
    return _path


def get_contract(instrument_id=''):
    df = DataAPI.FutuGet(secID=u"", ticker=instrument_id, exchangeCD=u"", contractStatus="", contractObject=u"",
                         prodID="",
                         field=u"", pandas="1")
    return df


def get_mul_num(instrument_id=''):
    df = get_contract(instrument_id=instrument_id)
    return df['contMultNum'].values[0]


def write_json_file(file_path='', data=None):
    # if data == None:
    #     return
    with open(file_path, 'w') as outfile:
        j_data = json.dumps(data)
        outfile.write(j_data)


def load_json_file(filepath=''):
    with open(filepath) as infile:
        contents = infile.read()
        return json.loads(contents)


def get_trade_dates(start_date='20110920', end_date='20210921'):
    df = DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE", beginDate=start_date, endDate=end_date, isOpen=u"1", field=u"",
                             pandas="1")
    df = df[df.isOpen == 1]
    return df


def _is_trading_time(time_str=''):
    if time_str > '09:00:00.000' and time_str <= '15:00:00.000' or time_str > '21:00:00.000' and time_str < '23:00:01.000':
        return True
    else:
        return False


if __name__ == "__main__":
    start_ts = time.time()
    test_time()
    end_ts = time.time()
    print(end_ts - start_ts)
    # df = get_instrument_ids(start_date='20210901', end_date='20210930', product_id='RB')
    # print(df)
    df = get_contract(instrument_id='rb2201', exchangeCD=u"XSGE")
    print(df)
