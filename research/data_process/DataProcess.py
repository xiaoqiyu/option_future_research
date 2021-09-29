#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/29 10:33
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : DataProcess.py

from ...utils.define import *
import os
import pandas as pd
from datetime import datetime
import shutil


def cache_depth_mkt():
    lst = os.listdir('C:\projects\pycharm\option_future_research\cache\mkt_pycache')
    for item in lst:
        _tmp = item.split('.')[0].split('_')
        try:
            os.system('mkdir C:\projects\pycharm\option_future_research\cache\\{0}'.format(_tmp[-1]))
            shutil.copy('C:\projects\pycharm\option_future_research\cache\mkt_pycache\{0}'.format(item),
                        'C:\projects\pycharm\option_future_research\cache\{0}\{1}.csv'.format(_tmp[-1], _tmp[0]))
        except Exception as ex:
            print(ex)
    # os.system('rm -r C:\projects\pycharm\option_future_research\cache\\mkt_pycache')


def prepare_test_mkt():
    with open('../../utils/trade_dates.txt', 'r') as f:
        lines = f.readlines()
        dates = [item.strip() for item in lines]
        # create folders
        for item in dates:
            try:
                os.system('mkdir C:\projects\pycharm\option_future_research\cache\\{0}'.format(item))
            except Exception as ex:
                print("folder exist", item)


def transaction_analysis():
    df = pd.read_csv('results/trans_m2105_20210104.csv')
    long_open_ts = list(df[df.direction == 0]['timestamp'])
    long_close_ts = list(df[df.direction == 1]['timestamp'])
    short_open_ts = list(df[df.direction == 10]['timestamp'])
    short_close_ts = list(df[df.direction == 11]['timestamp'])

    assert len(long_open_ts) == len(long_close_ts)
    assert len(short_open_ts) == len(short_close_ts)

    long_tran_time = []
    for idx, item in enumerate(long_open_ts):
        _sec = datetime.strptime(long_close_ts[idx], '%H:%M:%S') - datetime.strptime(item, '%H:%M:%S')
        long_tran_time.append(_sec.seconds)
    short_tran_time = []
    for idx, item in enumerate(short_open_ts):
        _sec = datetime.strptime(short_close_ts[idx], '%H:%M:%S') - datetime.strptime(item, '%H:%M:%S')
        print(idx, _sec.seconds, item, short_close_ts[idx])
        short_tran_time.append(_sec.seconds)
    # print(long_tran_time)
    # print(short_tran_time)
    # print(max(long_tran_time)/60, min(long_tran_time))
    # print(max(short_tran_time)/60, min(short_tran_time))


if __name__ == '__main__':
    cache_depth_mkt()
