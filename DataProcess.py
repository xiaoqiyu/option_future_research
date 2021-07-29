#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/29 10:33
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : DataProcess.py

from define import *
import os
import pandas as pd


def prepare_test_mkt():
    with open('trade_dates.txt', 'r') as f:
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
    print(df.head())


if __name__ == '__main__':
    transaction_analysis()
