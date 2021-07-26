#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/22 21:00
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : main.py


from BackTester import backtesting


def search_label(lst=[], win_len=5):
    ret = [0.0]
    for idx, item in enumerate(lst):
        if idx < 1:
            continue
        ret.append(item / lst[idx - 1] - 1)
        if len(ret) < 5:
            continue
        # if ret[idx-1] <0 and ret[idx-2]<0 and ret[idx-3]<0:


if __name__ == "__main__":
    # backtesting(product_id='ru', trade_date='20210401', prev_date='2021-03-31', volitily=18.0, k1=0.2, k2=0.2,
    #             stop_profit=100.0, fee=3.0)
    backtesting(product_id='m', trade_date='20210401', prev_date='2021-03-31')
