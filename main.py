#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/22 21:00
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : main.py


from backtester.BackTester import backtesting
from editorconfig import get_properties, EditorConfigError
from define import *
import logging
import hashlib
import time

if __name__ == "__main__":
    product_id = 'm'
    total_return, total_fee, precision = 0.0, 0.0, []
    # test_dates = [('20210401', '2021-03-31'),
    #               ('20210402', '2021-04-01')]
    backtesting_config = ''
    try:
        options = get_properties(file_name)
    except EditorConfigError:
        logging.warning("Error getting EditorConfig propterties", exc_info=True)
    else:
        for key, value in options.items():
            backtesting_config = '{0},{1}:{2}'.format(backtesting_config, key, value)
    f = open("results/results_{0}.txt".format(product_id), "a")
    result_fname_digest = hashlib.sha256(bytes(backtesting_config, encoding='utf-8')).hexdigest()
    f.write("{0}:{1}\n".format(backtesting_config, result_fname_digest))
    with open('cache/dates.txt', 'rb') as f:
        dates = get_trade_dates()
        for idx, trade_date in enumerate(dates):
            if trade_date < '20210106':
                continue
            if idx < 1:
                continue
            start_ts = time.time()
            ret = backtesting(product_id=product_id, trade_date=trade_date.strip(), prev_date=dates[idx - 1].strip())
            end_ts = time.time()
            print("back testing time:", end_ts - start_ts)
            if ret:
                _total_return, _total_fee, _precision = ret
                total_return += _total_return
                total_fee += _total_fee
            break
        print(total_return, total_fee, precision)
