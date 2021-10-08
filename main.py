#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/22 21:00
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : main.py


from backtester.BackTester import backtesting
from editorconfig import get_properties, EditorConfigError
import utils.define as define
import utils.utils as utils
import logging
import hashlib
import time
import os

if __name__ == "__main__":
    product_id = 'm'
    total_return, total_fee, precision = 0.0, 0.0, []
    # test_dates = [('20210401', '2021-03-31'),
    #               ('20210402', '2021-04-01')]
    backtesting_config = ''
    try:
        _conf_file = utils.get_path([define.CONF_DIR, define.CONF_FILE_NAME])
        print("conf file in main:", _conf_file)
        options = get_properties(_conf_file)
    except EditorConfigError:
        logging.warning("Error getting EditorConfig propterties", exc_info=True)
    else:
        for key, value in options.items():
            backtesting_config = '{0},{1}:{2}'.format(backtesting_config, key, value)
    _ret_path = utils.get_path([define.RESULT_DIR, define.BT_DIR, 'results_{0}.txt'.format(product_id)])
    f = open(_ret_path, "a")
    result_fname_digest = hashlib.sha256(bytes(backtesting_config, encoding='utf-8')).hexdigest()
    f.write("{0}:{1}\n".format(backtesting_config, result_fname_digest))
    dates = ['20210104', '20210105']
    for idx, trade_date in enumerate(dates):
        if idx < 1:
            continue
        start_ts = time.time()
        ret = backtesting(product_id=product_id, trade_date=trade_date.strip().replace('-', ''),
                          prev_date=dates[idx - 1].strip().replace('-', ''), options=options)
        end_ts = time.time()
        print("back testing time:", end_ts - start_ts)
        if ret:
            _total_return, _total_fee, _precision = ret
            total_return += _total_return
            total_fee += _total_fee
        break
    print(total_return, total_fee, precision)
