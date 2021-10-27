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
from research.tick_models.ModelProcess import *
import logging
import hashlib
import time
import gc
import os


logging.basicConfig(filename='logs/{0}.txt'.format(os.path.split(__file__)[-1].split('.')[0]), level=logging.DEBUG)
logger = logging.getLogger()


def train_models(start_date: str = '20210701', end_date: str = '20210730', product_id: str = 'rb'):
    """

    :param start_date:
    :param end_date:
    :param product_id:
    :return:
    """
    trade_date_df = utils.get_trade_dates(start_date=start_date, end_date=end_date)
    trade_date_df = trade_date_df[trade_date_df.exchangeCD == 'XSHE']
    trade_dates = list(trade_date_df['calendarDate'])
    train_days = 3
    num_date = len(trade_dates)
    idx = 0
    # [20, 60, 120, 600, 1200]
    predict_window_lst = [20]  # 10s, 30s,1min,5min,10min-- 10s
    lag_window_lst = [120]  # 10s, 30s,1min,5min,
    for predict_win in predict_window_lst:
        for lag_win in lag_window_lst:
            logger.info('train for predict windows:{0} and lag_windows:{1}-----------'.format(predict_win, lag_win))
            while idx < num_date - train_days:
                train_model_reg_intraday(predict_windows=predict_win,
                                         lag_windows=lag_win,
                                         top_k_features=20,
                                         start_date=trade_dates[idx],
                                         end_date=trade_dates[idx + train_days],
                                         train_days=train_days, product_id=product_id.upper())
                idx += 1
                gc.collect()


def backtest(start_date: str = '20210707', end_date: str = '20210709', product_id: str = 'rb'):
    """
    :param start_date:
    :param end_date:
    :param product_id:
    :return:
    """
    trade_date_df = utils.get_trade_dates(start_date=start_date, end_date=end_date)
    trade_date_df = trade_date_df[trade_date_df.exchangeCD == 'XSHE']
    trade_dates = list(trade_date_df['calendarDate'])

    total_return, total_fee, precision = 0.0, 0.0, []
    backtesting_config = ''
    try:
        _conf_file = utils.get_path([define.CONF_DIR, define.CONF_FILE_NAME])
        logger.info("conf file in main:", _conf_file)
        options = get_properties(_conf_file)
    except EditorConfigError:
        logging.warning("Error getting EditorConfig propterties", exc_info=True)
    else:
        for key, value in options.items():
            backtesting_config = '{0},{1}:{2}'.format(backtesting_config, key, value)
    _ret_path = utils.get_path([define.RESULT_DIR, define.BT_DIR, 'results_{0}.txt'.format(product_id)])
    f = open(_ret_path, "a")
    result_fname_digest = hashlib.sha256(bytes(backtesting_config, encoding='utf-8')).hexdigest()
    logger.info('digest', result_fname_digest)
    f.write("{0}:{1}\n".format(backtesting_config, result_fname_digest))
    # dates = ['20210707', '20210708', '20210709']
    for idx, trade_date in enumerate(trade_dates):
        logger.info('start back test for date:{0}'.format(trade_date))
        start_ts = time.time()
        ret = backtesting(product_id=product_id, trade_date=trade_date.strip().replace('-', ''),
                          signal_name='RegSignal', result_fname_digest=result_fname_digest, options=options)
        end_ts = time.time()
        logger.info("back testing time:", end_ts - start_ts)
        if ret:
            _total_return, _total_fee, _precision = ret
            total_return += _total_return
            total_fee += _total_fee
    logger.info(total_return, total_fee, precision)


if __name__ == "__main__":
    train_models(start_date='20210701', end_date='20210730', product_id='rb')
    # backtest(start_date='20210707', end_date='20210715', product_id='hc')
