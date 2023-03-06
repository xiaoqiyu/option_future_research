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
import pandas as pd
import matplotlib.pyplot as plt
import pprint

logging.basicConfig(filename='logs/{0}.txt'.format(os.path.split(__file__)[-1].split('.')[0]), level=logging.DEBUG)
logger = logging.getLogger()


def data_visialize(n_records=100):
    df = pd.read_csv('C:\\projects\\pycharm\\option_future_research\\cache\\factors\\factor_rb2110_20210706.csv')
    print(df.shape)
    # df[['oir', 'label_1']].plot()
    # plt.show()
    # df.hist(bins=80, figsize=(9, 6))
    # df[['oir','oi','cos','label_0']].hist(bins=80, figsize=(9, 6))
    cols = list(df.columns)
    cols.remove('norm_bs_tag')


    # df[cols].hist()

    # sns.pairplot(df_final_features.iloc[:,2:], height=1.5)

    from random import sample
    # df0 = df[df.label_clf_1 == 0]
    # df1 = df[df.label_clf_1 == 1]
    #
    # num0 = df0.shape[0]
    # sample_0 = sample(list(range(num0)), int(num0 * 0.1))
    # df = df1.append(df0.iloc[sample_0])
    cols = df.columns
    for c in cols:
        df.plot.scatter(c, 'label_1')
        plt.show()
    # df.plot.scatter('realized_vol', 'future_vol')
    # df.plot.scatter('aratio', 'bratio')
    # df = df[df.label_clf_1 != 0]
    # df.plot.scatter('oir', 'label_1')
    # plt.xlim(-50,50)
    plt.show()

    # x_idx = list(range(n_records))
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(x_idx, list(df['cos'])[:n_records], '-', label='oir')
    # # ax.plot(time, Rn, '-', label='Rn')
    # ax2 = ax.twinx()
    # ax2.plot(x_idx, list(df['price_chg_1'])[:n_records], '-r', label='label')
    # ax.legend(loc=0)
    # ax.grid()
    # # ax.set_xlabel("Time (h)")
    # # ax.set_ylabel(r"Radiation ($MJ\,m^{-2}\,d^{-1}$)")
    # # ax2.set_ylabel(r"Temperature ($^\circ$C)")
    # ax2.set_ylim(-15, 15)
    # ax.set_ylim(-2, 2)
    # ax2.legend(loc=0)
    # # plt.savefig('0.png')
    # plt.show()


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
    train_days = 4
    num_date = len(trade_dates)
    idx = 0
    predict_window_lst = [10,20]  # 10s, 30s,1min,5min,10min-- 10s
    lag_window_lst = [10, 60]  # 10s, 30s,1min,5min,

    print(
        'train for predict windows:{0} and lag_windows:{1}'.format(predict_window_lst, lag_window_lst))
    lst_score = []
    while idx < num_date - train_days:
        print(trade_dates[idx], trade_dates[idx + train_days])
        # train_model_reg_without_feature_preselect(predict_windows=predict_window_lst,
        #                                           lag_windows=lag_window_lst,
        #                                           top_k_features=20,
        #                                           start_date=trade_dates[idx],
        #                                           end_date=trade_dates[idx + train_days],
        #                                           train_days=train_days, product_id=product_id.upper())

        ret_scores = train_model_reg(predict_windows=predict_window_lst, lag_windows=lag_window_lst,
                                     start_date=trade_dates[idx],
                                     end_date=trade_dates[idx + train_days],
                                     top_k_features=-10, train_days=train_days, product_id=product_id.upper())
        # ret_scores = train_model_clf(predict_windows=predict_window_lst, lag_windows=lag_window_lst,
        #                              start_date=trade_dates[idx],
        #                              end_date=trade_dates[idx + train_days],
        #                              top_k_features=-10, train_days=train_days, product_id=product_id.upper())
        pprint.pprint(ret_scores)
        lst_score.append(ret_scores)
        idx += 1
        gc.collect()
    # df_score = pd.DataFrame(lst_score, columns=list(lst_score[0].keys()))
    # _train_results_file = os.path.join(os.path.abspath(os.pardir), define.BASE_DIR, define.RESULT_DIR,
    #                                    define.TICK_MODEL_DIR,
    #                                    'model_evaluation_{0}.csv'.format(product_id))
    # df_score.to_csv(_train_results_file, index=False)


def backtest(start_date: str = '20210707', end_date: str = '20210709', product_id: str = 'rb',
             strategy_name: str = 'T0Signal'):
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
    multiplier = 10
    margin_rate = 0.1
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
    print('digest', result_fname_digest)
    f.write("{0}:{1}\n".format(backtesting_config, result_fname_digest))
    dates = ['20210708']
    trans_lst = []
    for idx, trade_date in enumerate(trade_dates):
        print('start back test for date:{0}'.format(trade_date))
        start_ts = time.time()
        ret = backtesting(product_id=product_id, trade_date=trade_date.strip().replace('-', ''),
                          signal_name=strategy_name, result_fname_digest=result_fname_digest, options=options)
        end_ts = time.time()
        print("back testing time:", end_ts - start_ts)
        if ret:
            _total_return, _total_fee, _precision, _transaction = ret
            total_return += _total_return
            total_fee += _total_fee
            [item.append(trade_date) for item in _transaction]
            trans_lst.extend(_transaction)

    trans_df = pd.DataFrame(trans_lst,
                            columns=['idx', 'instrument_id', 'direction', 'last_price', 'filled_price', 'filled_vol',
                                     'open_price',
                                     'fee', 'open_ts',
                                     'close_ts', 'holding_time',
                                     'return', 'trade_date'])

    trans_df['open_cost'] = trans_df['filled_vol'] * trans_df['open_price'] * multiplier * margin_rate
    trans_df['return_ratio'] = (trans_df['return'] / trans_df['open_cost']) * 10000
    trans_df['return_cut'] = pd.cut(trans_df['return_ratio'], 5)
    group_cnt = trans_df[trans_df.holding_time > 0].groupby('return_cut')[['idx']].count()
    total_trans_num = len(group_cnt)
    group_ratio = [item / sum(group_cnt['idx']) * 100 for item in group_cnt['idx']]
    # trans_df['return_count'] = list(group_cnt['idx'])
    _idx_lst = list(range(len(group_cnt)))
    xtick_labels = group_cnt.index
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.bar(_idx_lst, group_ratio)
    ax1.set_xticks(_idx_lst[::1])
    plt.xticks(rotation=75)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.3)
    min_lst = []
    ax1.set_xticklabels(xtick_labels[::1])
    _bt_evaluation_path = utils.get_path(
        [define.RESULT_DIR, define.BT_DIR, '{0}_{1}_{2}.jpg'.format(product_id, start_date, end_date)])
    plt.savefig(_bt_evaluation_path)
    _trans_path = utils.get_path(
        [define.RESULT_DIR, define.BT_DIR, 'trans_{0}_{1}_{2}.csv'.format(product_id, start_date, end_date)])
    trans_df.to_csv(_trans_path, index=False)
    print(total_return, total_fee, precision)


if __name__ == "__main__":
    # data_visialize()
    start_date = '20210701'
    bt_start_date = '20210707'
    end_date = '20210709'
    product_id = 'm'
    # train_models(start_date=start_date, end_date=end_date, product_id=product_id)
    backtest(start_date=bt_start_date, end_date=end_date, product_id=product_id, strategy_name='ClfSignal')
    # data_visialize()
