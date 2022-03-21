#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/15 13:33
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : com_analysis.py

import matplotlib.pyplot as plt
import uqer
import pprint
import numpy as np
from uqer import DataAPI

uqer_client = uqer.Client(token="6aa0df8d4eec296e0c25fac407b332449112aad6c717b1ada315560e9aa0a311")


# 返回相应品种主力合约阶段的日频率 价格波动，收盘价涨跌幅波动，结算价涨跌幅波动，分钟收盘价波动均值
def get_daily_vol(start_date='', end_date='', product_id='RB'):
    df = DataAPI.MktFutdGet(secID=u"", ticker=u"", tradeDate=u"", beginDate=start_date, endDate=end_date, exchangeCD="",
                            field=u"", pandas="1")
    df = df[df.mainCon == 1]
    #     df_m = df[df.ticker == 'M2205']
    df_rb = df[df.contractObject == product_id]
    df_hc = df[df.contractObject == 'HC']
    #     _flag = [item.startswith('m') for item in df['ticker']]
    #     df_m = df[_flag]
    #     print(df_hc.head().T)
    #     print(df_hc['CHG1'].mean(), df_hc['CHG1'].std())
    #     print(df_hc['closePrice'].mean(), df_hc['closePrice'].std())
    #     plt.plot(df_rb['CHG'])
    #     plt.show()
    #     plt.plot(df_rb['closePrice'])
    # #     df_hc['ClosePrice']
    #     plt.show()
    diff = df_rb['closePrice'] - df_hc['closePrice']
    diff = [item - list(df_hc['closePrice'])[idx] for idx, item in enumerate(df_rb['closePrice'])]
    plt.plot(diff)
    plt.show()
    min_std = []
    for d in df_rb['tradeDate']:
        df_min = DataAPI.FutBarHistOneDay2Get(instrumentID=u"rb2205", date=d, unit=u"1", field=u"", pandas="1")
        #         print(d, df_min['closePrice'].std())
        min_std.append(df_min['closePrice'].std())
    return df_rb['closePrice'].std(), df_rb['CHG'].std(), df_rb['CHG1'].std(), sum(min_std) / len(min_std)


if __name__ =="__main__":
    daily_close_std, daily_chg_std, daily_chg1_std, min_std = get_daily_vol("20220208", "20220314", 'RB')
    print('daily close std:',daily_close_std,'daily close chg std:', daily_chg_std,'daily clear chg std:', daily_chg1_std, 'min_std:', min_std)
