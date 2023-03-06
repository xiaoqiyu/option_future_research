#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 14:02
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : Account.py

class Account(object):
    def __init__(self, init_cash=100000):
        self.transaction = list()
        self.fee = 0.0
        self.risk_ratio = [0.0]
        self.occupied_margin = list()
        self.market_value = init_cash
        self.available_margin = list()
        self.trade_market_values = [init_cash]
        self.settle_market_values = [init_cash]

    def add_transaction(self, val=[]):
        self.transaction.append(val)

    def cache_transaction(self):
        pass

    def update_fee(self, val):
        self.fee += val

    def update_risk_ratio(self, val):
        self.risk_ratio.append(val)

    def update_occupied_margin(self, val):
        self.occupied_margin.append(val)

    def update_market_value(self, trade_val, settle_val, fee):
        _trade_val = self.trade_market_values[-1]
        # print("old mkt val", _trade_val)
        self.trade_market_values.append(_trade_val - fee + trade_val)
        # print("new mkt val", self.trade_market_values[-1])
        _settle_val = self.settle_market_values[-1]
        # print("old mkt val 1", _settle_val)
        self.settle_market_values.append(_settle_val - fee + settle_val)
        # print("new mkt val 1", self.settle_market_values[-1])

    def update_available_margin(self, val):
        self.available_margin.append(val)

    def get_trade_market_values(self):
        return self.trade_market_values

    def get_settle_market_values(self):
        return self.settle_market_values
