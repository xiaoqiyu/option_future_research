#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/26 14:02
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : Account.py

class Account(object):
    def __init__(self):
        self.transaction = list()
        self.fee = 0.0
        self.risk_ratio = list()
        self.occupied_margin = list()
        self.market_value = 0.0
        self.available_margin = list()

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

    def update_market_value(self, val):
        self.market_value += val

    def update_available_margin(self, val):
        self.available_margin.append(val)
