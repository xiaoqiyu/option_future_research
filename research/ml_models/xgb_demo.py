#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/25 22:14
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : xgb_demo.py

import xgboost as xgb

# read in data
dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# specify parameters via map
param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)
