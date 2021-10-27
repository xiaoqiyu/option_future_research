#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/10/26 10:41
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : reg_demos.py

from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

boston = load_boston()
re_model = RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=0)
ret = cross_val_score(re_model, boston.data, boston.target, cv=10, scoring='neg_mean_squared_error')
print(ret)
