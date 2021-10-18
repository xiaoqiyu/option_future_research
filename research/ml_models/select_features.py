#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/14 10:05
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : select_features.py

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA


def iris_feature_test():
    iris = load_iris()
    # mutual_info_classif, chi2
    model1 = SelectKBest(mutual_info_classif, k=3)
    ret = model1.fit_transform(iris.data, iris.target)

    print(model1.scores_)
    print(model1.pvalues_)
    # chi2test1(iris.target, iris.data[:,0])
    print(ret.shape)


def chi2test1(y, x):
    con_table = pd.crosstab(y, x)
    chi2, p, df, ex = chi2_contingency(con_table)
    print(chi2, p, df)


def pca_demo():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=1, svd_solver='arpack')
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    print(pca.transform([[2, 3]]))


if __name__ == '__main__':
    # iris_feature_test()
    # iris = load_iris()
    # (iris.target, iris.data[:, 0])
    pca_demo()
