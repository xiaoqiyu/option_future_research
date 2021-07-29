#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/29 15:50
# @Author  : rpyxqi@gmail.com
# @Site    : 
# @File    : utils.py

import time


def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        result = func(*args, **kwargs)
        te = time.time()
        # logger.info('%r (%r, %r) %2.2f sec' % (func.__name__, args, kwargs, te - ts))
        print('%r (%r, %r) %2.2f sec' % (func.__name__, args, kwargs, te - ts))
        return result

    return
