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


def test_time():
    for item in range(10):
        time.sleep(1)


def is_trade(start_timestamp=None, end_timestamp=None, update_time=None):
    if update_time > start_timestamp and update_time < '22:59:30':
        return True
    elif update_time > '09:00:30' and update_time < end_timestamp:
        return True
    else:
        return False
    return False


if __name__ == "__main__":
    start_ts = time.time()
    test_time()
    end_ts = time.time()
    print(end_ts - start_ts)
