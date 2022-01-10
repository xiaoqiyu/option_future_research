from datetime import datetime
from collections import defaultdict, deque
import os
import pickle as pk

from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from tqdm import tqdm

import core
import core.theo as T
import core.fitting as F


def get_daily_info(startdate, enddate):
    cache_file = os.path.join(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3]), 'data', 'stock_daily_info.csv')
    daily_info = pd.read_csv(cache_file, dtype={'date': str, 'symbol': str})
    daily_info['refPrice'] = round((daily_info['price_limit_up'] + daily_info['price_limit_down']) / 2 * 100 - 0.1)/ 100
    daily_info = daily_info.set_index(['symbol', 'date'])
    return daily_info


def cross_edge_notional_multilevel(book_with_valuation, symbol, posNotionals, edges, daily_info, use_ratio=True, start_extime=93200000000):
    results = defaultdict(lambda: defaultdict(dict))
    for date, sub_df in sorted(book_with_valuation.groupby('date')):
        if (symbol, date) in daily_info.index:
            refPrice = daily_info.loc[(symbol, date), 'refPrice']
        else:
            print('%s: %s not in daily info. skip.' % (date, symbol))
            continue

        for posNotional in posNotionals:
            position_normal = int(posNotional / refPrice // 100 * 100)
            # print(date, '%d / %.2f = %d' % (posNotional, refPrice, position_normal))
            if position_normal == 0:
                # print('posNormal = 0. Skip')
                continue

            if sub_df.shape[0] - sub_df.dropna().shape[0] > 50 or sub_df.dropna().shape[0] < 50:
                pass
            else:
                print(date, sub_df.dropna().shape[0], sub_df.shape[0])
                book_events = sub_df.to_dict('records')
                for edge in edges:
                    trades = []
                    buyprice = defaultdict(int)
                    sellprice = defaultdict(int)
                    cur_position = 0
                    for event in book_events:
                        if event['ExchangeTime'] < start_extime:
                            continue
                        if not np.isnan(event['value']) and event['Bid1'] != 0 and event['Ask1'] != 0:
                            midpt = (event['Bid1'] + event['Ask1']) / 2
                            fv = midpt + event['value']
                            for level in range(5):
                                i = level + 1
                                if (not use_ratio and (fv - event['Ask%d' % i]) > edge) or (use_ratio and (fv - event['Ask%d' % i]) / midpt > edge):
                                    new_buy_qty = min(max(0, event['Ask%dSize' % i] - buyprice[event['Ask%d' % i]]), position_normal - cur_position)
                                    if new_buy_qty > 0:
                                        trades.append({'dir': False, 'price': event['Ask%d' % i], 'qty': new_buy_qty, 'extime': event['ExchangeTime'], 'date': event['date'], 'TimeStamp': event['TimeStamp']})
                                        buyprice[event['Ask%d' % i]] += new_buy_qty
                                        cur_position += new_buy_qty

                                if (not use_ratio and (event['Bid%d' % i] - fv)) > edge or (use_ratio and (event['Bid%d' % i] - fv) / midpt > edge):
                                    new_sell_qty = min(max(0, event['Bid%dSize' % i] - sellprice[event['Bid%d' % i]]), cur_position + position_normal)
                                    if new_sell_qty:
                                        trades.append({'dir': True, 'price': event['Bid%d' % i], 'qty': new_sell_qty, 'extime': event['ExchangeTime'], 'date': event['date'], 'TimeStamp': event['TimeStamp']})
                                        sellprice[event['Bid%d' % i]] += new_sell_qty
                                        cur_position -= new_sell_qty

                            buyprice_new = defaultdict(int)
                            sellprice_new = defaultdict(int)
                            buyprice_new.update({k: v for k, v in buyprice.items() if k > event['Bid1']})
                            sellprice_new.update({k: v for k, v in sellprice.items() if k < event['Ask1']})
                            buyprice = buyprice_new
                            sellprice = sellprice_new
                    results[posNotional][edge][date] = trades

    return results



def constraint_trades_time(trades, start_extime, end_extime):
    trades = trades.copy()
    for date in trades:
        tmp = trades[date]
        trades[date] = [v for v in tmp if v['extime'] >= start_extime and v['extime'] < end_extime]

    return trades


def constraint_trades_maxvolume(trades_date, maxvolume):
    pos_total = 0
    neg_total = 0
    new_trades = []
    for t in trades_date:
        if t['dir'] and t['qty'] and neg_total < maxvolume:
            new_trades.append(t.copy())
            new_trades[-1]['qty'] = min(maxvolume - neg_total, new_trades[-1]['qty'])
            neg_total += new_trades[-1]['qty']
        if not t['dir'] and t['qty'] and pos_total < maxvolume:
            new_trades.append(t.copy())
            new_trades[-1]['qty'] = min(maxvolume - pos_total, new_trades[-1]['qty'])
            pos_total += new_trades[-1]['qty']

    return new_trades


def handle_match(fills, t):
    matches = []
    t = t.copy()
    t['remain_qty'] = t['qty']
    if not fills or (fills[0]['dir'] and t['dir']) or (not fills[0]['dir'] and not t['dir']):
        fills.append(t)
    else:
        while fills:
            fill = fills[0]
            match_qty = min(fill['remain_qty'], t['remain_qty'])
            fill['remain_qty'] -= match_qty
            t['remain_qty'] -= match_qty
            if fill['remain_qty'] == 0:
                fills.popleft()
            holdtime = t['TimeStamp'] - fill['TimeStamp']
            # print(fill)
            # print(t)
            matches.append({'qty': match_qty,
                            'entry_dir': fill['dir'],
                            'entryprice': fill['price'],
                            'exitprice': t['price'],
                            'entrytime': datetime.fromtimestamp(fill['TimeStamp'] / 1e6),
                            'exittime': datetime.fromtimestamp(t['TimeStamp'] / 1e6)})

            if t['remain_qty'] == 0:
                break

        if t['remain_qty'] > 0:
            fills.append(t)

    return matches


def match(trades_date):
    matches = []
    fills = deque()
    for t in trades_date:
        pos = t['qty'] * (1.0 if not t['dir'] else -1.0)
        if abs(pos) != 0:
            tmp_matches = handle_match(fills, t)
            matches.extend(tmp_matches)

    return matches, fills


def get_open_pnl(lastbook_date, remain_fills):
    open_pnl = 0
    for f in remain_fills:
        open_pnl += (1.0 if f['dir'] else -1.0) * f['remain_qty'] * (f['price'] - lastbook_date['midpt'])
    return open_pnl


def get_open_notional(remain_fills):
    open_notional = 0
    for f in remain_fills:
        open_notional += f['remain_qty'] * f['price'] * (1.0 if not f['dir'] else -1.0)
    return open_notional


def get_remain_qty(remain_fills):
    remain_qty = 0
    for f in remain_fills:
        remain_qty += f['remain_qty'] * (1.0 if not f['dir'] else -1.0)
    return remain_qty


def get_lastbook(book_with_valuation, end_extime):
    book_with_valuation = book_with_valuation[book_with_valuation['ExchangeTime'] >= end_extime]
    lastbook = {}
    for date, sub_df in book_with_valuation.groupby('date'):
        sub_df = sub_df.sort_values('ExchangeTime')
        lastbook[date] = sub_df.iloc[0]
    return lastbook


def end_penalty_half_spread(last_book_date, remain_qty):
    if last_book_date['Ask1'] == 0 or last_book_date['Bid1'] == 0:
        return 0
    else:
        return abs(remain_qty) * 1 / 2 * (last_book_date['Ask1'] - last_book_date['Bid1'])


def end_penalty_whole_book(last_book_date, remain_qty):
    if last_book_date['Ask1'] == 0 or last_book_date['Bid1'] == 0:
        return 0
    else:
        if remain_qty > 0:
            total_qty = 0
            total_notional = 0
            first_bid = last_book_date['Bid1']
            last_bid = 0
            remain_qty = abs(remain_qty)
            for level in range(1, 6):
                if last_book_date['Bid%d' % level] != 0:
                    bid = last_book_date['Bid%d' % level]
                    bidsize = last_book_date['Bid%dSize' % level]
                    matchqty = min(bidsize, remain_qty - total_qty)
                    total_qty += matchqty
                    total_notional += matchqty * bid
                    if total_qty == remain_qty:
                        break
                    last_bid = bid
                else:
                    break

            if remain_qty > total_qty:
                total_notional += (remain_qty - total_qty) * last_bid
                total_qty = remain_qty

            return last_book_date['midpt'] * remain_qty - total_notional

        if remain_qty < 0:
            total_qty = 0
            total_notional = 0
            first_ask = last_book_date['Ask1']
            last_ask = 0
            remain_qty = abs(remain_qty)
            for level in range(1, 6):
                if last_book_date['Ask%d' % level] != 0:
                    ask = last_book_date['Ask%d' % level]
                    asksize = last_book_date['Ask%dSize' % level]
                    matchqty = min(asksize, remain_qty - total_qty)
                    total_qty += matchqty
                    total_notional += matchqty * ask
                    if total_qty == remain_qty:
                        break
                    last_ask = ask
                else:
                    break

            if remain_qty > total_qty:
                total_notional += (remain_qty - total_qty) * last_ask
                total_qty = remain_qty

            return total_notional - last_book_date['midpt'] * remain_qty


def get_holdtime(matches_date):
    total_qty = 0
    total_holdtime = 0
    for m in matches_date:
        total_qty += m['qty']
        holdtime = (m['exittime'] - m['entrytime']).total_seconds()
        if (m['entrytime'].hour + m['entrytime'].minute / 60) <= 11.7 and (m['exittime'].hour + m['exittime'].minute / 60) >= 12.8:
            holdtime = max(0, holdtime - 5400)
        total_holdtime += m['qty'] * holdtime

    return total_holdtime / total_qty if total_qty > 0 else 0.0, total_qty


def get_holdtime2(matches_date, remain_fills, lastbook_date):
    total_holdtime, total_qty = get_holdtime(matches_date)
    total_qty2 = 0
    total_holdtime2 = 0

    exit_time = datetime.fromtimestamp(lastbook_date['TimeStamp'] / 1e6)
    for f in remain_fills:
        total_qty2 += f['remain_qty']
        entry_time = datetime.fromtimestamp(f['TimeStamp'] / 1e6)
        holdtime = (exit_time - entry_time).total_seconds()
        if (entry_time.hour + entry_time.minute / 60) <= 11.7 and (exit_time.hour + exit_time.minute / 60) >= 12.8:
            holdtime = max(0, holdtime - 5400)

        total_holdtime2 += f['remain_qty'] * holdtime

    combined_qty = total_qty + total_qty2
    combined_holdtime = total_holdtime * total_qty + total_holdtime2
    return combined_holdtime / combined_qty if combined_qty > 0 else 0.0, combined_qty


def get_summary(daily_info, symbol, trades, book_with_valuation, maxNotional, start_extime, end_extime, fee_ratio=0.0014, startdate=None, enddate=None):
    summary = defaultdict(lambda: defaultdict(dict))
    lastbook = get_lastbook(book_with_valuation, end_extime)
    for posnotional in trades:
        trades_posnotional = trades[posnotional]
        for edge_ratio in trades_posnotional:
            trades_edge = trades_posnotional[edge_ratio]
            trades_time = constraint_trades_time(trades_edge, start_extime, end_extime)
            for date in trades_time:
                if (startdate is not None and startdate > date) or (enddate is not None and enddate < date):
                    continue
                try:
                    lastbook_date = lastbook[date]
                except Exception as e:
                    print(symbol, 'lastbook error', e)
                    continue
                maxvolume = maxNotional / daily_info.loc[(symbol, date), 'refPrice'] // 100 * 100
                trades_date = trades_time[date]
                trades_date_maxvolume = constraint_trades_maxvolume(trades_date, maxvolume)
                matches_date, remaining_fills = match(trades_date_maxvolume)
                holdtime, matchqty = get_holdtime(matches_date)
                holdtime2, matchqty2 = get_holdtime2(matches_date, remaining_fills, lastbook_date)
                try:
                    if len(matches_date):
                        df = pd.DataFrame(matches_date)
                        # df['entryTimeBin'] = df['entrytime'].apply(lambda x: x.hour + x.minute // 30 * 0.5)
                        df['pnl'] = df['qty'] * (df['entryprice'] - df['exitprice']) * df['entry_dir'].apply(lambda x: 1.0 if x else -1.0)
                        df['notional'] = df['qty'] * (df['entryprice'] + df['exitprice']) / 2
                        df['fee'] = df['notional'] * fee_ratio
                        sum_ = df[['fee', 'pnl', 'notional']].sum()
                    else:
                        sum_ = {
                            'fee': 0,
                            'pnl': 0,
                            'notional': 0,
                        }
                    open_pnl = get_open_pnl(lastbook_date, remaining_fills)
                    open_notional = get_open_notional(remaining_fills)
                    remaining_qty = get_remain_qty(remaining_fills)
                    date_perf = {
                        'closed_pnl': sum_['pnl'],
                        'closed_fee': sum_['fee'],
                        'maxNotional': maxNotional,
                        'closed_notional': sum_['notional'],
                        'open_qty': remaining_qty,
                        'open_pnl': open_pnl,
                        'open_notional': open_notional,
                        'open_fee': abs(open_notional) * fee_ratio,
                        'holdtime': holdtime,
                        'holdtime2': holdtime2,
                        'open_penalty_half_spread': end_penalty_half_spread(lastbook_date, remaining_qty),
                        'open_penalty_whole_book': end_penalty_whole_book(lastbook_date, remaining_qty),
                    }
                    summary[posnotional][edge_ratio][date] = date_perf
                except Exception as e:
                    print('except', e)
                    import pdb; pdb.set_trace()
    return summary


def gen_valuation(symbols: list, make_theolist, weights, date_first: str, date_last: str) -> pd.DataFrame:
    """
    Returns:
        DataFrame: book and 'value' column
    """
    quote_src = core.quote.load_data(symbols, date_first=date_first, date_last=date_last)
    quotes = [quote_src.get_quote(symbol) for symbol in symbols]
    quotes[0].alias = 'self'
    markup = core.markup.gen_markup(quotes[0], alpha=0.5, lag=3) 

    theofunc_list = [T.MakeTimeTheo(quotes[0])] + make_theolist(quotes)
    
    theomatrix = T.make_theomatrix(theofunc_list, quotes[0].sampler, quote_src.get_data_num())
    
    # Sampling
    sampling_param = {'sample_type': 'none'}
    sampler = F.FitSampling(sampling_param)
    
    contract = quote_src.dict_sym2contract[symbols[0]]
    book_with_val = []
    
    for idx in range(len(quote_src.dateStr)):
        day = quote_src.dateStr[idx]
        raw = quote_src.dfdict[contract][idx]
        thmatrix_samp, markup_samp = sampler(quotes[0], theomatrix, markup, 0, selectedDays=[idx])
        valuation = np.dot(weights, thmatrix_samp[1:,])
        df_val = pd.DataFrame({'TimeStamp': thmatrix_samp[0,:], 'value': valuation})
        book_with_val.append(raw.merge(df_val, on='TimeStamp'))
        
    
    return pd.concat(book_with_val)


class BasicCrossSimulationParam:
    def __init__(self, date_first, date_last, position_notionals, edge_ratios):
        self.position_notionals = position_notionals
        self.edge_ratios = edge_ratios
        self.date_first = date_first
        self.date_last = date_last


def run_basic_cross_strategy(book_with_valuation, symbol: str, param: BasicCrossSimulationParam, daily_info):
    book_with_valuation['midpt'] = (book_with_valuation['Bid1'] * (1 + (book_with_valuation['Ask1Size'] == 0)) + book_with_valuation['Ask1'] * (1 + (book_with_valuation['Bid1Size'] == 0))) / 2
    book_with_valuation['date'] = book_with_valuation['TimeStamp'].apply(lambda x: datetime.fromtimestamp(x / 1e6).strftime('%Y%m%d'))
    resultsMultiLevel = cross_edge_notional_multilevel(
        book_with_valuation, symbol, param.position_notionals, param.edge_ratios, daily_info)
    print(resultsMultiLevel)

    max_notional = 2000000
    start_extime = 93200000000
    end_extime = 145000000000
    summary = get_summary(daily_info, symbol, resultsMultiLevel, book_with_valuation, max_notional, start_extime, end_extime)
    return summary


if __name__ == '__main__':
    # book_parquet_dir = '/home/mhyang/Data/Experiment/DynamicEdgeIntraDay/book_with_valuation_GBDT_SH'
    # trade_dir = '/home/mhyang/Data/Experiment/PythonSim/GBDT_SH_Trades'
    # output_dir = 'GBDT_SH_Summary_0932_1450'

    book_parquet_dir = 'example/valuation/'
    trade_dir = 'example/trades'
    output_dir = 'example/summary'

    symbols = sorted([_.split('.p')[0] for _ in os.listdir(trade_dir)])
    symbols = ['600519']
    os.makedirs(output_dir, exist_ok=True)
    max_notional = 2000000
    start_extime = 93200000000
    end_extime = 145000000000

    daily_info = get_daily_info('20210801', '20210831')

    def default_to_regular(d):
        if isinstance(d, defaultdict):
            d = {k: default_to_regular(v) for k, v in d.items()}
        return d

    def run(symbol):
        print(symbol)
        output_path = os.path.join(output_dir, '%s.p' % symbol)
        if not os.path.exists(output_path):
            try:
                book_fpath = os.path.join(book_parquet_dir, '%s.parquet.gzip' % symbol)
                book_with_valuation = pd.read_parquet(book_fpath)
                book_with_valuation['date'] = book_with_valuation['TimeStamp'].apply(lambda x: datetime.fromtimestamp(x / 1e6).strftime('%Y%m%d'))
                book_with_valuation['midpt'] = (book_with_valuation['Bid1'] * (1 + (book_with_valuation['Ask1Size'] == 0)) + book_with_valuation['Ask1'] * (1 + (book_with_valuation['Bid1Size'] == 0))) / 2
                trade_fpath = os.path.join(trade_dir, '%s.p' % symbol)
                trades = pk.load(open(trade_fpath, 'rb'))
                summary = get_summary(daily_info, symbol, trades, book_with_valuation, max_notional, start_extime, end_extime)
                pk.dump(default_to_regular(summary), open(output_path, 'wb'))
            except Exception as e:
                print(symbol, e)
        else:
            print('%s exists' % output_path)

    import multiprocessing

    n = 8
    pool = multiprocessing.Pool(n)
    pool.map(run, symbols)
