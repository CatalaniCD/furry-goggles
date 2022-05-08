#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 22:59:07 2021

@author: q

GOAL : Develop a Technical Analysis Tool for Mr. Liu

Sources : a) https://www.investopedia.com/top-7-technical-analysis-tools-4773275        
         b) https://school.stockcharts.com/doku.php?
         c) http://mrjbq7.github.io/ta-lib/doc_index.html

MACD, Moving Average Convergence Divergence 
STOCHRSI, Stochastic Relative Strength Index
    
"""

# =============================================================================
# imports
# =============================================================================

# data handling
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt

# technical indicators
import talib as ta

# market data
import yfinance as yf

# =============================================================================
# technical indicators
# =============================================================================


def MACDsignal(df, fast = 12, slow = 26, signal = 9):
    
    """
    :input: ohlcv DataFrame, pd.DataFrame
            period, int
    
    :return: MACD value,
             MACD hist value,
             MACD value label
             MACD hist label
    
    The moving average convergence divergence (MACD) indicator helps 
    traders see the trend direction, as well as the momentum of that trend. 
    It also provide a number of trade signals.

    When the MACD is above zero, the price is in an upward phase. 
    If the MACD is below zero, it has entered a bearish period.
    
    The indicator is composed of two lines: the MACD line and a signal line, 
    which moves slower. When MACD crosses below the signal line, 
    it indicates that the price is falling. When the MACD line crosses 
    above the signal line, the price is rising. 
    
    Looking at which side of zero the indicator is on aids in determining 
    which signals to follow. For example, if the indicator is above zero, 
    watch for the MACD to cross above the signal line to buy. 
    If the MACD is below zero, the MACD crossing below the signal line may 
    provide the signal for a possible short trade. 
    
    """
    
    
    # set kwargs
    kwargs = {'fastperiod'    : fast,
              'slowperiod'    : slow,
              'signalperiod'  : signal, 
              }
    
    df['MACD'], _ , df['MACDhist'] = ta.MACD(df['Close'], **kwargs)
    # calc close mean
    n = 3
    price = df['Close'][-(slow+n):]
    # select valleys
    valleys = price[(price.diff() > 0).astype(int).diff() == -1]
    # calculate uptrend
    price_lr = np.polyfit(range(len(valleys.index)), valleys.values, 1)            
    # macd values to evaluate
    macd_value = df['MACD'].values[-1]
    macd_hist = df['MACDhist'].values[-1]
    # check valleys going higher
    if price_lr[0] > 0:
        
        if macd_value > 0:    
            value_label = 1 # buy
        else:
            value_label = 0 # not Buy
        
        if macd_hist > 0:
            signal_label = 1 # buy
        else:
            signal_label = 0 # not buy
        
    else:
        value_label = 0 # sell
        signal_label = 0 # sell
    
    label_0 = f'MACD({fast}, {slow}, {signal}) value'
    label_1 = f'MACD({fast}, {slow}, {signal}) hist'
    label_2 = f'MACD({fast}, {slow}, {signal}) value signal'
    label_3 = f'MACD({fast}, {slow}, {signal}) hist signal'
    return {label_0 : macd_value, label_1 : macd_hist, label_2 : value_label, label_3 : signal_label}

     
def STOCHRSIsignal(df, rsi_period = 26, stoch_period = (5, 3), levels = (30, 70)):

    """
    :input: ohlcv DataFrame, pd.DataFrame
            period, int
    
    :return: STOCHRSI value,
             STOCHRSI uptrend,
             STOCHRSI downtrend,
             
    StochRSI is an indicator of an indicator, which makes it the 
    second derivative of price. This means it is two steps (formulas) 
    removed from the price of the underlying security. Price has 
    undergone two changes to become StochRSI. Converting prices to RSI 
    is one change. Converting RSI to the Stochastic Oscillator is 
    the second change. This is why the end product (StochRSI) looks 
    much different than the original (price). 
    
    """
    # set kwargs
    kwargs = { 'timeperiod'   : rsi_period, 
               'fastk_period' : stoch_period[0], 
               'fastd_period' : stoch_period[1], 
               'fastd_matype' : 1,
              }
    
    df['FASTk'], df['FASTd'] = ta.STOCHRSI(df['Close'], **kwargs)
    stoch_rsi_value = df['FASTd'].values[-1]
    up_trend = (stoch_rsi_value > levels[1]).astype(int)
    down_trend = (stoch_rsi_value < levels[0]).astype(int) 
    
    return {'STOCHRSIk' : stoch_rsi_value, 'STOCHRSI OverBougth' : up_trend, 'STOCHRSI OverSold' : down_trend,}
    

if __name__ == '__main__':
    
    # dowload data
    # df = yf.download('AMZN', '2019-1-1','2019-12-27', threads = True )    
    
    df = pd.DataFrame()
    for i in range(5):
        df[f'random_stock_{i}'] = pd.Series([np.random.normal(loc = 0.0, scale = 1.0, size = None,) for x in range(1000)], name = 'Close').cumsum()
    
    df.plot(grid = True)
        
    table = pd.DataFrame()
    for signal in [STOCHRSIsignal, MACDsignal]:
        for stock in [x for x in df.columns if 'stock' in x]: 
            df['Close'] = df[stock]
            indicator = signal(df)
            for sig in indicator.keys():
                table.loc[sig, stock] = str(indicator[sig])[:4]
    
    print(table)