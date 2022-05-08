#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 08:30:12 2021

@author: q

GOAL : Test TA-lib

Try 1:
a) Move talib.tar to $HOME : mv talib.tar $HOME
b) Extract .tar : sudo tar -xvf ta-lib-0.4.0-src.tar.gz
c) cd ta-lib
d) ./configure --prefix=/home/q/anaconda3/lib/python3.7/site-packages/
   *** Use print(module) to get the address where the libraries are installed 
e) make
f) sudo make install
Result : Not working
    

Try 2:
conda install -c conda-forge ta-lib
Result : Working !

"""

# import
import talib as ta
import yfinance as yf

# dowload data
aapl = yf.download('AAPL', '2019-1-1','2019-12-27')

# create indicators
aapl['Simple MA'] = ta.SMA(aapl['Close'],14)
aapl['EMA'] = ta.EMA(aapl['Close'], timeperiod = 14)

# check df tail
print(aapl.tail())

# plot 
aapl[['Adj Close', 'EMA']].plot()
