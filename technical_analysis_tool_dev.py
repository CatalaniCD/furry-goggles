#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 22:59:07 2021

@author: q

GOAL : Develop a Technical Analysis Tool for Mr. Liu

Source : a) https://www.investopedia.com/top-7-technical-analysis-tools-4773275        
         b) https://school.stockcharts.com/doku.php?
         c) http://mrjbq7.github.io/ta-lib/doc_index.html

/// 7 Technical Indicators
1. OBV, On Balance Volume @
2. AC, Accumulation/Distribution Line @
3. ADX, Average Directional Index @
4. Aroon, Aroon Indicator @
5. MACD, Moving Average Convergence Divergence @
6. RSI, Relative Strength Index @
7. STCH, Stochastic Oscillator @

/// Installation

pip install TA-lib
pip install pykalman

/// Technical Indicators

Price Based Indicators :
    - KF, Kalman Filter 
    - EMA, Exponential Moving Average
    - DEMA, Double Exponential Moving Average
    - TEMA, Triple Exponential Moving Average
    - TRIX, TEMA ROC
    - ROC, Rate of Change
    - MOM, Momentum
    - MACD, Moving Average Convergence Divergence
    - STC, Stochastic Oscillator
    - STC KF, Stochastic Oscillator Kalman Filter
    - STCH RSI, Stochastic RSI 
    - ADX, Average Directional Index 
    - BB, Bollinger Bands
        - STD, Standard Deviation
    - KC, Keltner Channels
        - ATR, Average True Range
    - LR, Linear Regression 
    - SR, Support/Resistence 
    - PT, Peak/Through on EMA
    
Volume Based Indicators :
    - MFI, Money Flow Index @
    - OBV, On Balance Volume @
    - AC, Accumulaion and Distribution Line @
    - VWAP, Volume Weighted Average Price @
    - KJI, Klinger Volume Oscillator 
    
    
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
# import yfinance as yf
import ccxt
import datetime as dt

# =============================================================================
# program test
# =============================================================================

class Exchange():
    
    """
    This class provides the access to API; 
    The constructor enables the user to request to the API; 
    """
    
    def __init__(self):
        self.api = 'Evev8LfMMYYkO0yBrrlloj6NjywEbvkJziTok85ZErLYWyjBRqV5NVYJAPViB5B9'
        self.sec = 'tkv4vlCOrSrltn8Lc43RC7xpwyekzfkX1SH4sn5bNf1i6UMrPABi4kz0CRs6tHt6'
        self.exchange = ccxt.binance({'apiKey' : self.api, 'secret' : self.sec})
        pass

    def get_symbols(self, quote = '/USDT'):
        markets = self.exchange.load_markets()
        return [x[0] for x in markets.items() if (x[-1]['active'] == True) and ('SPOT' in x[-1]['info']['permissions']) and (x[0].endswith(quote)) ]



class Utils():
    
    """
    This class stores helper functions to hinherit by various classes
    """
    
    
    def process_pct(self, elem, array):
       n = array.index(elem)
       L = len(array)
       pct = self.calc_pct(n, L)
       print(f'Process : {pct} % ', end = '\r')
       pass
   
    def calc_pct(self, n, L):
        return round( ( n / (L - 1) ) * 100, 2)
    

    def from_timestamp_to_datetime_UTC(self, timestamp):
        tz = dt.timezone(dt.timedelta(hours = 0))
        # t = dt.datetime.now().astimezone(tz)
        return dt.datetime.fromtimestamp(timestamp/1000).astimezone(tz)

    def from_ohlc_to_dataframe(self, ohlc, symbol):
        """
        Input : list in Binance Formar [timestamp, open, high, low, close, volume]
        Returns : 
            
            pd.DataFrame()
            - index.name = self.symbol
            - index : datetime_UTC_series 
            - columns : (open, high, low, close, volume)
        """
        index        = pd.Series([x[0] for x in ohlc], name = symbol).apply(self.from_timestamp_to_datetime_UTC)
        open_price   = pd.Series([float(x[1]) for x in ohlc], name = 'Open')
        high_price   = pd.Series([float(x[2]) for x in ohlc], name = 'High')
        low_price    = pd.Series([float(x[3]) for x in ohlc], name = 'Low')
        close_price  = pd.Series([float(x[4]) for x in ohlc], name = 'Close')
        volume       = pd.Series([float(x[5]) for x in ohlc], name = 'Volume')
        return pd.concat([index, open_price, high_price, low_price, close_price, volume], axis = 1).set_index(symbol)       

    def from_ohlc_to_series(self, ohlc):
        index        = pd.Series([x[0] for x in ohlc], name = self.symbol).apply(self.from_timestamp_to_datetime_UTC)
        close_price  = pd.Series([float(x[4]) for x in ohlc], name = 'Close')
        return pd.Series(close_price, index = index)
    


if __name__ == '__main__':
    
    # dowload data
    # df = yf.download('AAPL', '2019-1-1','2019-12-27') 
    # df = pd.Series([np.random.normal(loc = 0.0, scale = 1.0, size = None,) for x in range(1000)], name = 'Close').cumsum().to_frame()
    _ = Exchange()
    symbol = 'BTC/USDT'
    timeframe = '1h'
    utils = Utils()
    # iterate market symbols           
    marketdata = dict()    
    kwargs = {'symbol' : symbol,
              'timeframe' : timeframe,
              'limit' : 1000,
              }
    
    df = utils.from_ohlc_to_dataframe( _.exchange.fetchOHLCV(**kwargs), symbol)
    # df['Close'].plot()

    OBV = True # Ready
    AD = False  # Ready
    ADX = False # Ready
    AROON = False # Ready
    MACD = False  # Ready
    RSI = False   # Ready
    STOCH = False # Ready
    STOCHRSI = False # Ready
    TRIX = False   # Ready
    BB   = False   # Ready
    MFI  = False   # Ready
    VWAP = False   # Ready
    KC   = False   # Ready
    DC   = False   # Ready
    MOM  = False   # Ready
    CCI  = False   # Ready
    SAR  = False   # Ready
    BOP  = False   # Ready
    WIR  = False   # Ready
    CHK  = False   # Ready

    """ Indicators Ready 20/20 """
    
    
    if OBV:
        
        def OBVcd(df, cd_period = 20):
            
            """
            :input: ohlcv DataFrame, pd.DataFrame
                    cd_period, int
            
            :return: obv last value, 
                     Buy/Sell label
            
            On-balance volume indicator (OBV) to measure the positive and 
            negative flow of volume in a security over time.
            
            When OBV is rising, it shows that buyers are willing to step 
            in and push the price higher. When OBV is falling, the selling 
            volume is outpacing buying volume, which indicates lower prices. 
            In this way, it acts like a trend confirmation tool. 
            If price and OBV are rising, that helps indicate a continuation 
            of the trend. 
            
            Traders who use OBV also watch for divergence. 
            This occurs when the indicator and price are going in different 
            directions. If the price is rising but OBV is falling, 
            that could indicate that the trend is not backed by strong 
            buyers and could soon reverse. 
            
            """
            
            # compute indicator
            obv = ta.OBV(df['Close'], df['Volume']).rolling(3).mean()
            L = 3 # number of maximum values to consider
            # max values covergence / divergence check
            price = df['Close'].rolling(3).mean()
            price_cd, obv_cd = price[(price.diff() > 0).astype(int).diff() == 1][-(cd_period+L):], obv[(obv.diff() > 0).astype(int).diff() == 1][-(cd_period+L):]
            price_model = np.polyfit(range(len(price_cd.index)), price_cd.values, 1)
            obv_model = np.polyfit(range(len(obv_cd.index)), obv_cd.values, 1)   
            # check convergence / divergence
            label = ''
            if price_model[0] > 0 and obv_model[0] > 0:
                label = 'Up Convergence'
            if price_model[0] < 0 and obv_model[0] > 0:
                label = 'Up Divergence'
            elif price_model[0] < 0 and obv_model[0] < 0:
                label = 'Down Convergence'
            elif price_model[0] > 0 and obv_model[0] < 0:
                label = 'Down Divergence'
            
            signal = {'Down Divergence' : 1, 'Up Convergence' : 1,
                      'Down Convergence' : 0, 'Up Divergence' : 0}
    
            return { 'OBV' : list(obv.values).pop(), 'OBVcd' : signal[label] }
    
        df['OBV'] = ta.OBV(df['Close'], df['Volume']).rolling(3).mean() 
        
        df[['Close', 'OBV']][-100:].plot( subplots = True, grid = True )
    
        print(OBVcd(df, cd_period = 60))
    
        
    if AD:
        
        def AD(df, period=21):
            
            """
            :input: ohlcv DataFrame, pd.DataFrame
                    acds_period, int        
                    cd_period, int
                    
            :return: A/D line, pd.Series   
            
            1. Money Flow Multiplier = [(Close  -  Low) - (High - Close)] /(High - Low) 
            2. Money Flow Volume = Money Flow Multiplier x Volume for the Period
            3. ADL = Previous ADL + Current Period's Money Flow Volume
            """
            df['MFV'] = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])) * df['Volume']
            df['MFV'].replace(np.nan, 0, inplace = True)
            return df['MFV'].cumsum().rolling(period).mean()
        
        def ADcd(df, ad_period = 5, cd_period = 20):
            
            """
            :input: ohlcv DataFrame, pd.DataFrame
                    ad_period, int        
                    cd_period, int
            
            :return: obv last value, 
                     Buy/Sell label
            
            the accumulation/distribution line (A/D line), used to
            determine the money flow in and out of a security 
             
            If the indicator line is trending up, it shows buying interest,
            since the stock is closing above the halfway point of the range. 
            This helps confirm an uptrend. On the other hand, 
            if A/D is falling, that means the price is finishing in the 
            lower portion of its daily range, and thus volume is 
            considered negative. This helps confirm a downtrend. 
            
            Traders using the A/D line also watch for divergence. 
            If the A/D starts falling while the price is rising, 
            this signals that the trend is in trouble and could reverse.
            Similarly, if the price is trending lower and A/D starts rising, 
            that could signal higher prices to come. 
            
            """
            # compute indicator
            ad = AD(df, period = ad_period)
            # L = 3  # number of maximum values to consider
            price = df['Adj Close'][-cd_period:]
            price_lr = np.polyfit(range(len(price.index)), price.values, 1)
            
            if price_lr[0] > 0:    
                """ Valleys going Higher """
                price_cd = price[(price.diff() > 0).astype(int).diff() == 1]
                # ad_cd = ad[-cd_period:].nsmallest(L).sort_index()
                ad_cd = ad[-cd_period:][(ad.diff() > 0).astype(int).diff() == 1]
            else:
                """ Peaks going Lower """
                price_cd = price[(price.diff() > 0).astype(int).diff() == -1]
                # ad_cd = ad[-cd_period:].nlargest(L).sort_index()
                ad_cd = ad[-cd_period:][(ad.diff() > 0).astype(int).diff() == -1]
                
            # max values covergence / divergence check
            price_model = np.polyfit(range(len(price_cd.index)), price_cd.values, 1)
            ad_model = np.polyfit(range(len(ad_cd.index)), ad_cd.values, 1) 
            # check convergence / divergence
            label = ''
            if price_model[0] > 0 and ad_model[0] > 0:
                label = 'Up Convergence'
            if price_model[0] < 0 and ad_model[0] > 0:
                label = 'Up Divergence'
            elif price_model[0] < 0 and ad_model[0] < 0:
                label = 'Down Convegence'
            elif price_model[0] > 0 and ad_model[0] < 0:
                label = 'Down Divergence'
                
            signal = {'Down Divergence' : 1, 'Up Convergence' : 1,
                    'Down Convergence' : 0, 'Up Divergence' : 0}
                
            return { 'AD' : list(ad.values).pop() , 'ADcd' : signal[label] }
        
        n = 5
        
        df['AD'] = AD(df, period = n)
        df[['Adj Close', 'AD']][-60:].plot( subplots = True, grid = True )
        print(ADcd(df, ad_period = n, cd_period = 60))
        
    if ADX:
        
        def ADXsignal(df, period = 20):
            
            """
            :input: ohlcv DataFrame, pd.DataFrame
                    period, int
            
            :return: ADX value,
                     PLUS_DI value,
                     MINUS_DI value,
                     Buy/Sell label
                     
            The average directional index (ADX) is a trend indicator used 
            to measure the strength and momentum of a trend.
            
            When the ADX indicator is below 20, the trend is considered to 
            be weak or non-trending.
            
            * ADX above 20 and DI+ above DI-: That's an uptrend.
            * ADX above 20 and DI- above DI+: That's a downtrend.
            * ADX below 20 is a weak trend or ranging period, often associated 
                with the DI- and DI+ rapidly crisscrossing each other.
            
            """
            # set kwargs
            kwargs = {'high'   : df['High'],
                      'low'    : df['Low'],
                      'close'  : df['Close'],
                      'timeperiod' : period, 
                      }
            # calc ADX, PLUS_DI, MINUS_DI
            df['ADX'] = ta.ADX(**kwargs)            
            df['PLUS_DI'] = ta.PLUS_DI(**kwargs)  
            df['MINUS_DI'] = ta.MINUS_DI(**kwargs)
            
            # eval trending
            adx_value = df['ADX'].values[-1]
            if adx_value > 20:
                adx_signal = 'Trending'
            else:
                adx_signal = 'Ranging'
                
            # eval direction
            plus_di = df['PLUS_DI'].values[-1]
            minus_di = df['MINUS_DI'].values[-1]
            
            if plus_di > minus_di:
                directional_index = 'Up Trend'
            else:
                directional_index = 'Down Trend'
            
            if adx_signal == 'Trending' and directional_index == 'Up Trend':
                label = 'Buy'
            elif adx_signal == 'Trending' and directional_index == 'Down Trend':
                label = 'Sell'
            elif adx_signal == 'Ranging':
                label = 'Not Buy'
    
            return {'ADX': adx_value, 'PLUS_DI' : plus_di, 'MINUS_DI' : minus_di, 'ADXsignal' : label}

    
        # set kwargs
        kwargs = {'high'   : df['High'],
                  'low'    : df['Low'],
                  'close'  : df['Close'],
                  'timeperiod' : 20, 
                  }
        
        # calc ADX
        df['ADX'] = ta.ADX(**kwargs)            
        df[['Adj Close', 'ADX']][-60:].plot( subplots = True, grid = True )
        print(ADXsignal(df, period = 20))
        
        
    if AROON:
        
        
        def AROONsignal(df, period = 25):
            
            """
            :input: ohlcv DataFrame, pd.DataFrame
                    period, int
            
            :return: AROON_UP value,
                     AROON_DOWN value,
                     Buy/Sell label
            
            The Aroon oscillator is a technical indicator used to measure whether 
            a security is in a trend, and more specifically if the price is 
            hitting new highs or lows over the calculation period (typically 25). 
            
            The indicator can also be used to identify when a new trend is 
            set to begin. The Aroon indicator comprises two lines: an Aroon-up line 
            and an Aroon-down line. 
            
            """
            
            # set kwargs
            kwargs = {'high'   : df['High'],
                      'low'    : df['Low'],
                      'timeperiod' : period, 
                      }
                
            df['AROON_DOWN'], df['AROON_UP'] = ta.AROON(**kwargs)
            value_down, value_up = df['AROON_DOWN'].values[-1], df['AROON_UP'].values[-1]
            
            if value_up > value_down:
                label = 'BUY'
            else:
                label = 'SELL'
        
            return {'AROON_UP' : value_up, 'AROON_DOWN' : value_down, 'AROONsignal' : label}
        
        
        n = 25
        
        # set kwargs
        kwargs = {'high'   : df['High'],
                  'low'    : df['Low'],
                  'timeperiod' : n, 
                  }
    
        df[[ 'Adj Close' , 'AROON_UP', 'AROON_DOWN']].plot(subplots = True, grid = True)
        print(AROONsignal(df, period = n))
        
        
    if MACD:
        
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
            price = df['Adj Close'][-(slow+n):]
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
                    value_label = 'BUY'
                else:
                    value_label = 'NOT BUY'
                
                if macd_hist > 0:
                    signal_label = 'BUY'
                else:
                    signal_label = 'NOT BUY'
                
            else:
                value_label = 'SELL'
                signal_label = 'SELL'
            
            return {'MACD value' : macd_value, 'MACD hist' : macd_hist, 'MACD label' : value_label, 'MACD hist label' : signal_label}
            
        # set kwargs
        kwargs = {'fastperiod'    : 12,
                  'slowperiod' : 26,
                  'signalperiod': 9
                  }
        
        # check indicator
        df['MACD'], df['MACDsignal'], df['MACDhist'] = ta.MACD(df['Close'], **kwargs)

    
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(df['Adj Close'])
        axs[0].grid()
        axs[1].plot(df[['MACD', 'MACDsignal', 'MACDhist']])
        axs[1].grid()
        axs[1].axhline(0.0, lw = 1, ls = 'dashed', color = 'black')
        plt.show()
        
        print(MACDsignal(df))
        
    if RSI:
        
        # set kwargs
        kwargs = { 'timeperiod' : 10,        
                  }
            
        df['RSI'] = ta.RSI(df['Close'], **kwargs)
        
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(df['Close'])
        axs[0].grid()
        axs[1].plot(df['RSI'])
        axs[1].grid()
        axs[1].axhline(70, lw = 1, ls = 'dashed', color = 'black')
        axs[1].axhline(30, lw = 1, ls = 'dashed', color = 'black')
        plt.show()
        
        pass
        
    if STOCH:
           
        def Stoch(df, fast = 26, signal = 9, levels = (20, 80)):
            
            """
            :input: ohlcv DataFrame, pd.DataFrame
                    period, int
            
            :return: stoch_d value, float
                     stoch_k value, flaot
            
            Stochastic Oscillator
            
            """
            # set kwargs
            kwargs = {'high'  : df['Close'],
                      'low'   : df['Close'],
                      'close' : df['Close'],
                      'fastk_period' : fast,
                      'fastd_period' : signal,
                      'fastd_matype'  : 1,
                      }
                
            df['FASTk'], df['FASTd'] = ta.STOCHF(**kwargs)
            # calc close mean
            n = 3
            price = df['Close'][-(fast+n):].rolling(n).mean()
            # select valleys
            valleys = price[(price.diff() > 0).astype(int).diff() == 1]
            # calculate uptrend
            price_lr = np.polyfit(range(len(valleys.index)), valleys.values, 1)            
            # stoch values to evaluate
            stoch_value = df['FASTk'].values[-1]
            # check valleys going higher
            if price_lr[0] > 0:
                if stoch_value < levels[1]:    
                    value_label = 1 # 'BUY'
                elif stoch_value > levels[1]:
                    value_label = 0 # 'SELL' 
                
            else:
                value_label = 0 # 'NOT BUY'
            
            return {'STOCH value' : stoch_value, 'STOCH Signal' : value_label}
            

        
        # set kwargs
        kwargs = {'high'  : df['Close'],
                  'low'   : df['Close'],
                  'close' : df['Close'],
                  'fastk_period' : 26,
                  'fastd_period' : 9,
                  'fastd_matype'  : 1,
                  }
            
        n = -100
        
        df['FASTk'], df['FASTd'] = ta.STOCHF(**kwargs)
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(df['Close'][-n:])
        axs[0].grid()
        axs[1].plot(df[['FASTk', 'FASTd']][-n:])
        axs[1].grid()
        axs[1].axhline(0.0, lw = 1, ls = 'dashed', color = 'black')
        plt.show()
        
        
        print(Stoch(df, fast = 26, signal = 9))
        
    if STOCHRSI:
        
        # fastk, fastd = STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        
         
        def STOCHRSIsignal(df, rsi_period = 26, stoch_period = (5, 3), levels = (30, 70)):
        
            """
            :input: ohlcv DataFrame, pd.DataFrame
                    period, int
            
            :return: STOCHRSI value,
                     STOCHRSI uptrend,
                     STOCHRSI downtrend,
            
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
            
            return {'STOCHRSIk' : stoch_rsi_value, 'STOCHRSI uptrend' : up_trend, 'STOCHRSI downtrend' : down_trend,}
            
        # set kwargs
        kwargs = { 'timeperiod'   : 26, 
                   'fastk_period' : 5, 
                   'fastd_period' : 3, 
                   'fastd_matype' : 1,
                  }
            
        df['FASTk'], df['FASTd'] = ta.STOCHRSI(df['Close'], **kwargs)
        n = 150
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(df['Close'][-n:])
        axs[0].grid()
        axs[1].plot(df[['FASTk', 'FASTd']][-n:])
        axs[1].grid()
        axs[1].axhline(70, lw = 1, ls = 'dashed', color = 'black')
        axs[1].axhline(30, lw = 1, ls = 'dashed', color = 'black')
        plt.show()
        
        print(STOCHRSIsignal(df, rsi_period = 26, stoch_period=(5, 3), levels = (30, 70)))

    if TRIX:
        
        
        def TRIXsignal(df, period = 9):
            
            """
            :input: ohlcv DataFrame, pd.DataFrame
                    period, int
            
            :return: TRIX value,
                     TRIX signal,
            
            The Triple Exponential Average (TRIX) is a momentum indicator 
            used by technical traders that shows the percentage change in a 
            triple exponentially smoothed moving average. 
            When it is applied to triple smoothing of moving averages, 
            it is designed to filter out price movements that are considered 
            insignificant or unimportant.
            
            """
            # set kwargs
            kwargs = { 'timeperiod'   : 9, 
                     }
            
            df['TRIX'] = ta.TRIX(df['Close'], **kwargs)
                
            value = df['TRIX'].tolist().pop()
            bull = int(value > 0)
            bear = int(value < 0)
    
            return {'TRIX value' : value, 'TRIX Bullish' : bull, 'TRIX Bearish' : bear }
        
        
        # set kwargs
        kwargs = { 'timeperiod'   : 9, 
                 }
            
        df['TRIX'] = ta.TRIX(df['Close'], **kwargs)
        
        df[['Close', 'TRIX']].plot(subplots = True, grid = True)
        plt.axhline( y = 0.0,lw = 1.0, ls = 'dashed', color = 'black')
        
        print(TRIXsignal(df, period = 9))
        
    
    if BB:
        
        def BBsignal(df, period = 21, factor = 2.0):
        
            """
            :input: ohlcv DataFrame, pd.DataFrame
                    period, int
            
            :return: BB value,
                     BB signal,
            
            A common approach when using Bollinger Bands® is to identify 
            overbought or oversold market conditions. When the price of the 
            asset breaks below the lower band of the Bollinger Bands, 
            prices have perhaps fallen too much and are due to bounce. 
            On the other hand, when price breaks above the upper band, 
            the market is perhaps overbought and due for a pullback. 
            
            Using the bands as overbought/oversold indicators relies on the 
            concept of mean reversion of the price. Mean reversion assumes 
            that, if the price deviates substantially from the mean or average,
            it eventually reverts back to the mean price. 
            
            """
            
            # set kwargs
            kwargs = { 'timeperiod' : period,
                       'nbdevup' : factor,
                       'nbdevdn' : factor,
                       'matype' : 1,
                     }
            
            df['BBup'], _, df['BBdown'] = ta.BBANDS(df['Close'], **kwargs)
                
            price = df['Close'].to_list().pop()
            bbup  = df['BBup'].shift(1).to_list().pop()
            bbdo  = df['BBdown'].shift(1).to_list().pop()
            
            if price > bbup: signal = -1
            elif price < bbdo: signal = 1
            else: signal = 0
                
            return {'BBup' : bbup, 'BBdown' : bbdo, 'price' : price, 'BB signal' : signal }
        
         # set kwargs
        kwargs = { 'timeperiod' : 9,
                   'nbdevup' : 2.0,
                   'nbdevdn' : 2.0,
                   'matype' : 1.0,
                 }
        
        df['BBup'], df['BBmd'], df['BBdown'] = ta.BBANDS(df['Close'], **kwargs)
        
        df[['BBup', 'BBdown', 'Close']].plot()
        
        print(BBsignal(df, period = 10, factor = 1.0))
        
    if MFI:
        
        def MFIsignal(df, period = 10, levels = (20, 80)):
            
            """
            :input: ohlcv DataFrame, pd.DataFrame
                    period, int
            
            :return: MFI value,
                     MFI signal,
            
            The Money Flow Index (MFI) is a technical indicator that generates 
            overbought or oversold signals using both prices and volume data.
            
            An MFI reading above 80 is considered overbought and an MFI reading
            below 20 is considered oversold, although levels of 90 and 10 
            are also used as thresholds. 
            
            Unlike conventional oscillators such as the Relative Strength Index 
            (RSI), the Money Flow Index incorporates both price and 
            volume data, as opposed to just price. For this reason, some 
            analysts call MFI the volume-weighted RSI
            
            """
            # set kwargs
            kwargs = { 'high' : df['High'],
                       'low' : df['Low'],
                       'close' : df['Close'],
                       'volume' : df['Volume'],
                       'timeperiod' : period,
                     }
                
            df['MFI'] = ta.MFI(**kwargs)
            
            value = df['MFI'].to_list().pop()
            overb = int(value > levels[1])
            overs = int(value < levels[0])
                        
            return {'MFI value' : value, 'MFI OverBought' : overb, 'MFI OverSold' : overs } 
        
        # set kwargs
        kwargs = { 'high' : df['High'],
                   'low' : df['Low'],
                   'close' : df['Close'],
                   'volume' : df['Volume'],
                   'timeperiod' : 10,
                 }
        
        df['MFI'] = ta.MFI(**kwargs)
        
        df[['Close', 'MFI']].plot(subplots = True, grid = True)
        plt.axhline(y = 80, lw = 1, ls = 'dashed', color = 'black')
        plt.axhline(y = 20, lw = 1, ls = 'dashed', color = 'black')
        
        print(MFIsignal(df, period = 21))
        
    if VWAP:
        
        df['Volume'] = df['Volume'] * df['Close']
        def vwap(df):
            return (df['Volume'] * ((df['High'] + df['Low']) /2)).cumsum() / df['Volume'].cumsum()
        
        def VWAPsignal(df):
            
            """
            :input: ohlcv DataFrame, pd.DataFrame
                    period, int
            
            :return: VWAP value,
                     VWAP signal,
            
            The volume weighted average price (VWAP) is a trading benchmark 
            used by traders that gives the average price a security has traded 
            at throughout the day, based on both volume and price. 
            It is important because it provides traders with insight into both 
            the trend and value of a security. 
           
            """
            
            def vwap(df):
                return (df['Volume'] * ((df['High'] + df['Low']) /2)).cumsum() / df['Volume'].cumsum()
        
            
            df['VWAP'] = vwap(df) 
            signal = df['VWAP'].to_list().pop()
            price = df['Close'].to_list().pop()
            overb = int(price > signal)
            overs = int(price < signal)
           
            return {'VWAP' : signal, 'Price' : price, 'VWAP OverBougth' : overb, 'VWAP OverSold' : overs }
         
        df['VWAP'] = vwap(df)
        df[['Close', 'VWAP']].plot()
        
        print(VWAPsignal(df))
        
    if KC:
        
        def KCsignal(df, period = 14, factor = 1.0):
            
            """
            
            :input: ohlcv DataFrame, pd.DataFrame
                    period, int
            
            :return: KCup value,
                     KCdown value,
                     KC Bullish,
                     KC Bearish,
            
            Keltner Channels are volatility-based bands that are placed on 
            either side of an asset's price and can aid in determining the 
            direction of a trend. The exponential moving average (EMA) of a 
            Keltner Channel is typically 20 periods, although this can be 
            adjusted if desired.
            The upper and lower bands are typically set two times the average 
            true range (ATR) above and below the EMA, although the multiplier 
            can also be adjusted based on personal preference.
            Price reaching the upper Keltner Channel band is bullish, while 
            reaching the lower band is bearish.
            The angle of the Keltner Channel also aids in identifying the 
            trend direction. The price may also oscillate between the upper 
            and lower Keltner Channel bands, which can be interpreted as 
            resistance and support levels.

            """
            
            # atr kwargs
            kwargs_atr = {'high'  : df['High'], 
                          'low'   : df['Low'],
                          'close' : df['Close'],
                          'timeperiod' : period}
            
            df['ATR'] = ta.ATR(**kwargs_atr)
            
            kwargs_ema = {'timeperiod' : period,
                          }
            df['EMA'] = ta.EMA(df['Close'], **kwargs_ema)
            
            df['KCup'], df['KCdown'] = df['EMA'] + df['ATR'] * factor, df['EMA'] - df['ATR'] * factor
        
            price = df['Close'].to_list().pop()
            kcup  = df['KCup'].shift(1).to_list().pop()
            kcdo  = df['KCdown'].shift(1).to_list().pop()
            
            if price > kcup: bull = 1
            else: bull = 0 
            if price < kcdo: bear = 1
            else: bear = 0
                
            return {'KCup' : kcup, 'KCdown' : kcdo, 'price' : price, 'KC Bullish' : bull, 'KC Bearish' : bear }
        
        p = 10
        # atr kwargs
        kwargs_atr = {'high'  : df['High'], 
                      'low'   : df['Low'],
                      'close' : df['Close'],
                      'timeperiod' : p}
        df['ATR'] = ta.ATR(**kwargs_atr)
        
        kwargs_ema = {'timeperiod' : p,
                       }
        
        df['EMA'] = ta.EMA(df['Close'], **kwargs_ema)
        
        factor = 0.5
        
        df['KCup'], df['KCdown'] = df['EMA'] + df['ATR'] * factor, df['EMA'] - df['ATR'] * factor
        
        df[['KCup', 'Close', 'KCdown']].plot()
        
        print(KCsignal(df, period = 10, factor = 0.5))

    if DC:
        
        def DCsignal(df, period = 10):
            
            """
            
            :input: ohlcv DataFrame, pd.DataFrame
                    period, int
            
            :return: DCup value,
                     DCmid value,
                     DCdown value,
                     DC Bullish,
                     DC Bearish,
            
            The indicator seeks to identify bullish and bearish extremes that 
            favor reversals as well as higher and lower breakouts, breakdowns, 
            and emerging trends.
            The middle band simply computes the average between the highest
            high over N periods and the lowest low over N periods, 
            identifying a median or mean reversion price.


            """
            
            def DonchianChannels(df, period = 10):
                df['DCup'], df['DCdown'] = df['High'].rolling(period).max(), df['Low'].rolling(period).min() 
                df['DCmid'] = (df['DCup'] - df['DCdown'])/2 + df['DCdown']
                return df['DCup'], df['DCmid'], df['DCdown']
        
            
            df['DCup'], df['DCmid'], df['DCdown'] = DonchianChannels(df, period = 20)
        
            price = df['Close'].to_list().pop()
            dcup  = df['DCup'].shift(1).to_list().pop()
            dcmid  = df['DCmid'].shift(1).to_list().pop()
            dcdo  = df['DCdown'].shift(1).to_list().pop()
            
            if price > dcup: bull = 1
            else: bull = 0 
            if price < dcdo: bear = 1
            else: bear = 0
            
            return {'DCup' : dcup, 'DCmid' : dcmid, 'DCdown' : dcdo, 'DC Bullish' : bull, 'DC Bearish' : bear }
        
        
        df['DCup'], df['DCmid'], df['DCdown'] = DonchianChannels(df, period = 20)
        
        df[['DCup', 'Close', 'DCdown']].plot()
        
        print(DCsignal(df, period = 10))

    if MOM:
        
        def MOMsignal(df, period = 10):
            
            """
            
            :input: ohlcv DataFrame, pd.DataFrame
                    period, int
            
            :return: MOM value,
                     MOM Bullish,
                     MOM Bearish,
            
            Momentum is the speed or velocity of price changes in a 
            stock, security, or tradable instrument.
            Momentum shows the rate of change in price movement over
            a period of time to help investors determine 
            the strength of a trend.
            Investors use momentum to trade stocks whereby a stock 
            can exhibit bullish momentum–the price is rising–or 
            bearish momentum–the price is falling.

            """
            # atr kwargs
            kwargs = { 'timeperiod' : 10,
                      }
            
            df['MOM'] = ta.MOM(df['Close'], **kwargs)
            
            value = df['MOM'].to_list().pop()
            bull = int(value > 0)
            bear = int(value < 0)
        
            return {'MOM value' : value, 'MOM Bullish' : bull, 'MOM Bearish' : bear, }
        
        # atr kwargs
        kwargs = { 'timeperiod' : 10,
                  }
        
        df['MOM'] = ta.MOM(df['Close'], **kwargs)
    
        df[['Close', 'MOM']].plot( subplots = True)
        plt.axhline( y = 0.0, lw = 1.0, ls = 'dashed', color = 'black')
        
        print(MOMsignal(df, period = 14))
    
    
    if CCI:
        
            
        kwargs = {'high'  : df['High'],
                  'low'   : df['Low'],
                  'close' : df['Close'],
                  'timeperiod' : 55,
                  }
        
        df['CCI'] = ta.CCI(**kwargs)
        
        df[['Close', 'CCI']].plot( subplots = True, grid = True)
        
        
        def CCIsignal(df, period = 20):
            
            """
            :input: ohlcv DataFrame, pd.DataFrame
                    period, int
            
            :return: CCI value,
                     CCI Bullish,
                     CCI Bearish,
           
            The Commodity Channel Index (CCI) is a technical indicator 
            that measures the difference between the current price 
            and the historical average price.
            When the CCI is above zero, it indicates the price is 
            above the historic average. Conversely, when the CCI is 
            below zero, the price is below the historic average.
            The CCI is primarily used for spotting new trends.
            When the CCI moves from negative or near-zero territory 
            to above 100, that may indicate the price is starting 
            a new uptrend.
            The same concept applies to an emerging downtrend.
            When the indicator goes from positive or near-zero 
            readings to below -100, then a downtrend may be starting.
            
            """
            
            kwargs = {  'high'  : df['High'],
                        'low'   : df['Low'],
                        'close' : df['Close'],
                        'timeperiod' : 10,
                        }
            
            df['CCI'] = ta.CCI(**kwargs)
            value = df['CCI'].to_list().pop()
            bull = int(value > 100)
            bear = int(value < -100)
            
            return {'CCI value ' : value, 'CCI Bullish' : bull, 'CCI Bearish' : bear }
            

        
        print(CCIsignal(df, period = 34))
        
        
    if SAR:
        
        # real = SAR(high, low, acceleration=0, maximum=0)

        kwargs = {'high'  : df['High'],
                  'low'   : df['Low'],
                  'acceleration' : 1,
                  'maximum' : 1,
                  }
        
        
        df['SAR'] = ta.SAR(**kwargs)
    
        df[['Close', 'SAR']][-100:].plot( grid = True )
        
        
        def SARsignal(df, params = (0.5, 0.5)):
            
            """
            :input: ohlcv DataFrame, pd.DataFrame
                    period, int
            
            :return: SAR value,
                     SAR Bullish,
                     SAR Bearish,
                     
            The parabolic SAR is a technical indicator used to 
            determine the price direction of an asset, as well as 
            draw attention to when the price direction is changing. 
            Sometimes known as the "stop and reversal system," the 
            parabolic SAR was developed by J. Welles Wilder Jr., 
            creator of the relative strength index (RSI)
            
            A SAR is placed below the price when it is trending upward,
            and above the price when it is trending downward.
            
            """
            # set kwargs
            kwargs = {'high'  : df['High'],
                      'low'   : df['Low'],
                      'acceleration' : params[0],
                      'maximum' : params[1],
                      }
        
            # SAR
            df['SAR'] = ta.SAR(**kwargs)
            price = df['Close'].to_list().pop()
            value = df['SAR'].to_list().pop()
            bull = int(price > value)
            bear = int(price < value)

            return { 'SAR' : value, 'SAR Bullish' : bull, 'SAR Bearish' : bear }        

        print(SARsignal(df))


    if BOP:
        
      
        def BOPsignal(df, period = 10):
            
            """
            :input: ohlcv DataFrame, pd.DataFrame
                    period, int
            
            :return: BOP value,
                     BOP Bullish,
                     BOP Bearish,
                     
            The Balance of Power indicator shows the direction and 
            extent of price change during the trading period. 
            A rising BOP line indicates an upward trend and a 
            falling BOP line indicates a downward trend
            
            """
            # set kwargs
            kwargs = {'open'  : df['Open'],
                      'high'  : df['High'],
                      'low'   : df['Low'],
                      'close' : df['Close'],
                      }
        
            df['BOP'] = ta.BOP(**kwargs)
            df['BOPm'] = ta.EMA(df['BOP'], period)
            
            value = df['BOPm'].to_list().pop()
            bull = int(value > 0)
            bear = int(value < 0)
            return { 'BOP' : value, 'BOP Bullish' : bull, 'BOP Bearish' : bear }        
        
        kwargs = {'open'  : df['Open'],
                 'high'  : df['High'],
                 'low'   : df['Low'],
                 'close'   : df['Close'],
                }
      
        df['BOP'] = ta.BOP(**kwargs)
        df[['Close', 'BOP']][-100:].plot(subplots = True, grid = True)
        df['BOPm'] = ta.EMA(df['BOP'], 10)       
        df[['Close', 'BOPm']][-100:].plot(subplots = True, grid = True)
        
        print(BOPsignal(df, period = 10))
        
    if WIR:
        
        # real = WILLR(high, low, close, timeperiod=14)
      
        def WIRsignal(df, period = 10):
            
            """
            
            :input: ohlcv DataFrame, pd.DataFrame
                    period, int
            
            :return: WIR value,
                     WIR Bullish,
                     WIR Bearish,
                     
            The indicator is telling a trader where the current price 
            is relative to the highest high over the last 14 periods 
            (or whatever number of lookback periods is chosen).

            When the indicator is between -20 and zero the price is 
            overbought, or near the high of its recent price range. 
            When the indicator is between -80 and -100 the price 
            is oversold, or far from the high of its recent range.

            During an uptrend, traders can watch for the indicator 
            to move below -80. When the price starts moving up, and 
            the indicator moves back above -80, it could signal that 
            the uptrend in price is starting again.

            The same concept could be used to find short trades in a 
            downtrend. When the indicator is above -20, watch for the
            price to start falling along with the Williams %R moving 
            back below -20 to signal a potential continuation of the 
            downtrend. 
            
            """
            kwargs = { 'high'   : df['High'],
                   'low'    : df['Low'],
                   'close'  : df['Close'],
                   'timeperiod' : period,
                 }
      
            df['WIR'] = ta.WILLR(**kwargs)
        
            value = df['WIR'].to_list().pop()
            bull = int(value < -80)
            bear = int(value > -20)
            
            return { 'WIR' : value, 'WIR Bullish' : bull, 'WIR Bearish' : bear }        
        
        kwargs = { 'high'   : df['High'],
                 'low'    : df['Low'],
                 'close'  : df['Close'],
                 'timeperiod' : 10,
               }
      
        df['WIR'] = ta.WILLR(**kwargs)
        df[['Close', 'WIR']][-100:].plot(subplots = True, grid = True)
        
        print(WIRsignal(df, period = 20))
        
    if CHK:
        
        # real = ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        
        def CHKsignal(df, periods = (3, 10)):
            
            """
            
            :input: ohlcv DataFrame, pd.DataFrame
                    periods, tuple int
            
            :return: CHK value,
                     CHK Bullish,
                     CHK Bearish,
                     
            

            The Chaikin Indicator applies MACD to the 
            accumulation-distribution line rather than closing price.
            A cross above the accumulation-distribution line 
            indicates that market players are accumulating shares, 
            securities or contracts, which is typically bullish.
            
            """
            kwargs = { 'high'   : df['High'],
                    'low'    : df['Low'],
                    'close'  : df['Close'],
                    'volume'  : df['Volume'],
                    'fastperiod' : 3,
                    'slowperiod' : 10,
                    }
        
            df['CHK'] = ta.ADOSC(**kwargs)
            
            value = df['CHK'].to_list().pop()
            bull = int(value > 0)
            bear = int(value < 0)
            
            return {'Chaikin Osc' : value, 'Chaikin Bullish' : bull, 'Chaikin Bearish' : bear }
            
        
        kwargs = { 'high'   : df['High'],
                    'low'    : df['Low'],
                    'close'  : df['Close'],
                    'volume'  : df['Volume'],
                    'fastperiod' : 3,
                    'slowperiod' : 10,
                    }
        
        df['CHK'] = ta.ADOSC(**kwargs)

        df[['Close', 'CHK']][-100:].plot(subplots = True, grid = True)
       
        print(CHKsignal(df, periods = (3, 10)))