# used for time
import config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics

import time
import datetime
import talib
import numpy as np
import pandas as pd
import Historical_Data as HD

# needed for the binance API / websockets / Exception handling
from binance.client import Client
from binance.exceptions import BinanceAPIException
from requests.exceptions import ReadTimeout, ConnectionError
from binance.enums import *


class Trade(HD.HistoricalData):

    def __init__(self, symbol, interval):
        HD.HistoricalData.__init__(self, symbol, interval)
        self.optimal_ratio: float = 0.999
        self.quantity: float = 0.039
        self.minutes = 30

        self.recent_LOHCV = np.array([[]], dtype="d")

        self.df = pd.DataFrame()
        self.up_down = np.array([], dtype="int32")
        # latest 50 intervals
        self.close = np.array([], dtype="d")
        self.high = np.array([], dtype="d")
        self.low = np.array([], dtype="d")
        self.volume = np.array([], dtype="d")
        self.open = np.array([], dtype="d")

        self.test_interval = Client.KLINE_INTERVAL_1MINUTE

    def set_price(self) -> str:
        # Fetch 500 most recent price
        klines = HD.client.get_klines(
            symbol=self.symbol, interval=self.interval, limit=25)

        self.recent_LOHCV = np.array(klines, dtype="d")

    def get_history_timestamp(self) -> str:
        return self.recent_LOHCV[-1][0]

    def append_current_to_df(self):
        predictors = {
            "EMA": self.ema_arr,
            "CMO": self.cmo_arr,
            "MINUSDM": self.minusdm_arr,
            "PLUSDM": self.plusdm_arr,
            "CLOSE": self.close_arr,
            "CLOSEL1": self.closel1_arr,
            "CLOSEL2": self.closel2_arr,
            "CLOSEL3": self.closel3_arr,
            "3O": self.threeoutsideupdown,
            "CMB": self.closingmaru
        }
        self.df = pd.DataFrame(predictors)

        if (self.close_arr[-1] > self.close_arr[-2]):
            self.up_down = np.append(self.up_down, 1)
        else:
            self.up_down = np.append(self.up_down, 0)

        self.df["UP_DOWN"] = self.up_down
        self.df = self.df.dropna()

    def append_history_to_df(self):
        predictors = {
            "EMA": self.ema_arr,
            "CMO": self.cmo_arr,
            "MINUSDM": self.minusdm_arr,
            "PLUSDM": self.plusdm_arr,
            "CLOSE": self.close_arr,
            "CLOSEL1": self.closel1_arr,
            "CLOSEL2": self.closel2_arr,
            "CLOSEL3": self.closel3_arr,
            "3O": self.threeoutsideupdown,
            "CMB": self.closingmaru
        }

        self.df = pd.DataFrame(predictors)

        # Modify up_down array such that timelag = 1
        self.up_down = np.append(self.up_down, np.nan)
        for i in range(self.close_arr.size-1):
            j = i+1
            if (self.close_arr[i] != np.nan):
                if (self.close_arr[j] > self.close_arr[i]):
                    self.up_down = np.append(self.up_down, 1)
                else:
                    self.up_down = np.append(self.up_down, 0)
            else:
                self.up_down[i] = np.nan

        # add a column for logistic regression
        self.df["UP_DOWN"] = self.up_down
        # drop na
        # time lag = 1
        self.df["UP_DOWN"] = self.df["UP_DOWN"].shift(-1)
        self.df = self.df.dropna()
        self.df["UP_DOWN"] = self.df["UP_DOWN"].astype(np.int64)

    # Update Technical Indicators per 5 minutes(after model prediction)
    def UpdateModelperInterval(self):
        self.set_price()

        # update recent array, get latest TA array from them
        self.close_arr = np.append(self.close_arr, self.recent_LOHCV[-1][4])
        self.high_arr = np.append(self.high_arr, self.recent_LOHCV[-1][2])
        self.low_arr = np.append(self.low_arr, self.recent_LOHCV[-1][3])
        self.open_arr = np.append(self.open_arr, self.recent_LOHCV[-1][1])
        self.volume_arr = np.append(self.volume_arr, self.recent_LOHCV[-1][5])

        self.closel1_arr = np.append(
            self.closel1_arr, self.recent_LOHCV[-2][4])
        self.closel2_arr = np.append(
            self.closel2_arr, self.recent_LOHCV[-3][4])
        self.closel3_arr = np.append(
            self.closel3_arr, self.recent_LOHCV[-4][4])

        # update TA arrays to time t
        self.ema_arr = talib.EMA(self.close_arr, self.timelag)
        self.cmo_arr = talib.CMO(self.close_arr, self.timelag)
        self.minusdm_arr = talib.MINUS_DM(
            self.high_arr, self.low_arr, self.timelag)
        self.plusdm_arr = talib.PLUS_DM(
            self.high_arr, self.low_arr, self.timelag)
        # recent close
        self.threeoutsideupdown = talib.CDL3OUTSIDE(
            self.open_arr, self.high_arr, self.low_arr, self.close_arr)
        self.closingmaru = talib.CDLCLOSINGMARUBOZU(
            self.open_arr, self.high_arr, self.low_arr, self.close_arr)

        if self.threeoutsideupdown[-1] == 100:
            self.threeoutsideupdown[-1] = 1
        if self.threeoutsideupdown[-1] == -100:
            self.threeoutsideupdown[-1] == 0
        if self.closingmaru[-1] == 100:
            self.closingmaru[-1] = 1
        if self.closingmaru[-1] == -100:
            self.closingmaru[-1] == 0

        # todo


    def FitModel(self):
        predictors = ["CMO","MINUSDM", "PLUSDM", "CLOSE", "3O", "CMB"]
        # Split test & training set
        X = self.df[predictors]  # predictor
        y = self.df.UP_DOWN  # response
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1)

        # Normalize data
        scaler = StandardScaler().fit(X_train[predictors[0:-2]])
        X_train[predictors[0:-2]] = scaler.transform(X_train[predictors[0:-2]])
        X_test[predictors[0:-2]] = scaler.transform(X_test[predictors[0:-2]])

        # Fit Model using training data
        svm = SVC(kernel="rbf", C=1)
        svm = svm.fit(X_train, y_train)

        return svm, predictors, scaler


    def GetPrediction(self, svm, predictors, scaler):
        

        # Prediction

        X_new = self.df.iloc[-1:]
        X_new = X_new[predictors]

        X_new[predictors[0:-2]] = scaler.transform(X_new[predictors[0:-2]])

        up_down = svm.predict(X_new)

        return up_down[0]

    def SetQuantity(self):
        curr_price: float = self.close_arr[-1]
        usdt_balance: float = float(
            HD.client.get_asset_balance(asset="USDT")['free'])

        # Kelly's criteriation
        self.optimal_ratio: float = 0.999
        self.quantity = self.optimal_ratio * (usdt_balance/curr_price)
        self.quantity = round(self.quantity, 4)

        print("Amount of BTC Long Position: ", self.quantity)

    def long(self, sym: str, size: float, p: str):
        order = HD.client.order_limit_buy()(
            symbol=sym,
            quantity=size,
            price=p
        )

        return order

    def short(self, sym, size):
        order = HD.client.order_market_sell(
            symbol=sym,
            quantity=size
        )

        return order


# human readable format
def datetime_from_utc_to_local(utc_datetime):
    now_timestamp = time.time()
    offset = datetime.datetime.fromtimestamp(
        now_timestamp) - datetime.datetime.utcfromtimestamp(now_timestamp)
    return utc_datetime + offset


def get_now_timestamp() -> datetime.datetime:
    """
    Returns today's timestamp
    """
    curr_time = datetime.datetime.now()
    now = curr_time.timestamp()
    return now


def convertTimestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp/1000)


def main():

    if config.trade_interval == "1m":
        trade = Trade("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE)
        trade.minutes = 1

    if config.trade_interval == "3m":
        trade = Trade("BTCUSDT", Client.KLINE_INTERVAL_3MINUTE)
        trade.minutes = 3

    if config.trade_interval == "15m":
        trade = Trade("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE)
        trade.minutes = 15

    if config.trade_interval == "30m":
        trade = Trade("BTCUSDT", Client.KLINE_INTERVAL_30MINUTE)
        trade.minutes = 30

    if config.trade_interval == "4h":
        trade = Trade("BTCUSDT", Client.KLINE_INTERVAL_4HOUR)
        trade.minutes = 60*4


    print("Appending Historical Data...")

    trade.InitAllHistoricalData()
    trade.append_history_to_df()
    svm, predictors, scaler = trade.FitModel()

    print("Finished Appending Historical Data...")

    now = int(get_now_timestamp())*1000

    prev_usdt_balance = HD.client.get_asset_balance(asset="USDT")
    prev_BTC_balance = HD.client.get_asset_balance(asset="BTC")
    print("\n", "Your Balance: ", prev_usdt_balance, prev_BTC_balance, "\n")
    last_quantity: float = trade.quantity

    # get the latest&first timestamp in string
    delta = 60000*trade.minutes
    trade.set_price()
    latest_timestamp: int = trade.get_history_timestamp()
    first_dealtime: int = int(latest_timestamp+delta)

    print("Binance Auto Trading Starts...\n")
    print("Time for first trade: ", convertTimestamp(first_dealtime))

    isLong:bool = True
    global order #Store order status

    # time increment
    dealtime = first_dealtime  # for continuous orders

    while True:
        now: int = int(datetime.datetime.now().timestamp())*1000
        # sync first deal
        if (now == first_dealtime):
            time.sleep(1)

            trade.UpdateModelperInterval()  # predictor arr is now filled
            trade.append_current_to_df()  # update whole model
            if trade.GetPrediction(svm, predictors, scaler) == 0:
                isLong = False
                print("Shorting is not allowed in the developer's jurisdiction. No action is performed. \n")

            elif (trade.GetPrediction(svm, predictors, scaler) == 1):
                isLong = True
                #   set quantity

                trade.SetQuantity()
                last_quantity = trade.quantity

                #   First order
                order = HD.client.order_limit_buy(
                    symbol=trade.symbol, quantity=trade.quantity, price=str(trade.close_arr[-1]))

                print("Long", order)

                usdt_balance = HD.client.get_asset_balance(asset="USDT")
                BTC_balance = HD.client.get_asset_balance(asset="BTC")
                print("Your Balance: ", usdt_balance, BTC_balance, "\n")
            
            dealtime += delta
            print("Time for next Trade: ", convertTimestamp(dealtime))


        if (now == dealtime):
            time.sleep(1)
            trade.UpdateModelperInterval()  # predictor arr is now filled
            trade.append_current_to_df()  # update whole model

            #   Get the order status from prev time.
            if isLong:

                #Outcomes:
                #1. Long (Not filled) -> Predict Long -> Cancel old. Order new
                #2. Long (Not filled) -> Predict Short -> Cancel old, Do nothing
                #3. Long (Filled) -> Predict Long -> Do Nothing
                #4. Long (Filled) -> Predict Short -> Close Long -> Do Nothing

                order_status = HD.client.get_order(symbol=trade.symbol, orderId=order["orderId"])

                LongIsFilled: bool
                if (order_status["status"] == "FILLED"):
                    LongIsFilled = True
                else:
                    LongIsFilled = False

                #Outcome 3
                if trade.GetPrediction(svm, predictors, scaler) == 1 and LongIsFilled == True:
                    #   Do Nothing since Long -> Long
                    print("Previous order is filled")
                    print("Long position remains open.\n")
                    isLong = True
                    LongIsFilled = True
                
                #Outcome 4
                elif trade.GetPrediction(svm, predictors, scaler) == 0 and LongIsFilled == True:
                    #   Close Long Order at market price since Long -> Short
                    order = HD.client.order_market_sell(
                        symbol=trade.symbol, quantity=last_quantity)
                    print("Close Long \n", order)
                    LongIsFilled = True
                    isLong = False

                    usdt_balance = HD.client.get_asset_balance(asset="USDT")
                    BTC_balance = HD.client.get_asset_balance(asset="BTC")
                    print("Your Balance: ", usdt_balance, BTC_balance)
                    print("P/L in this trade: ",
                            float(usdt_balance["free"]) - float(prev_usdt_balance["free"]), "\n")
                    prev_usdt_balance = usdt_balance

                    print("Shorting is not allowed in the developer's jurisdiction. No action is performed. \n")
                #Outcome 1
                elif trade.GetPrediction(svm, predictors, scaler) == 1 and LongIsFilled == False:
                    #Cancel old order

                    result = HD.client.cancel_order(symbol=trade.symbol, orderId=order["orderId"])
                    print("Order Cancelled. ", result, "\n")

                    prev_usdt_balance = HD.client.get_asset_balance(asset="USDT")
                    trade.SetQuantity()
                    

                    #New order
                    order = HD.client.order_limit_buy(
                        symbol=trade.symbol, quantity=trade.quantity, price=str(trade.close_arr[-1]))
                    print("Long \n", order)
                    last_quantity = trade.quantity
                    isLong = True
                    LongIsFilled = False
                    

                    

                #Outcome 2
                elif trade.GetPrediction(svm, predictors, scaler) == 0 and LongIsFilled == False:

                    result = HD.client.cancel_order(symbol=trade.symbol, orderId=order["orderId"])
                    print("Order Cancelled. ", result, "\n")
                    print("Shorting is not allowed in the developer's jurisdiction. No action is performed. \n")
                    isLong = False


            else:
                #Outcomes
                #1. Short -> Predict Long -> Long
                #2. Short -> Predict Short -> Do Nothing
                print("No position to be closed. \n")
                #Case 2
                if(trade.GetPrediction(svm, predictors, scaler) == 0):
                    print("Shorting is not allowed in the developer's jurisdiction. No action is performed. \n")
                    isLong = False
                #Case 1
                else:
                    prev_usdt_balance = HD.client.get_asset_balance(asset="USDT")
                    trade.SetQuantity()
                    
                    order = HD.client.order_limit_buy(
                        symbol=trade.symbol, quantity=trade.quantity, price=str(trade.close_arr[-1]))
                    print("Long \n", order)
                    isLong = True
                    last_quantity = trade.quantity
                    

                    usdt_balance = HD.client.get_asset_balance(asset="USDT")
                    BTC_balance = HD.client.get_asset_balance(asset="BTC")
                    print("Your Balance: ", usdt_balance, BTC_balance)
                    

            dealtime += delta
            print("Time for next Trade: ", convertTimestamp(dealtime))



main()
