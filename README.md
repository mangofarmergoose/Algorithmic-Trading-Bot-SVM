# Algorithmic-Trading-Bot-SVM

### Description

This Binance trading bot analyses the Up/Down probability of user defined interval using supervised statistical learning techniques. 
Support Vector Machine is used in this trading bot. You may check the jupyter notebook for implementation details. 
For details about SVM, click [here](https://en.wikipedia.org/wiki/Support_vector_machine). Due to compilance problems, short position is not available. 
BTC/USDT spot trading pair is selected due to Binance's 0% trading fee. A 56% correct classification rate is observed with minimal multicollinearity.

### Trading Logic

The bot initalizes historical data from 15 June, 2022 onwards. After loading, it creates a limit order at your specified interval. If the predicted long probability > 0.5, a long order would be executed via limit order at time t. Depending on the prediction on time t+1, a market order may be created at time t+1 to close out the long position at time t.

> Example

1. Goose selects interval = 5 mins. The bot finishes loading at 20:07.
2. During 20:07 and 20:10, nothing will happen.
3. At 20:10, the bot predicts long. The bot longs BTC.
4. At 20:15, the bot predicts long/short.
5. If long is predicted, long position remains open. Else long position is closed.
6. Loop between Step 3 and Step 5

### READ BEFORE USE

1. Check the isTestNet variable in config.py = False. Otherwise you will be using REAL money.

### Binance Guide

For Testnet, please follow this [guide](https://www.binance.com/en/support/faq/ab78f9a1b8824cf0a106b4229c76496d)

For Mainnet, please follow this [guide](https://www.binance.com/en/support/faq/360002502072)

If you do not have a binance account, please consider registering with the link below.
https://www.binance.com/en/activity/referral-entry/CPA?fromActivityPage=true&ref=CPA_00SB23W192

### Setup Guide
---
Must have Python 3.9+ Installed.
Libraries Required:
- numpy
- pandas
- sklearn
- [talib](https://github.com/mrjbq7/ta-lib)
- [python-binance](https://github.com/sammchardy/python-binance)

```
#If you're using macOS

git clone https://github.com/mangofarmergoose/Algorithmic-Trading-Bot
cd Algorithmic-Trading-Bot

pip3 install numpy
pip3 install pandas
pip3 install sklearn

python3 bot.py #To run the bot
```
For talib and python-binance, please follow the original documentations linked above.

---


### Configuration
---
Open config.py.
- api_key_test = "your_testnet_api_key"
- api_secret_test = "your_testnet_api_secret_key"
- api_key_real = "your_real_api_key"
- api_secret_real = "your_real_api_secret_key"
- isTestNet = True/False
- trade_interval = "5m" or "15m" or "1h" (15m is recommended)
---

