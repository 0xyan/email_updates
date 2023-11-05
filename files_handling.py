from datetime import datetime
import pandas as pd
import numpy as np


# CREATING A DATETIME FILE/LIST
def datetime_list():
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M")

    time_file = open(r"C:\Users\Administrator\Desktop\LSAstrategy\time_txt.txt", "a")
    time_file.write("\n" + date_time)
    time_file.close()

    datetime_list_raw = open(
        r"C:\Users\Administrator\Desktop\LSAstrategy\time_txt.txt"
    ).readlines()
    datetime_list = []
    for i in datetime_list_raw:
        datetime_list.append(i.rstrip())
    datetime_series = pd.Series(datetime_list)

    return datetime_series


# CREATING AN EQUITY FILE/LIST
def equity_list(total_equity):
    equity_file = open(
        r"C:\Users\Administrator\Desktop\LSAstrategy\equity_txt.txt", "a"
    )
    equity_file.write(f"\n{total_equity}")
    equity_file.close()

    equity_list_raw = open(
        r"C:\Users\Administrator\Desktop\LSAstrategy\equity_txt.txt"
    ).readlines()
    equity_list = []
    for i in equity_list_raw:
        equity_list.append(i.rstrip())
    equity_list = list(np.float_(equity_list))
    equity_series = pd.Series(equity_list)

    return equity_series


# STORING BTC & ETH PRICES
def btceth_lists(btcprice, ethprice):
    # writing current prices
    btc_file = open(r"C:\Users\Administrator\Desktop\LSAstrategy\btc_prices.txt", "a")
    eth_file = open(r"C:\Users\Administrator\Desktop\LSAstrategy\eth_prices.txt", "a")
    btc_file.write(f"\n{btcprice}")
    eth_file.write(f"\n{ethprice}")
    btc_file.close()
    eth_file.close()

    # btc series
    btc_list_raw = open(
        r"C:\Users\Administrator\Desktop\LSAstrategy\btc_prices.txt"
    ).readlines()
    btc_list = []
    for i in btc_list_raw:
        btc_list.append(i.rstrip())
    btc_list = list(np.float_(btc_list))
    btc_series = pd.Series(btc_list)

    # eth series
    eth_list_raw = open(
        r"C:\Users\Administrator\Desktop\LSAstrategy\eth_prices.txt"
    ).readlines()
    eth_list = []
    for i in eth_list_raw:
        eth_list.append(i.rstrip())
    eth_list = list(np.float_(eth_list))
    eth_series = pd.Series(eth_list)

    return btc_series, eth_series
