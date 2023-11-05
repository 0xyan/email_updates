from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib
import pandas as pd
from datetime import datetime
import asyncio
import matplotlib.pyplot as plt
import schedule
import time
import os
from binance import AsyncClient
from dotenv import load_dotenv
from files_handling import datetime_list, btceth_lists, equity_list

load_dotenv()


def binance_init():
    binance_api_key = os.getenv("BINANCE_API_KEY")
    binance_secret = os.getenv("BINANCE_SECRET")
    client = AsyncClient(binance_api_key, binance_secret)

    return client


async def daily_change(client, symbol):
    for i in await client.futures_ticker():
        if i["symbol"] == symbol:
            daily_change = round(float(i["priceChangePercent"]), 2)
        else:
            continue
    return daily_change


async def token_price(client, symbol):
    ticker = await client.futures_symbol_ticker(symbol=symbol)
    price = round(float(ticker["price"]), 2)
    return price


async def get_balance(client):
    perp_account_info = await client.futures_account()
    margin_account_info = await client.get_margin_account()
    # fetching perp balance
    perp_balance = perp_account_info["totalMarginBalance"]
    perp_balance = round(float(perp_balance), 2)
    # fetching margin balance
    btc_price = await token_price(client, "BTCUSDT")
    margin_balance = float(margin_account_info["totalNetAssetOfBtc"]) * float(btc_price)
    margin_balance = round(float(margin_balance), 2)
    total_equity = round(margin_balance + perp_balance, 2)
    return total_equity


async def get_positions(client):
    perp_account_info = await client.futures_account()
    margin_account_info = await client.get_margin_account()
    raw_positions_perp = perp_account_info["positions"]
    raw_positions_margin = margin_account_info["userAssets"]

    # creating dictionary with perp positions
    perp_pos_ticker = []
    perp_pos_size = []
    for i in raw_positions_perp:
        if i["notional"] != "0":
            perp_pos_ticker.append(i["symbol"])
            perp_pos_size.append(round(float(i["positionAmt"]), 3))
    perp_pos_dict = dict(zip(perp_pos_ticker, perp_pos_size))

    # creating dictionary with margin positions
    margin_pos_ticker = []
    margin_pos_size = []
    for i in raw_positions_margin:
        if i["netAsset"] != "0":
            if i["asset"] != "USDT":
                margin_pos_ticker.append(i["asset"] + "USDT")
                margin_pos_size.append(round(float(i["netAsset"]), 3))
    margin_pos_dict = dict(zip(margin_pos_ticker, margin_pos_size))

    # total poisitons
    all_positions = {**perp_pos_dict, **margin_pos_dict}

    return perp_pos_dict, margin_pos_dict, all_positions


async def get_exposure(client, all_positions, total_equity):
    gross_exposure = 0
    for k, v in all_positions.items():
        token_price_k = await token_price(client, k)
        b = token_price_k * float(v)
        gross_exposure += abs(b)
        gross_exposure = round(gross_exposure, 2)

    net_exposure = 0
    for k, v in all_positions.items():
        token_price_k = await token_price(client, k)
        b = token_price_k * float(v)
        net_exposure = net_exposure + b
        net_exposure = round(net_exposure, 2)

    net_exposure_pct = (net_exposure / total_equity) * 100
    gross_exposure_pct = (gross_exposure / total_equity) * 100

    return net_exposure_pct, gross_exposure_pct


############## datetime file/list management
############## equity file/list management
############## btc & eth files/lists management


# Creating a dataframe with with historical returns
def dataframe(datetime_series, equity_series, btc_series, eth_series):
    df = pd.DataFrame()
    df["datetime"] = pd.to_datetime(datetime_series)
    df["equity"] = equity_series
    df["btc_price"] = btc_series
    df["eth_price"] = eth_series
    df["strategy_return"] = df["equity"].pct_change(1)
    df["btc_return"] = df["btc_price"].pct_change(1)
    df["eth_return"] = df["eth_price"].pct_change(1)
    df["cum_ret_strategy"] = (df["strategy_return"] + 1).cumprod() - 1
    df["cum_ret_btc"] = (df["btc_return"] + 1).cumprod() - 1
    df["cum_ret_eth"] = (df["eth_return"] + 1).cumprod() - 1
    df = df.fillna(0)
    return df


# creating a plot
def plot(df):
    now = datetime.now()
    date_time_name = now.strftime("%Y-%m-%d %H-%M")
    plt.figure(figsize=(12, 8))
    plt.plot(df["datetime"], df["cum_ret_strategy"], "g", label="Account equity")
    plt.plot(df["datetime"], df["cum_ret_btc"], "y", label="BTC")
    plt.plot(df["datetime"], df["cum_ret_eth"], "b", label="ETH")
    plt.legend(loc="upper left", fontsize=15)
    plt.grid(axis="y")
    plt.xlabel("date", fontsize=15)
    plt.ylabel("performance", fontsize=15)
    plt.title("Strategy relative performance", fontsize=15)
    plt.savefig(f"Strategy Performance {date_time_name}.png")
    return date_time_name


############ performance & stdev calculation
############ EMAIL DOC


# calculating performance
def perf_calc(dataframe, series, timeframe):
    if timeframe > len(dataframe):
        a = "no data"
    else:
        a = round(
            ((dataframe[series].iloc[-1]) / (dataframe[series].iloc[-timeframe]) - 1)
            * 100,
            2,
        )
    return a


async def email_doc_creation(
    df,
    client,
    perp_pos_dict,
    margin_pos_dict,
    gross_exposure_pct,
    net_exposure_pct,
    total_equity,
):
    # calculating returns
    last_day_pnl = perf_calc(df, "equity", 2)
    last_week_pnl = perf_calc(df, "equity", 8)
    last_month_pnl = perf_calc(df, "equity", 31)
    total_pnl = perf_calc(df, "equity", -0)

    last_day_eth = perf_calc(df, "eth_price", 2)
    last_week_eth = perf_calc(df, "eth_price", 8)
    last_month_eth = perf_calc(df, "eth_price", 31)
    total_eth_return = perf_calc(df, "eth_price", -0)

    last_day_btc = perf_calc(df, "btc_price", 2)
    last_week_btc = perf_calc(df, "btc_price", 8)
    last_month_btc = perf_calc(df, "btc_price", 31)
    total_btc_return = perf_calc(df, "btc_price", -0)

    # calculating stdevs
    week_vol_strategy = round(
        df["strategy_return"].tail(7).std() * (365**0.5) * 100, 2
    )
    month_vol_strategy = round(
        df["strategy_return"].tail(30).std() * (365**0.5) * 100, 2
    )
    total_vol_strategy = round(df["strategy_return"].std() * (365**0.5) * 100, 2)

    week_vol_eth = round(df["eth_return"].tail(7).std() * (365**0.5) * 100, 2)
    month_vol_eth = round(df["eth_return"].tail(30).std() * (365**0.5) * 100, 2)
    total_vol_eth = round(df["eth_return"].std() * (365**0.5) * 100, 2)

    week_vol_btc = round(df["btc_return"].tail(7).std() * (365**0.5) * 100, 2)
    month_vol_btc = round(df["btc_return"].tail(30).std() * (365**0.5) * 100, 2)
    total_vol_btc = round(df["btc_return"].std() * (365**0.5) * 100, 2)

    # sharpe, sortino, correlation
    sharpe = round(
        (df["strategy_return"].mean() / df["strategy_return"].std()) * (365**0.5), 2
    )
    sortino = round(
        (
            df["strategy_return"].mean()
            / df["strategy_return"][df["strategy_return"] < 0].std()
        )
        * (365**0.5),
        2,
    )
    corr_btc = round(df["strategy_return"].corr(df["btc_return"]), 2)

    ############### EMAIL DOC

    daily_email = open("daily_email.txt", "w")
    daily_email.write(f"Recorded {df.shape[0]} days\n")
    daily_email.write(f"Sharpe ratio: {sharpe}\n")
    daily_email.write(f"Sortino ratio: {sortino}\n")
    daily_email.write(f"Correlation w/ BTC: {corr_btc}\n")

    daily_email.write("\n Performance\tdaily\t\tweekly\t\t30D\t\ttotal\n")
    daily_email.write(
        f"\n LSA\t\t{last_day_pnl}%\t\t{last_week_pnl}%\t\t{last_month_pnl}%\t\t{total_pnl}%"
    )
    daily_email.write(
        f"\n ETH\t\t{last_day_eth}%\t\t{last_week_eth}%\t\t{last_month_eth}%\t\t{total_eth_return}%"
    )
    daily_email.write(
        f"\n BTC\t\t{last_day_btc}%\t\t{last_week_btc}%\t\t{last_month_btc}%\t\t{total_btc_return}%\n"
    )

    daily_email.write("\n Volatility\tweekly\t\t30D\t\ttotal\n")
    daily_email.write(
        f"\n LSA\t\t{week_vol_strategy}%\t\t{month_vol_strategy}%\t\t{total_vol_strategy}%"
    )
    daily_email.write(
        f"\n ETH\t\t{week_vol_eth}%\t\t{month_vol_eth}%\t\t{total_vol_eth}%"
    )
    daily_email.write(
        f"\n BTC\t\t{week_vol_btc}%\t\t{month_vol_btc}% \t\t{total_vol_btc}%\n"
    )

    daily_email.write("\n\n Positions: \n")
    daily_email.write("\n Perps:")
    for k, v in perp_pos_dict.items():
        token_price_k = await token_price(client, k)
        daily_email.write(
            f" \n {k} \t {float(v)} \t ${round(v * token_price_k,2)} \t {daily_change(k)} %"
        )

    if margin_pos_dict:
        daily_email.write("\n\n Margin:")
        for k, v in margin_pos_dict.items():
            token_price_k = await token_price(client, k)
            daily_email.write(
                f" \n {k} \t {float(v)} \t ${round(v * token_price_k,2)} \t {daily_change(k)} %"
            )

    daily_email.write("\n \n Exposure: \n")
    # daily_email.write(f" \n Gross: {gross_exposure}")
    daily_email.write(f" \n Gross%: {round(gross_exposure_pct, 2)}%")
    # daily_email.write(f"\n Net : {net_exposure}")
    daily_email.write(f"\n Net% : {round(net_exposure_pct, 2)}% \n")

    daily_email.write(f"\nTotal equity: ${total_equity}")

    daily_email.close()


# MAIL FILE
def mail_send(date_time_name):
    contacts = []
    contacts.append(os.getenv("EMAIL"))
    msg = MIMEMultipart()
    msg["Subject"] = f"Strategy Performance {date_time_name}"
    msg["From"] = os.getenv("EMAIL")
    msg["To"] = ", ".join(contacts)

    filename = "daily_email.txt"
    with open(filename, "r") as filename:
        text = MIMEText(filename.read())
    msg.attach(text)

    attachment = f"Strategy Performance {date_time_name}.png"

    with open(attachment, "rb") as fp:
        img = MIMEImage(fp.read())
    msg.attach(img)

    with smtplib.SMTP("127.0.0.1", 1025) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()

        smtp.login(os.getenv("EMAIL"), os.getenv("EMAIL_KEY"))

        smtp.send_message(msg)


async def main():
    client = binance_init()
    total_equity = await get_balance(client)
    perp_pos_dict, margin_pos_dict, all_positions = await get_positions(client)
    net_exposure_pct, gross_exposure_pct = await get_exposure(
        client, all_positions, total_equity
    )
    datetime_series = datetime_list()
    btcprice = await token_price(client, "BTCUSDT")
    ethprice = await token_price(client, "ETHUSDT")
    btcprice = round(float(btcprice), 2)
    ethprice = round(float(ethprice), 2)
    btc_series, eth_series = btceth_lists(btcprice, ethprice)
    equity_series = equity_list(total_equity)
    df = dataframe(datetime_series, equity_series, btc_series, eth_series)
    date_time = plot(df)
    email_doc_creation(
        df,
        client,
        perp_pos_dict,
        margin_pos_dict,
        gross_exposure_pct,
        net_exposure_pct,
        total_equity,
    )
    mail_send(date_time)


if __name__ == "__main__":
    asyncio.run(main())


"""
schedule.every().day.at("00:00").do(exec)

while True:
    schedule.run_pending()
    time.sleep(1)
"""
