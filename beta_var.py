import pandas as pd
from binance import AsyncClient


async def get_klines(client, symbol):
    try:
        dfi = await client.futures_continous_klines(
            pair=symbol, interval="1h", contractType="PERPETUAL", limit="350"
        )
        dfi = pd.DataFrame(dfi)
        df = pd.DataFrame()
        df["time"] = pd.to_datetime(dfi[0].astype(float), unit="ms")
        df["close"] = dfi[4].astype(float)
        df[f"{symbol}"] = df["close"].pct_change(1)
        df.set_index("time", inplace=True)
        df = df[[symbol]]
        df.dropna(inplace=True)
    except Exception as e:
        print(f"error processing {symbol}: {e}")

    return df


async def beta_calc(client, positions):
    df = pd.DataFrame()
    for symbol in positions.keys():
        if df.empty:
            try:
                df = await get_klines(client, symbol)
            except Exception as e:
                print(f"error processing {symbol}: {e}")
        else:
            try:
                dfi = await get_klines(client, symbol)
                df[symbol] = dfi[symbol]
            except Exception as e:
                print(f"error processing {symbol}: {e}")
    df["BTCUSDT"] = await get_klines(client, "BTCUSDT")

    # betas
    covariance = df.cov()
    beta = covariance["BTCUSDT"] / df["BTCUSDT"].var()
    beta = beta.round(2)

    return beta
