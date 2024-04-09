import pandas as pd


def get_klines(client, symbol):
    df = pd.DataFrame()
    try:
        dfi = pd.DataFrame(client.fetch_ohlcv(symbol, timeframe="1h", limit=350))
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


def beta_calc(client, positions):
    df = pd.DataFrame()
    for symbol in positions.keys():
        if df.empty:
            try:
                df = get_klines(client, symbol)
            except Exception as e:
                print(f"error processing {symbol}: {e}")
        else:
            try:
                dfi = get_klines(client, symbol)
                df[symbol] = dfi[symbol]
            except Exception as e:
                print(f"error processing {symbol}: {e}")
    df["BTCUSDT"] = get_klines(client, "BTCUSDT")

    # betas
    covariance = df.cov()
    beta = covariance["BTCUSDT"] / df["BTCUSDT"].var()
    beta = beta.round(2)

    return beta
