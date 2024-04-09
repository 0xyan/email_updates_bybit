from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import ccxt
import smtplib
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import schedule
import time
import os
from dotenv import load_dotenv
from files_handling import datetime_list, btceth_lists, equity_list
from beta_var import beta_calc

load_dotenv()


def bybit_init():
    bybit_api_key = os.getenv("BYBIT_API_KEY")
    bybit_secret = os.getenv("BYBIT_SECRET")
    client = ccxt.bybit(
        {
            "apiKey": bybit_api_key,
            "secret": bybit_secret,
            "enableRateLimit": True,
        }
    )

    return client


def daily_change(client, symbol):

    daily_change = round(
        float(client.fetch_ticker(symbol)["info"]["price24hPcnt"]) * 100, 2
    )

    return daily_change


def token_price(client, symbol):

    price = float(client.fetch_ticker(symbol)["info"]["lastPrice"])

    return price


def get_balance(client):
    total_equity = round(
        float(client.fetch_balance()["info"]["result"]["list"][0]["totalEquity"]), 2
    )
    return total_equity


def get_positions(client):
    raw_positions = client.fetch_positions()

    pos_ticker = []
    pos_size = []
    for i in raw_positions:
        pos_ticker.append(i["info"]["symbol"])
        pos_size.append(round(float(i["info"]["size"]), 3))
    pos_dict = dict(zip(pos_ticker, pos_size))

    return pos_dict


def get_exposure(client, all_positions, total_equity):
    gross_exposure = 0
    for k, v in all_positions.items():
        token_price_k = token_price(client, k)
        b = token_price_k * float(v)
        gross_exposure += abs(b)
        gross_exposure = round(gross_exposure, 2)

    net_exposure = 0
    for k, v in all_positions.items():
        token_price_k = token_price(client, k)
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


def email_doc_creation(
    df,
    client,
    all_positions,
    gross_exposure_pct,
    net_exposure_pct,
    total_equity,
    beta,
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
    week_vol_strategy = round(df["strategy_return"].tail(7).std() * (365**0.5) * 100, 2)
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

    daily_email = open(
        r"/Users/mac/Desktop/GMY/python_projects/email_updates_bybit/daily_email.txt",
        "w",
    )
    daily_email.write(f"Recorded {df.shape[0]} days\n")
    daily_email.write(
        f"Total return since inception: {round(((total_equity/100000)-1)*100, 2)}%\n"
    )
    daily_email.write(f"Sharpe ratio: {sharpe}\n")
    daily_email.write(f"Sortino ratio: {sortino}\n")
    daily_email.write(f"Correlation w/ BTC: {corr_btc}\n")

    daily_email.write("\n Return\t\tDaily\t\tWeekly\t\tMonthly\t\tTotal\n")
    daily_email.write(
        f"\n LSA\t\t{last_day_pnl}%\t\t{last_week_pnl}%\t\t{last_month_pnl}%\t\t{total_pnl}%"
    )
    daily_email.write(
        f"\n ETH\t\t{last_day_eth}%\t\t{last_week_eth}%\t\t{last_month_eth}%\t\t{total_eth_return}%"
    )
    daily_email.write(
        f"\n BTC\t\t{last_day_btc}%\t\t{last_week_btc}%\t\t{last_month_btc}%\t\t{total_btc_return}%\n"
    )

    daily_email.write("\n Volatility\tWeekly\t\tMonthly\t\tTotal\n")
    daily_email.write(
        f"\n LSA\t\t{week_vol_strategy}%\t\t{month_vol_strategy}%\t\t{total_vol_strategy}%"
    )
    daily_email.write(
        f"\n ETH\t\t{week_vol_eth}%\t\t{month_vol_eth}%\t\t{total_vol_eth}%"
    )
    daily_email.write(
        f"\n BTC\t\t{week_vol_btc}%\t\t{month_vol_btc}% \t\t{total_vol_btc}%\n"
    )

    total_beta_exp = 0
    daily_email.write("\n Positions\tAmount\t\tSize\t\tBeta\tBeta exp.\t24h ret.\n")
    for k, v in all_positions.items():
        token_price_k = token_price(client, k)
        daily_change_k = daily_change(client, k)
        size_k = round(v * token_price_k, 2)
        beta_exposure_k = round(beta[k] * size_k, 2)
        daily_email.write(
            f" \n {k}\t{float(v)}\t\t${size_k}\t{round(beta[k], 2)}\t${beta_exposure_k}\t{daily_change_k}%"
        )
        total_beta_exp += beta_exposure_k

    daily_email.write("\n \n Exposure: \n")
    daily_email.write(f" \n Gross%: {round(gross_exposure_pct, 2)}%")
    daily_email.write(f"\n Net%: {round(net_exposure_pct, 2)}%")
    daily_email.write(f"\n Beta exp.%: {round(total_beta_exp/total_equity*100,2)}%")
    daily_email.write(f"\n Beta exp.$: ${round(total_beta_exp, 2)} \n")

    daily_email.write(f"\nTotal equity: ${total_equity}")

    daily_email.close()


# MAIL FILE


def mail_send(date_time_name):
    contacts = [os.getenv("EMAIL")]
    msg = MIMEMultipart()
    msg["Subject"] = f"Strategy Performance {date_time_name}"
    msg["From"] = os.getenv("EMAIL")
    msg["To"] = ", ".join(contacts)

    # Attach the email text content
    with open("daily_email.txt", "r") as file:  # Use a different variable name here
        text_content = file.read()
    text_part = MIMEText(text_content, "plain")  # Explicitly specify subtype as 'plain'
    msg.attach(text_part)

    # Attach the image
    attachment_path = f"Strategy Performance {date_time_name}.png"
    with open(attachment_path, "rb") as file:
        img_part = MIMEImage(file.read())
        img_part.add_header(
            "Content-Disposition",
            "attachment",
            filename=os.path.basename(attachment_path),
        )
    msg.attach(img_part)

    # Send the email
    with smtplib.SMTP("127.0.0.1", 1025) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(os.getenv("EMAIL"), os.getenv("EMAIL_KEY"))
        smtp.send_message(msg)


def main():
    client = bybit_init()
    total_equity = get_balance(client)
    all_positions = get_positions(client)
    net_exposure_pct, gross_exposure_pct = get_exposure(
        client, all_positions, total_equity
    )
    datetime_series = datetime_list()
    btcprice = token_price(client, "BTCUSDT")
    ethprice = token_price(client, "ETHUSDT")
    btcprice = round(float(btcprice), 2)
    ethprice = round(float(ethprice), 2)
    btc_series, eth_series = btceth_lists(btcprice, ethprice)
    equity_series = equity_list(total_equity)

    beta = beta_calc(client, all_positions)

    df = dataframe(datetime_series, equity_series, btc_series, eth_series)
    date_time = plot(df)
    email_doc_creation(
        df,
        client,
        all_positions,
        gross_exposure_pct,
        net_exposure_pct,
        total_equity,
        beta,
    )
    mail_send(date_time)


if __name__ == "__main__":
    main()


"""
schedule.every().day.at("00:00").do(exec)

while True:
    schedule.run_pending()
    time.sleep(1)
"""
