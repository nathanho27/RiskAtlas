"""Market Data Pipeline pulls all S&P 500 tickers,downloads prices,and stores in PostgreSQL.Output:prices(date,ticker,adj_close)"""

import pandas as pd
import yfinance as yf
import psycopg2
from io import StringIO
import requests

# establish connection to local PostgreSQL database
def get_connection():
    return psycopg2.connect(dbname="risk_atlas",user="nathanho",host="localhost",port="5432")

# fetch S&P 500 ticker universe from Wikipedia and clean symbols for Yahoo Finance
def get_sp500_tickers():
    url="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers={"User-Agent":"Mozilla/5.0"}
    response=requests.get(url,headers=headers,timeout=30)
    response.raise_for_status()
    tables=pd.read_html(StringIO(response.text),match="Symbol")
    table=tables[0]
    tickers=table["Symbol"].tolist()
    tickers=[t.replace(".","-") for t in tickers]
    return tickers

# download historical prices in chunks and return long format DataFrame date,ticker,adj_close
def get_market_prices(tickers,start_date="2010-01-01"):
    all_prices=[]
    chunk_size=50

    for i in range(0,len(tickers),chunk_size):
        chunk=tickers[i:i+chunk_size]
        print(f"Downloading {i}->{i+len(chunk)}")

        try:
            raw=yf.download(chunk,start=start_date,progress=False,threads=False,auto_adjust=True)
        except Exception as e:
            print("Error:",e)
            continue

        if raw.empty:
            continue

        # extract closing prices and reshape to long format
        prices=raw["Close"]
        prices=(prices.reset_index().melt(id_vars="Date",var_name="ticker",value_name="adj_close").rename(columns={"Date":"date"}).dropna())

        all_prices.append(prices)

    return pd.concat(all_prices,ignore_index=True)

# store cleaned price data into PostgreSQL and create indexes for faster queries
def save_to_db(df):
    conn=get_connection()
    cur=conn.cursor()

    df["date"]=pd.to_datetime(df["date"])
    df["adj_close"]=df["adj_close"].astype(float)

    cur.execute("""DROP TABLE IF EXISTS prices;
CREATE TABLE prices(
    date DATE,
    ticker TEXT,
    adj_close DOUBLE PRECISION
);""")

    buffer=StringIO()
    df.to_csv(buffer,index=False,header=False)
    buffer.seek(0)

    cur.copy_from(buffer,"prices",sep=",")

    cur.execute("CREATE INDEX idx_ticker ON prices(ticker);")
    cur.execute("CREATE INDEX idx_date ON prices(date);")

    conn.commit()
    cur.close()
    conn.close()

# run full pipeline fetch tickers download prices save to database
def main():
    tickers=get_sp500_tickers()
    print(f"Total tickers:{len(tickers)}")

    df=get_market_prices(tickers)
    print(f"Rows:{len(df)}")

    save_to_db(df)

    print("DONE")

if __name__=="__main__":
    main()