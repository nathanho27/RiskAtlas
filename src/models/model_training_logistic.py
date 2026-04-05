# model_training_logistic.py
# trains logistic regression model to predict risk_event (future 10-day drop)

import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from io import StringIO

# load model dataset from postgres
def load_data():
    conn=psycopg2.connect(dbname="risk_atlas",user="nathanho",host="localhost",port="5432")
    query="SELECT * FROM model_dataset"
    df=pd.read_sql(query,conn)
    conn.close()
    return df

def main():
    df=load_data()

    # drop missing values
    df=df.dropna().copy()

    # feature set
    features=[
        "daily_return",
        "vol_20","vol_30","vol_60",
        "ma_20","ma_50","ma_200",
        "price_to_ma50","price_to_ma200"
    ]

    X=df[features]
    y=df["risk_event"]

    # chronological split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)

    # model
    model=LogisticRegression(max_iter=1000,class_weight="balanced")
    model.fit(X_train,y_train)

    # predictions
    y_prob=model.predict_proba(X_test)[:,1]
    y_pred=(y_prob>0.5).astype(int)

    # evaluation
    print(classification_report(y_test,y_pred))

    print(pd.DataFrame({
        "actual":y_test.values[:10],
        "predicted":y_pred[:10],
        "probability":y_prob[:10]
    }))

    # build predictions dataframe
    test_df=df.loc[X_test.index].copy()

    pred_df=pd.DataFrame({
    "date":test_df["date"].values,
    "ticker":test_df["ticker"].values,
    "risk_prob":y_prob,
    "risk_pred":y_pred
    })

    # write to postgres
    conn=psycopg2.connect(dbname="risk_atlas",user="nathanho",host="localhost",port="5432")
    cur=conn.cursor()

    cur.execute("""
    DROP TABLE IF EXISTS predictions;
    CREATE TABLE predictions(
        date DATE,
        ticker TEXT,
        risk_prob DOUBLE PRECISION,
        risk_pred INT
    );
    """)

    buffer=StringIO()
    pred_df.to_csv(buffer,index=False,header=False)
    buffer.seek(0)

    cur.copy_from(buffer,"predictions",sep=",")

    conn.commit()
    cur.close()
    conn.close()

    print("predictions table created")

if __name__=="__main__":
    main()