# model_training_logistic.py
# trains a logistic regression baseline for future 10-day downside risk

import os
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import psycopg2

from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)


DB_NAME="risk_atlas"
DB_USER="nathanho"


# connection used when writing predictions
def get_connection():
    return psycopg2.connect(dbname=DB_NAME,user=DB_USER,host="localhost",port="5432")


# engine used when loading data with pandas
def get_engine():
    return create_engine(f"postgresql+psycopg2://{DB_USER}@localhost:5432/{DB_NAME}")


# load model dataset in chronological order
def load_data():
    query="""
    SELECT
        date,
        ticker,
        daily_return,
        vol_20,
        vol_30,
        vol_60,
        price_to_ma50,
        price_to_ma200,
        risk_event
    FROM model_dataset
    ORDER BY date,ticker;
    """

    engine=get_engine()

    with engine.connect() as conn:
        df=pd.read_sql(query,conn)

    engine.dispose()
    df["date"]=pd.to_datetime(df["date"])

    return df


# create chronological train, validation, and test sets
def chronological_split(df,train_size=0.70,validation_size=0.15,purge_days=10):
    dates=df["date"].drop_duplicates().sort_values().reset_index(drop=True)

    train_split=int(len(dates)*train_size)
    validation_split=int(len(dates)*(train_size+validation_size))

    validation_start=dates.iloc[train_split]
    test_start=dates.iloc[validation_split]

    train_end=dates.iloc[train_split-purge_days]
    validation_end=dates.iloc[validation_split-purge_days]

    train_df=df[df["date"]<=train_end].copy()

    validation_df=df[
        (df["date"]>=validation_start)
        & (df["date"]<=validation_end)
    ].copy()

    test_df=df[df["date"]>=test_start].copy()

    return train_df,validation_df,test_df


# choose classification threshold using validation data
def select_threshold(y_true,y_score):
    precision,recall,thresholds=precision_recall_curve(y_true,y_score)

    precision=precision[:-1]
    recall=recall[:-1]

    f1_scores=2*precision*recall/(precision+recall+1e-10)
    best_index=np.argmax(f1_scores)

    return {
        "threshold":thresholds[best_index],
        "precision":precision[best_index],
        "recall":recall[best_index],
        "f1":f1_scores[best_index]
    }


# print model evaluation metrics
def evaluate_model(name,y_true,y_score,threshold):
    y_pred=(y_score>=threshold).astype(int)

    print(f"\n{name}")
    print("-"*50)
    print(f"Threshold: {threshold:.4f}")
    print(f"Risk-event prevalence: {y_true.mean():.4f}")

    print("\nClassification Report")
    print(classification_report(y_true,y_pred,digits=4,zero_division=0))

    print("Confusion Matrix")
    print(confusion_matrix(y_true,y_pred))

    print(f"\nROC-AUC: {roc_auc_score(y_true,y_score):.4f}")
    print(f"PR-AUC: {average_precision_score(y_true,y_score):.4f}")

    return y_pred


# save test predictions to postgres
def save_predictions(pred_df):
    conn=get_connection()
    cur=conn.cursor()

    cur.execute("""
    DROP TABLE IF EXISTS predictions;

    CREATE TABLE predictions(
        date DATE NOT NULL,
        ticker TEXT NOT NULL,
        actual_risk_event INT NOT NULL,
        risk_score DOUBLE PRECISION NOT NULL,
        risk_pred INT NOT NULL,
        model_name TEXT NOT NULL,
        PRIMARY KEY(date,ticker,model_name)
    );
    """)

    buffer=StringIO()
    pred_df.to_csv(buffer,index=False,header=False)
    buffer.seek(0)

    cur.copy_expert("""
    COPY predictions(
        date,
        ticker,
        actual_risk_event,
        risk_score,
        risk_pred,
        model_name
    )
    FROM STDIN WITH CSV
    """,buffer)

    conn.commit()
    cur.close()
    conn.close()


def main():
    df=load_data()
    df=df.dropna().copy()

    features=[
        "daily_return",
        "vol_20",
        "vol_30",
        "vol_60",
        "price_to_ma50",
        "price_to_ma200"
    ]

    train_df,validation_df,test_df=chronological_split(df)

    X_train=train_df[features]
    y_train=train_df["risk_event"]

    X_validation=validation_df[features]
    y_validation=validation_df["risk_event"]

    X_test=test_df[features]
    y_test=test_df["risk_event"]

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Testing rows: {len(test_df):,}")

    print(f"Training period: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"Validation period: {validation_df['date'].min().date()} to {validation_df['date'].max().date()}")
    print(f"Testing period: {test_df['date'].min().date()} to {test_df['date'].max().date()}")

    # scale features and train logistic regression
    model=Pipeline([
        ("scaler",StandardScaler()),
        ("logistic",LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    model.fit(X_train,y_train)

    # select threshold using validation set
    validation_score=model.predict_proba(X_validation)[:,1]
    threshold_results=select_threshold(y_validation,validation_score)
    selected_threshold=threshold_results["threshold"]

    print("\nValidation Threshold Selection")
    print("-"*50)
    print(f"Selected threshold: {selected_threshold:.4f}")
    print(f"Validation precision: {threshold_results['precision']:.4f}")
    print(f"Validation recall: {threshold_results['recall']:.4f}")
    print(f"Validation F1: {threshold_results['f1']:.4f}")

    # evaluate on untouched test set
    test_score=model.predict_proba(X_test)[:,1]

    test_pred=evaluate_model(
        "Logistic Regression Test Results",
        y_test,
        test_score,
        selected_threshold
    )

    # show standardized feature coefficients
    coefficients=pd.DataFrame({
        "feature":features,
        "coefficient":model.named_steps["logistic"].coef_[0]
    }).sort_values("coefficient",ascending=False)

    print("\nFeature Coefficients")
    print("-"*50)
    print(coefficients.to_string(index=False))

    pred_df=pd.DataFrame({
        "date":test_df["date"].dt.date.values,
        "ticker":test_df["ticker"].values,
        "actual_risk_event":y_test.values,
        "risk_score":test_score,
        "risk_pred":test_pred,
        "model_name":"logistic_regression"
    })

    save_predictions(pred_df)

    model_artifact={
        "model":model,
        "features":features,
        "threshold":selected_threshold
    }

    os.makedirs("models",exist_ok=True)
    joblib.dump(model_artifact,"models/logistic_risk_model.joblib")

    print("\npredictions table created")
    print("model saved to models/logistic_risk_model.joblib")


if __name__=="__main__":
    main()