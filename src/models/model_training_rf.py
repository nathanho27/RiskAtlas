# model_training_rf.py
# trains random forest model to predict risk_event (future 10-day drop)

import os
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import psycopg2

from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

DB_NAME="risk_atlas"
DB_USER="nathanho"


def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host="localhost",
        port="5432"
    )


def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_USER}@localhost:5432/{DB_NAME}"
    )


def load_data():
    query="""
    SELECT *
    FROM model_dataset
    ORDER BY date,ticker
    """

    return pd.read_sql(query,get_engine())


def chronological_split(df,purge_periods=10):
    dates=np.sort(df["date"].unique())

    train_cutoff=int(len(dates)*0.70)
    validation_cutoff=int(len(dates)*0.85)

    train_dates=dates[:train_cutoff]
    validation_dates=dates[train_cutoff+purge_periods:validation_cutoff]
    test_dates=dates[validation_cutoff+purge_periods:]

    train_df=df[df["date"].isin(train_dates)].copy()
    validation_df=df[df["date"].isin(validation_dates)].copy()
    test_df=df[df["date"].isin(test_dates)].copy()

    return train_df,validation_df,test_df


def select_threshold(y_true,y_prob):
    precision,recall,thresholds=precision_recall_curve(y_true,y_prob)

    f1_scores=2*(precision[:-1]*recall[:-1])/(
        precision[:-1]+recall[:-1]+1e-10
    )

    best_index=np.argmax(f1_scores)

    return {
        "threshold":thresholds[best_index],
        "precision":precision[best_index],
        "recall":recall[best_index],
        "f1":f1_scores[best_index]
    }


def evaluate_model(y_true,y_prob,threshold):
    y_pred=(y_prob>=threshold).astype(int)

    print("\nRandom Forest Test Results")
    print("-"*50)
    print(f"Threshold: {threshold:.4f}")
    print(f"Risk-event prevalence: {y_true.mean():.4f}")

    print("\nClassification Report")
    print(classification_report(y_true,y_pred,digits=4))

    print("Confusion Matrix")
    print(confusion_matrix(y_true,y_pred))

    print(f"\nROC-AUC: {roc_auc_score(y_true,y_prob):.4f}")
    print(f"PR-AUC: {average_precision_score(y_true,y_prob):.4f}")

    return y_pred


def save_predictions(test_df,y_prob,y_pred):
    pred_df=pd.DataFrame({
        "date":test_df["date"].values,
        "ticker":test_df["ticker"].values,
        "actual_risk_event":test_df["risk_event"].astype(int).values,
        "risk_score":y_prob,
        "risk_pred":y_pred,
        "model_name":"random_forest"
    })

    conn=get_connection()
    cur=conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
            date DATE NOT NULL,
            ticker TEXT NOT NULL,
            actual_risk_event INT NOT NULL,
            risk_score DOUBLE PRECISION NOT NULL,
            risk_pred INT NOT NULL,
            model_name TEXT NOT NULL,
            PRIMARY KEY(date,ticker,model_name)
        );
    """)

    cur.execute("""
        DELETE FROM predictions
        WHERE model_name='random_forest';
    """)

    buffer=StringIO()
    pred_df.to_csv(buffer,index=False,header=False)
    buffer.seek(0)

    cur.copy_from(
        buffer,
        "predictions",
        columns=(
            "date",
            "ticker",
            "actual_risk_event",
            "risk_score",
            "risk_pred",
            "model_name"
        ),
        sep=","
    )

    conn.commit()
    cur.close()
    conn.close()

    print("\nrandom forest predictions saved")


def main():
    df=load_data()

    features=[
        "daily_return",
        "vol_20",
        "vol_30",
        "vol_60",
        "price_to_ma50",
        "price_to_ma200"
    ]

    required_columns=["date","ticker","risk_event"]+features
    df=df.dropna(subset=required_columns).copy()
    df["date"]=pd.to_datetime(df["date"])

    train_df,validation_df,test_df=chronological_split(df)

    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Testing rows: {len(test_df):,}")

    print(
        f"Training period: {train_df['date'].min().date()} "
        f"to {train_df['date'].max().date()}"
    )
    print(
        f"Validation period: {validation_df['date'].min().date()} "
        f"to {validation_df['date'].max().date()}"
    )
    print(
        f"Testing period: {test_df['date'].min().date()} "
        f"to {test_df['date'].max().date()}"
    )

    X_train=train_df[features]
    y_train=train_df["risk_event"].astype(int)

    X_validation=validation_df[features]
    y_validation=validation_df["risk_event"].astype(int)

    X_test=test_df[features]
    y_test=test_df["risk_event"].astype(int)

    model=RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train,y_train)

    validation_prob=model.predict_proba(X_validation)[:,1]
    threshold_results=select_threshold(y_validation,validation_prob)
    selected_threshold=threshold_results["threshold"]

    print("\nValidation Threshold Selection")
    print("-"*50)
    print(f"Selected threshold: {selected_threshold:.4f}")
    print(f"Validation precision: {threshold_results['precision']:.4f}")
    print(f"Validation recall: {threshold_results['recall']:.4f}")
    print(f"Validation F1: {threshold_results['f1']:.4f}")

    test_prob=model.predict_proba(X_test)[:,1]
    test_pred=evaluate_model(y_test,test_prob,selected_threshold)

    importance_df=pd.DataFrame({
        "feature":features,
        "importance":model.feature_importances_
    }).sort_values("importance",ascending=False)

    print("\nFeature Importances")
    print("-"*50)
    print(importance_df.to_string(index=False))

    os.makedirs("models",exist_ok=True)

    model_artifact={
        "model":model,
        "features":features,
        "threshold":selected_threshold
    }

    joblib.dump(
        model_artifact,
        "models/random_forest_risk_model.joblib"
    )

    save_predictions(test_df,test_prob,test_pred)

    print("model saved to models/random_forest_risk_model.joblib")


if __name__=="__main__":
    main()