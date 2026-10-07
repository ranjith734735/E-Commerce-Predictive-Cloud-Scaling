from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE=Path(__file__).resolve().parents[1]
MODEL_FILE=BASE/"model"/"workload_model.pkl"
FEATURES=["hour","day_of_week","day_of_year","is_weekend","event_factor",
          "active_users","requests_per_min","orders_per_min","memory_usage","network_usage"]
TARGET="cpu_usage"

def feature_frame(df):
    x=df.copy()
    ts=pd.to_datetime(x["timestamp"])
    x["hour"]=ts.dt.hour
    x["day_of_week"]=ts.dt.dayofweek
    x["day_of_year"]=ts.dt.dayofyear
    x["is_weekend"]=(ts.dt.dayofweek>=5).astype(int)
    return x[FEATURES]

def train_model(df):
    X=feature_frame(df)
    y=df[TARGET].astype(float)
    split=int(len(df)*0.8)
    model=RandomForestRegressor(n_estimators=250,max_depth=14,min_samples_leaf=2,random_state=42,n_jobs=-1)
    model.fit(X.iloc[:split],y.iloc[:split])
    pred=model.predict(X.iloc[split:])
    metrics={"mae":float(mean_absolute_error(y.iloc[split:],pred)),
             "rmse":float(np.sqrt(mean_squared_error(y.iloc[split:],pred))),
             "r2":float(r2_score(y.iloc[split:],pred)),
             "training_samples":split,"testing_samples":len(df)-split}
    joblib.dump(model,MODEL_FILE)
    return model,metrics

def load_or_train(df):
    if MODEL_FILE.exists():
        try:return joblib.load(MODEL_FILE)
        except Exception:pass
    model,_=train_model(df)
    return model

def predict_cpu(model, row):
    return float(np.clip(model.predict(feature_frame(pd.DataFrame([row])))[0],5,98))

def forecast_next(df, hours, event_factor=1.0, event_name="Normal Sales"):
    model=load_or_train(df)
    history=df.copy()
    last=pd.Timestamp(history.iloc[-1]["timestamp"])
    base=history.iloc[-1].to_dict()
    rows=[]
    for i in range(1,hours+1):
        ts=last+pd.Timedelta(hours=i)
        # use recent values adjusted for expected sales/event intensity
        recent=history.tail(24)
        scale=event_factor
        row={
          "timestamp":ts,
          "active_users":float(recent.active_users.mean())*scale,
          "requests_per_min":float(recent.requests_per_min.mean())*scale,
          "orders_per_min":float(recent.orders_per_min.mean())*scale,
          "memory_usage":min(95,float(recent.memory_usage.mean())*(0.88+0.12*scale)),
          "network_usage":min(98,float(recent.network_usage.mean())*scale),
          "event_factor":event_factor,
        }
        row["cpu_usage"]=predict_cpu(model,row)
        row["sales_type"]=event_name
        rows.append(row)
        history=pd.concat([history,pd.DataFrame([row])],ignore_index=True)
    return pd.DataFrame(rows)

def evaluate_model(df):
    model,metrics=train_model(df)
    return metrics
