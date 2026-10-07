from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
EVENT_FILE = BASE / "data" / "event_calendar.csv"

def load_event_calendar():
    return pd.read_csv(EVENT_FILE, parse_dates=["start_date", "end_date"])

def _weekend_event(ts):
    if ts.weekday() >= 5:
        return {
            "event_name": "Weekend Sales",
            "event_type": "Recurring",
            "start_date": ts.normalize(),
            "end_date": ts.normalize(),
            "intensity": 1.12,
        }
    return None

def current_event(ts):
    """Return the strongest event active at timestamp."""
    events = load_event_calendar()
    active=[]
    for _, e in events.iterrows():
        if e.start_date <= ts.normalize() <= e.end_date:
            active.append(e.to_dict())
    weekend=_weekend_event(ts)
    if weekend:
        active.append(weekend)
    if not active:
        return {"event_name":"Normal Sales","event_type":"Normal","intensity":1.0,
                "start_date":ts.normalize(),"end_date":ts.normalize()}
    return max(active, key=lambda x: float(x.get("intensity",1.0)))

def next_event(ts):
    """Find next dated event, considering recurring weekends."""
    events=load_event_calendar()
    candidates=[]
    for _,e in events.iterrows():
        start=pd.Timestamp(e.start_date)
        end=pd.Timestamp(e.end_date)
        if start > ts.normalize():
            candidates.append(e.to_dict())
    # next Saturday
    days_ahead=(5-ts.weekday())%7
    if days_ahead==0 and ts.hour>=23:
        days_ahead=7
    sat=(ts.normalize()+pd.Timedelta(days=days_ahead)).to_pydatetime()
    candidates.append({"event_name":"Weekend Sales","event_type":"Recurring",
                       "start_date":sat,"end_date":sat+timedelta(days=1),"intensity":1.12})
    return min(candidates, key=lambda x: pd.Timestamp(x["start_date"])) if candidates else None

def event_status(now=None):
    now=pd.Timestamp(now or datetime.now())
    active=current_event(now)
    nxt=next_event(now)
    if active["event_name"] != "Normal Sales":
        return {
            "mode":"active","event":active,"next_event":nxt,
            "hours_to_start":0,
            "days_to_start":0,
        }
    start=pd.Timestamp(nxt["start_date"])
    delta=start-now
    return {
        "mode":"upcoming","event":nxt,"next_event":nxt,
        "hours_to_start":max(0,round(delta.total_seconds()/3600,1)),
        "days_to_start":max(0,delta.total_seconds()/86400),
    }
