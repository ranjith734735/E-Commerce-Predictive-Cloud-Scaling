from flask import Flask, jsonify, render_template, request
from pathlib import Path
from datetime import datetime

import json
import math
import pandas as pd

from model.predict import (
    analyze_prediction,
    get_sales_context,
    get_next_event,
    BASELINE_INSTANCES,
    TARGET_CPU,
)

# ============================================================
# E-COMMERCE PREDICTIVE AUTO-SCALING
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "utilization.csv"
)

EVENT_FILE = (
    BASE_DIR
    / "data"
    / "events.csv"
)

METRICS_FILE = (
    BASE_DIR
    / "model"
    / "model_metrics.json"
)


# ============================================================
# CLOUD SIMULATOR STATE
# ============================================================

CURRENT_INSTANCES = BASELINE_INSTANCES
PREPARED_INSTANCES = BASELINE_INSTANCES

SCALING_PLAN = None

LAST_PREPARATION_TIME = None


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(
        DATA_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    return df


def load_events():

    events = pd.read_csv(
        EVENT_FILE
    )

    events["start_date"] = pd.to_datetime(
        events["start_date"]
    )

    events["end_date"] = pd.to_datetime(
        events["end_date"]
    )

    return events


# ============================================================
# SIMULATION TIME
# ============================================================

def get_simulation_time():

    value = (
        request.args.get("simulation_time")
        or request.args.get("date")
    )

    if not value:

        json_data = request.get_json(
            silent=True
        )

        if json_data:

            value = (
                json_data.get("simulation_time")
                or json_data.get("date")
            )

    if value:

        try:

            return pd.Timestamp(
                value
            )

        except Exception:

            pass

    # Use the real computer date/time.
    now = pd.Timestamp.now()

    df = load_data()

    minimum_time = df["timestamp"].min()
    maximum_time = df["timestamp"].max()

    # If computer time is inside the project dataset,
    # use it automatically.
    if (
        minimum_time
        <= now
        <= maximum_time
    ):

        return now.floor("h")

    # Otherwise use the latest available project timestamp.
    return maximum_time


# ============================================================
# FIND CURRENT DATA ROW
# ============================================================

def get_current_row(
    simulation_time
):

    df = load_data()

    available = df[
        df["timestamp"]
        <= simulation_time
    ]

    if available.empty:

        row = df.iloc[0]

    else:

        row = available.iloc[-1]

    return row


# ============================================================
# DAILY SUMMARY
# ============================================================

def get_daily_summary(
    simulation_time
):

    df = load_data()

    simulation_date = (
        simulation_time.date()
    )

    today = df[
        df["timestamp"].dt.date
        == simulation_date
    ]

    # If today's rows are not available,
    # use the current day's nearest available data.
    if today.empty:

        return {

            "estimated_orders_today":
                0,

            "estimated_requests_today":
                0,

            "daily_peak_cpu":
                0,

            "data_hours":
                0
        }

    estimated_orders = (
        today["orders_per_min"].sum()
        * 60
    )

    estimated_requests = (
        today["requests_per_min"].sum()
        * 60
    )

    daily_peak_cpu = (
        today["cpu_usage"].max()
    )

    return {

        "estimated_orders_today":
            round(
                float(estimated_orders)
            ),

        "estimated_requests_today":
            round(
                float(estimated_requests)
            ),

        "daily_peak_cpu":
            round(
                float(daily_peak_cpu),
                2
            ),

        "data_hours":
            int(len(today))
    }


# ============================================================
# FORMAT NUMBER
# ============================================================

def clean_number(
    value,
    decimals=2
):

    try:

        return round(
            float(value),
            decimals
        )

    except Exception:

        return 0


# ============================================================
# CURRENT DASHBOARD STATUS
# ============================================================

def get_current_status():

    simulation_time = (
        get_simulation_time()
    )

    row = get_current_row(
        simulation_time
    )

    context = get_sales_context(
        simulation_time
    )

    next_event = get_next_event(
        simulation_time
    )

    daily = get_daily_summary(
        simulation_time
    )

    upcoming_event = None

    if next_event is not None:

        hours_until = (

            next_event["start_date"]
            - simulation_time

        ).total_seconds() / 3600

        upcoming_event = {

            "name":
                str(
                    next_event[
                        "event_name"
                    ]
                ),

            "type":
                str(
                    next_event[
                        "event_type"
                    ]
                ),

            "start":
                str(
                    next_event[
                        "start_date"
                    ]
                ),

            "end":
                str(
                    next_event[
                        "end_date"
                    ]
                ),

            "intensity":
                clean_number(
                    next_event[
                        "intensity"
                    ]
                ),

            "hours_until":
                round(
                    hours_until,
                    2
                )
        }

    return {

        "simulation_time":
            str(
                simulation_time
            ),

        "timestamp":
            str(
                row["timestamp"]
            ),

        "sales_type":
            str(
                context[
                    "sales_type"
                ]
            ),

        "event_name":
            str(
                context[
                    "event_name"
                ]
            ),

        "event_active":
            int(
                context[
                    "event_active"
                ]
            ),

        "event_announced":
            int(
                context[
                    "event_announced"
                ]
            ),

        "hours_until_event":
            context[
                "hours_until_event"
            ],

        "cpu_usage":
            clean_number(
                row[
                    "cpu_usage"
                ]
            ),

        "memory_usage":
            clean_number(
                row[
                    "memory_usage"
                ]
            ),

        "network_usage":
            clean_number(
                row[
                    "network_usage"
                ]
            ),

        "requests_per_min":
            clean_number(
                row[
                    "requests_per_min"
                ]
            ),

        "orders_per_min":
            clean_number(
                row[
                    "orders_per_min"
                ]
            ),

        "active_users":
            clean_number(
                row[
                    "active_users"
                ]
            ),

        "current_instances":
            CURRENT_INSTANCES,

        "prepared_instances":
            PREPARED_INSTANCES,

        "upcoming_event":
            upcoming_event,

        "daily_summary":
            daily
    }


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def get_model_metrics():

    if not METRICS_FILE.exists():

        return {

            "model":
                "Random Forest Regression",

            "mae":
                None,

            "rmse":
                None,

            "r2_score":
                None
        }

    try:

        with open(
            METRICS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

    except Exception:

        return {

            "model":
                "Random Forest Regression",

            "mae":
                None,

            "rmse":
                None,

            "r2_score":
                None
        }


# ============================================================
# AUTOMATIC FORECAST HORIZON
# ============================================================

def calculate_default_forecast_hours(
    simulation_time
):

    next_event = get_next_event(
        simulation_time
    )

    if next_event is None:

        return 24

    hours_until = (

        next_event["start_date"]
        - simulation_time

    ).total_seconds() / 3600

    # When an event is approaching within 14 days,
    # forecast far enough to cover it.
    if 0 < hours_until <= 336:

        event_end = (
            next_event["end_date"]
            + pd.Timedelta(hours=23)
        )

        hours = (

            event_end
            - simulation_time

        ).total_seconds() / 3600

        return int(
            max(
                24,
                min(
                    math.ceil(hours),
                    336
                )
            )
        )

    return 24


# ============================================================
# AUTOMATIC PREPARATION
# ============================================================

def automatic_prepare(
    analysis
):

    global PREPARED_INSTANCES
    global SCALING_PLAN
    global LAST_PREPARATION_TIME

    if not analysis[
        "prepare_scaling_now"
    ]:

        return False

    required = int(
        analysis[
            "required_instances"
        ]
    )

    if required <= BASELINE_INSTANCES:

        return False

    PREPARED_INSTANCES = required

    SCALING_PLAN = {

        "status":
            "Prepared",

        "prepared_at":
            str(
                analysis[
                    "simulation_time"
                ]
            ),

        "baseline_instances":
            BASELINE_INSTANCES,

        "prepared_instances":
            required,

        "predicted_peak_cpu":
            analysis[
                "peak_cpu"
            ],

        "peak_time":
            analysis[
                "peak_time"
            ],

        "event_name":
            analysis[
                "upcoming_event"
            ],

        "message":
            (
                "Cloud capacity has been "
                "prepared before the predicted "
                "sales workload peak."
            )
    }

    LAST_PREPARATION_TIME = (
        analysis[
            "simulation_time"
        ]
    )

    return True


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health():

    return jsonify({

        "success":
            True,

        "message":
            "E-Commerce Predictive Scaling API is running."
    })


# ============================================================
# CURRENT DATA API
# ============================================================

@app.route(
    "/api/data",
    methods=["GET", "POST"]
)
def api_data():

    try:

        return jsonify({

            "success":
                True,

            "data":
                get_current_status()
        })

    except Exception as error:

        return jsonify({

            "success":
                False,

            "error":
                str(error)
        }), 500


# ============================================================
# EVENTS API
# ============================================================

@app.route("/api/events")
def api_events():

    try:

        events = load_events()

        result = []

        for _, event in events.iterrows():

            result.append({

                "name":
                    str(
                        event[
                            "event_name"
                        ]
                    ),

                "type":
                    str(
                        event[
                            "event_type"
                        ]
                    ),

                "start":
                    str(
                        event[
                            "start_date"
                        ]
                    ),

                "end":
                    str(
                        event[
                            "end_date"
                        ]
                    ),

                "intensity":
                    clean_number(
                        event[
                            "intensity"
                        ]
                    )
            })

        return jsonify({

            "success":
                True,

            "events":
                result
        })

    except Exception as error:

        return jsonify({

            "success":
                False,

            "error":
                str(error)
        }), 500


# ============================================================
# MODEL PERFORMANCE API
# ============================================================

@app.route(
    "/api/model-performance"
)
def model_performance():

    return jsonify({

        "success":
            True,

        "metrics":
            get_model_metrics()
    })


# ============================================================
# FORECAST API
# ============================================================

@app.route(
    "/api/forecast",
    methods=["GET", "POST"]
)
def forecast():

    try:

        simulation_time = (
            get_simulation_time()
        )

        json_data = request.get_json(
            silent=True
        )

        hours_value = (
            request.args.get("hours")
        )

        if not hours_value and json_data:

            hours_value = json_data.get(
                "hours"
            )

        if hours_value:

            try:

                hours = int(
                    hours_value
                )

            except Exception:

                hours = (
                    calculate_default_forecast_hours(
                        simulation_time
                    )
                )

        else:

            hours = (
                calculate_default_forecast_hours(
                    simulation_time
                )
            )

        hours = max(
            1,
            min(
                hours,
                336
            )
        )

        analysis = analyze_prediction(

            simulation_time,

            hours
        )

        was_prepared = (
            automatic_prepare(
                analysis
            )
        )

        return jsonify({

            "success":
                True,

            "analysis":
                analysis,

            "automatic_preparation":
                was_prepared,

            "prepared_instances":
                PREPARED_INSTANCES,

            "scaling_plan":
                SCALING_PLAN
        })

    except Exception as error:

        return jsonify({

            "success":
                False,

            "error":
                str(error)
        }), 500


# ============================================================
# PREPARE SCALING API
# ============================================================

@app.route(
    "/api/prepare-scaling",
    methods=["GET", "POST"]
)
def prepare_scaling():

    try:

        simulation_time = (
            get_simulation_time()
        )

        hours = (
            calculate_default_forecast_hours(
                simulation_time
            )
        )

        analysis = analyze_prediction(
            simulation_time,
            hours
        )

        automatic_prepare(
            analysis
        )

        if SCALING_PLAN is None:

            return jsonify({

                "success":
                    False,

                "message":
                    (
                        "No additional cloud "
                        "capacity is currently "
                        "required."
                    ),

                "analysis":
                    analysis
            })

        return jsonify({

            "success":
                True,

            "message":
                "Predictive scaling prepared.",

            "plan":
                SCALING_PLAN
        })

    except Exception as error:

        return jsonify({

            "success":
                False,

            "error":
                str(error)
        }), 500


# ============================================================
# EXECUTE SCALING
# ============================================================

@app.route(
    "/api/execute-scaling",
    methods=["GET", "POST"]
)
def execute_scaling():

    global CURRENT_INSTANCES

    if SCALING_PLAN is None:

        return jsonify({

            "success":
                False,

            "message":
                "No prepared scaling plan is available."
        }), 400

    CURRENT_INSTANCES = int(
        SCALING_PLAN[
            "prepared_instances"
        ]
    )

    SCALING_PLAN["status"] = (
        "Executed"
    )

    SCALING_PLAN["executed_at"] = (
        str(
            get_simulation_time()
        )
    )

    SCALING_PLAN["message"] = (
        "Scaling action executed in the "
        "cloud simulation."
    )

    return jsonify({

        "success":
            True,

        "message":
            (
                "Scaling executed successfully."
            ),

        "current_instances":
            CURRENT_INSTANCES,

        "plan":
            SCALING_PLAN
    })


# ============================================================
# SCALE DOWN
# ============================================================

@app.route(
    "/api/scale-down",
    methods=["GET", "POST"]
)
def scale_down():

    global CURRENT_INSTANCES
    global PREPARED_INSTANCES
    global SCALING_PLAN
    global LAST_PREPARATION_TIME

    previous_instances = (
        CURRENT_INSTANCES
    )

    CURRENT_INSTANCES = (
        BASELINE_INSTANCES
    )

    PREPARED_INSTANCES = (
        BASELINE_INSTANCES
    )

    SCALING_PLAN = None

    LAST_PREPARATION_TIME = None

    return jsonify({

        "success":
            True,

        "message":
            (
                f"Cloud capacity returned "
                f"from {previous_instances} "
                f"to {BASELINE_INSTANCES} "
                f"instances."
            ),

        "current_instances":
            CURRENT_INSTANCES,

        "baseline_instances":
            BASELINE_INSTANCES
    })


# ============================================================
# SCALING STATUS
# ============================================================

@app.route(
    "/api/scaling-status"
)
def scaling_status():

    simulation_time = (
        get_simulation_time()
    )

    context = get_sales_context(
        simulation_time
    )

    next_event = get_next_event(
        simulation_time
    )

    return jsonify({

        "success":
            True,

        "simulation_time":
            str(
                simulation_time
            ),

        "current_instances":
            CURRENT_INSTANCES,

        "prepared_instances":
            PREPARED_INSTANCES,

        "baseline_instances":
            BASELINE_INSTANCES,

        "target_cpu":
            TARGET_CPU,

        "current_sales_type":
            context[
                "sales_type"
            ],

        "current_event":
            context[
                "event_name"
            ],

        "event_active":
            context[
                "event_active"
            ],

        "event_announced":
            context[
                "event_announced"
            ],

        "upcoming_event":
            (
                next_event[
                    "event_name"
                ]
                if next_event is not None
                else None
            ),

        "plan":
            SCALING_PLAN
    })


# ============================================================
# COMPATIBILITY API
# ============================================================

@app.route(
    "/api/apply-scaling",
    methods=["GET", "POST"]
)
def apply_scaling():

    response = prepare_scaling()

    if response.status_code != 200:

        return response

    data = response.get_json()

    if not data.get("success"):

        return response

    execute_response = (
        execute_scaling()
    )

    return execute_response


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "E-COMMERCE PREDICTIVE "
        "AUTO-SCALING"
    )
    print("=" * 70)
    print(
        "Dashboard : "
        "http://127.0.0.1:5000"
    )
    print(
        "API       : "
        "http://127.0.0.1:5000/api/health"
    )
    print("=" * 70)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )