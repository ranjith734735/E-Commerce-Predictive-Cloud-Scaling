import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# AUTOMATIC E-COMMERCE WORKLOAD PREDICTION
# ============================================================
#
# This program:
#
# 1. Reads the event calendar
# 2. Detects the current sales type
# 3. Detects the next upcoming event
# 4. Predicts future CPU
# 5. Finds predicted peak CPU
# 6. Calculates required cloud instances
# 7. Decides whether to Scale Up / Maintain / Scale Down
#
# The prediction is based on the trained Random Forest model.
# ============================================================


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

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

MODEL_FILE = (
    BASE_DIR
    / "model"
    / "ecommerce_cpu_model.pkl"
)

FEATURE_FILE = (
    BASE_DIR
    / "model"
    / "feature_columns.json"
)


# ------------------------------------------------------------
# Cloud configuration
# ------------------------------------------------------------

BASELINE_INSTANCES = 5

TARGET_CPU = 70.0

MIN_INSTANCES = 1

MAX_INSTANCES = 20

ANNOUNCEMENT_WINDOW_HOURS = 168


# ------------------------------------------------------------
# Sales type codes
# ------------------------------------------------------------

SALES_TYPE_CODES = {

    "Normal Sales": 0,

    "Weekend Sales": 1,

    "Summer Sale": 2,

    "Mega Sale": 3,

    "Flash Sale": 4,

    "Diwali Sale": 5,

    "Black Friday": 6,

    "Christmas Sale": 7,

    "New Year Sale": 8
}


# ------------------------------------------------------------
# Validate files
# ------------------------------------------------------------

required_files = [

    DATA_FILE,

    EVENT_FILE,

    MODEL_FILE,

    FEATURE_FILE
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Load events
# ------------------------------------------------------------

events = pd.read_csv(
    EVENT_FILE
)

events["start_date"] = pd.to_datetime(
    events["start_date"]
)

events["end_date"] = pd.to_datetime(
    events["end_date"]
)

events["intensity"] = pd.to_numeric(
    events["intensity"],
    errors="coerce"
)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model = joblib.load(
    MODEL_FILE
)


# ------------------------------------------------------------
# Load feature list
# ------------------------------------------------------------

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as file:

    FEATURES = json.load(
        file
    )


# ------------------------------------------------------------
# Event end timestamp
# ------------------------------------------------------------

def event_end_timestamp(event):

    return (
        event["end_date"]
        + pd.Timedelta(hours=23)
    )


# ------------------------------------------------------------
# Get active special event
# ------------------------------------------------------------

def get_active_event(timestamp):

    matching = events[

        (events["start_date"] <= timestamp)

        & (

            events.apply(

                lambda row:
                event_end_timestamp(row)
                >= timestamp,

                axis=1
            )
        )
    ]

    if matching.empty:

        return None

    matching = matching.sort_values(
        "intensity",
        ascending=False
    )

    return matching.iloc[0]


# ------------------------------------------------------------
# Get upcoming special event
# ------------------------------------------------------------

def get_next_event(timestamp):

    future_events = events[

        events["start_date"]
        > timestamp

    ]

    if future_events.empty:

        return None

    future_events = future_events.sort_values(
        "start_date"
    )

    return future_events.iloc[0]


# ------------------------------------------------------------
# Calculate event duration
# ------------------------------------------------------------

def calculate_event_duration(event):

    if event is None:

        return 0

    days = (
        event["end_date"]
        - event["start_date"]
    ).days + 1

    return int(
        days * 24
    )


# ------------------------------------------------------------
# Get current sales context
# ------------------------------------------------------------

def get_sales_context(timestamp):

    timestamp = pd.Timestamp(
        timestamp
    )


    # --------------------------------------------------------
    # Check active event
    # --------------------------------------------------------

    active_event = get_active_event(
        timestamp
    )


    # --------------------------------------------------------
    # Check upcoming event
    # --------------------------------------------------------

    next_event = get_next_event(
        timestamp
    )


    event_active = 0

    event_announced = 0

    hours_until_event = -1

    event_duration_hours = 0

    event_intensity = 1.0

    upcoming_event_name = ""

    current_event_name = ""


    # --------------------------------------------------------
    # Active event
    # --------------------------------------------------------

    if active_event is not None:

        event_active = 1

        event_announced = 0

        hours_until_event = 0

        event_intensity = float(
            active_event["intensity"]
        )

        event_duration_hours = (
            calculate_event_duration(
                active_event
            )
        )

        current_event_name = (
            active_event["event_name"]
        )

        return {

            "sales_type":
                active_event["event_type"],

            "event_name":
                active_event["event_name"],

            "event_active":
                event_active,

            "event_announced":
                event_announced,

            "hours_until_event":
                hours_until_event,

            "event_duration_hours":
                event_duration_hours,

            "event_intensity":
                event_intensity,

            "upcoming_event":
                active_event["event_name"]
        }


    # --------------------------------------------------------
    # Upcoming event
    # --------------------------------------------------------

    if next_event is not None:

        hours_to_event = (

            next_event["start_date"]
            - timestamp

        ).total_seconds() / 3600


        if (
            0 < hours_to_event
            <= ANNOUNCEMENT_WINDOW_HOURS
        ):

            event_announced = 1

            hours_until_event = round(
                hours_to_event,
                2
            )

            event_intensity = float(
                next_event["intensity"]
            )

            event_duration_hours = (
                calculate_event_duration(
                    next_event
                )
            )

            upcoming_event_name = (
                next_event["event_name"]
            )


    # --------------------------------------------------------
    # Weekend
    # --------------------------------------------------------

    if timestamp.dayofweek >= 5:

        sales_type = "Weekend Sales"

        event_name = "Weekend Sales"

        if event_announced:

            event_name = (
                "Upcoming "
                + upcoming_event_name
            )

        return {

            "sales_type":
                sales_type,

            "event_name":
                event_name,

            "event_active":
                event_active,

            "event_announced":
                event_announced,

            "hours_until_event":
                hours_until_event,

            "event_duration_hours":
                event_duration_hours,

            "event_intensity":
                event_intensity,

            "upcoming_event":
                upcoming_event_name
        }


    # --------------------------------------------------------
    # Normal Sales
    # --------------------------------------------------------

    event_name = "Normal Sales"

    if event_announced:

        event_name = (
            "Upcoming "
            + upcoming_event_name
        )

    return {

        "sales_type":
            "Normal Sales",

        "event_name":
            event_name,

        "event_active":
            0,

        "event_announced":
            event_announced,

        "hours_until_event":
            hours_until_event,

        "event_duration_hours":
            event_duration_hours,

        "event_intensity":
            event_intensity,

        "upcoming_event":
            upcoming_event_name
    }


# ------------------------------------------------------------
# Historical reference metrics
# ------------------------------------------------------------

def get_reference_metrics(
    timestamp,
    sales_type
):

    timestamp = pd.Timestamp(
        timestamp
    )


    # --------------------------------------------------------
    # Match same sales type and same hour
    # --------------------------------------------------------

    reference = df[

        (df["sales_type"] == sales_type)

        & (

            df["timestamp"].dt.hour
            == timestamp.hour

        )
    ]


    # --------------------------------------------------------
    # Fallback to sales type
    # --------------------------------------------------------

    if reference.empty:

        reference = df[
            df["sales_type"]
            == sales_type
        ]


    # --------------------------------------------------------
    # Fallback to normal same hour
    # --------------------------------------------------------

    if reference.empty:

        reference = df[

            (df["sales_type"]
             == "Normal Sales")

            & (

                df["timestamp"].dt.hour
                == timestamp.hour

            )
        ]


    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    if reference.empty:

        reference = df


    return {

        "active_users":
            float(
                reference[
                    "active_users"
                ].median()
            ),

        "requests_per_min":
            float(
                reference[
                    "requests_per_min"
                ].median()
            ),

        "orders_per_min":
            float(
                reference[
                    "orders_per_min"
                ].median()
            ),

        "memory_usage":
            float(
                reference[
                    "memory_usage"
                ].median()
            ),

        "network_usage":
            float(
                reference[
                    "network_usage"
                ].median()
            )
    }


# ------------------------------------------------------------
# Build ML input row
# ------------------------------------------------------------

def build_feature_row(
    timestamp,
    previous_cpu
):

    context = get_sales_context(
        timestamp
    )

    sales_type = context[
        "sales_type"
    ]

    metrics = get_reference_metrics(
        timestamp,
        sales_type
    )


    row = {

        "hour":
            timestamp.hour,

        "day_of_week":
            timestamp.dayofweek,

        "day_of_month":
            timestamp.day,

        "month":
            timestamp.month,

        "is_weekend":
            int(
                timestamp.dayofweek >= 5
            ),

        "active_users":
            metrics[
                "active_users"
            ],

        "requests_per_min":
            metrics[
                "requests_per_min"
            ],

        "orders_per_min":
            metrics[
                "orders_per_min"
            ],

        "cpu_usage":
            previous_cpu,

        "memory_usage":
            metrics[
                "memory_usage"
            ],

        "network_usage":
            metrics[
                "network_usage"
            ],

        "sales_type_code":
            SALES_TYPE_CODES.get(
                sales_type,
                0
            ),

        "event_active":
            context[
                "event_active"
            ],

        "event_announced":
            context[
                "event_announced"
            ],

        "hours_until_event":
            context[
                "hours_until_event"
            ],

        "event_duration_hours":
            context[
                "event_duration_hours"
            ],

        "event_intensity":
            context[
                "event_intensity"
            ]
    }


    values = [

        row[feature]

        for feature in FEATURES
    ]


    return pd.DataFrame(
        [values],
        columns=FEATURES
    )


# ------------------------------------------------------------
# Get starting CPU
# ------------------------------------------------------------

def get_starting_cpu(
    simulation_time
):

    simulation_time = pd.Timestamp(
        simulation_time
    )


    historical_rows = df[

        df["timestamp"]
        <= simulation_time

    ]


    if historical_rows.empty:

        return float(
            df.iloc[0]["cpu_usage"]
        )


    return float(
        historical_rows.iloc[-1][
            "cpu_usage"
        ]
    )


# ------------------------------------------------------------
# Predict future CPU
# ------------------------------------------------------------

def predict_future_cpu(
    simulation_time,
    periods=24
):

    simulation_time = pd.Timestamp(
        simulation_time
    )


    periods = max(
        1,
        int(periods)
    )


    current_time = (
        simulation_time
    )


    current_cpu = get_starting_cpu(
        simulation_time
    )


    forecasts = []


    for _ in range(periods):

        feature_row = build_feature_row(

            current_time,

            current_cpu

        )


        predicted_cpu = float(
            model.predict(
                feature_row
            )[0]
        )


        predicted_cpu = float(
            np.clip(
                predicted_cpu,
                5,
                98
            )
        )


        prediction_time = (
            current_time
            + pd.Timedelta(hours=1)
        )


        context = get_sales_context(
            current_time
        )


        forecasts.append({

            "timestamp":
                str(prediction_time),

            "predicted_cpu":
                round(
                    predicted_cpu,
                    2
                ),

            "sales_type":
                context[
                    "sales_type"
                ],

            "event_name":
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

            "hours_until_event":
                context[
                    "hours_until_event"
                ],

            "event_intensity":
                context[
                    "event_intensity"
                ]
        })


        current_cpu = (
            predicted_cpu
        )

        current_time = (
            prediction_time
        )


    return pd.DataFrame(
        forecasts
    )


# ------------------------------------------------------------
# Calculate required cloud instances
# ------------------------------------------------------------

def calculate_required_instances(
    predicted_peak_cpu
):

    predicted_peak_cpu = float(
        predicted_peak_cpu
    )


    required = math.ceil(

        BASELINE_INSTANCES
        * predicted_peak_cpu
        / TARGET_CPU

    )


    return int(
        np.clip(
            required,
            MIN_INSTANCES,
            MAX_INSTANCES
        )
    )


# ------------------------------------------------------------
# Determine scaling action
# ------------------------------------------------------------

def determine_scaling_action(
    required_instances
):

    if required_instances > BASELINE_INSTANCES:

        return "Scale Up"

    if required_instances < BASELINE_INSTANCES:

        return "Scale Down"

    return "Maintain"


# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------

def analyze_prediction(
    simulation_time,
    periods=24
):

    simulation_time = pd.Timestamp(
        simulation_time
    )


    forecast = predict_future_cpu(

        simulation_time,

        periods

    )


    # --------------------------------------------------------
    # Peak prediction
    # --------------------------------------------------------

    peak_index = (
        forecast[
            "predicted_cpu"
        ].idxmax()
    )


    peak_row = (
        forecast.loc[
            peak_index
        ]
    )


    peak_cpu = float(
        peak_row[
            "predicted_cpu"
        ]
    )


    peak_time = (
        peak_row[
            "timestamp"
        ]
    )


    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    required_instances = (
        calculate_required_instances(
            peak_cpu
        )
    )


    scaling_action = (
        determine_scaling_action(
            required_instances
        )
    )


    # --------------------------------------------------------
    # Current context
    # --------------------------------------------------------

    current_context = (
        get_sales_context(
            simulation_time
        )
    )


    # --------------------------------------------------------
    # Next event
    # --------------------------------------------------------

    next_event = get_next_event(
        simulation_time
    )


    next_event_start = None

    hours_until_event = None


    if next_event is not None:

        next_event_start = str(
            next_event["start_date"]
        )

        hours_until_event = (

            next_event["start_date"]
            - simulation_time

        ).total_seconds() / 3600


        hours_until_event = round(
            hours_until_event,
            2
        )


    # --------------------------------------------------------
    # Decide whether preparation should happen now
    # --------------------------------------------------------

    prepare_scaling_now = (

        next_event is not None

        and hours_until_event is not None

        and 0 < hours_until_event
        <= ANNOUNCEMENT_WINDOW_HOURS

        and required_instances
        > BASELINE_INSTANCES
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {

        "simulation_time":
            str(simulation_time),

        "current_sales_type":
            current_context[
                "sales_type"
            ],

        "current_event":
            current_context[
                "event_name"
            ],

        "current_event_active":
            current_context[
                "event_active"
            ],

        "upcoming_event":
            (
                next_event["event_name"]
                if next_event is not None
                else None
            ),

        "upcoming_event_type":
            (
                next_event["event_type"]
                if next_event is not None
                else None
            ),

        "upcoming_event_start":
            next_event_start,

        "hours_until_upcoming_event":
            hours_until_event,

        "peak_cpu":
            round(
                peak_cpu,
                2
            ),

        "peak_time":
            peak_time,

        "baseline_instances":
            BASELINE_INSTANCES,

        "required_instances":
            required_instances,

        "scaling_action":
            scaling_action,

        "prepare_scaling_now":
            prepare_scaling_now,

        "forecast":
            forecast.to_dict(
                orient="records"
            )
    }


# ------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Automatic e-commerce workload "
            "prediction and predictive "
            "cloud scaling analysis."
        )
    )


    parser.add_argument(

        "--date",

        default=(
            "2026-10-07 12:00:00"
        ),

        help=(
            "Simulation date/time. "
            "Format: YYYY-MM-DD HH:MM:SS"
        )
    )


    parser.add_argument(

        "--hours",

        type=int,

        default=24,

        help=(
            "Number of future hours "
            "to predict."
        )
    )


    args = parser.parse_args()


    simulation_time = pd.Timestamp(
        args.date
    )


    result = analyze_prediction(

        simulation_time,

        args.hours
    )


    print()
    print("=" * 75)
    print(
        "AUTOMATIC E-COMMERCE "
        "WORKLOAD PREDICTION"
    )
    print("=" * 75)


    print()

    print(
        f"Simulation Time          : "
        f"{result['simulation_time']}"
    )

    print(
        f"Current Sales Type        : "
        f"{result['current_sales_type']}"
    )

    print(
        f"Current Event             : "
        f"{result['current_event']}"
    )

    print(
        f"Upcoming Event            : "
        f"{result['upcoming_event'] or 'None'}"
    )

    print(
        f"Event Start               : "
        f"{result['upcoming_event_start'] or 'N/A'}"
    )


    if (
        result[
            "hours_until_upcoming_event"
        ] is not None
    ):

        print(
            f"Hours Until Event         : "
            f"{result['hours_until_upcoming_event']}"
        )


    print()

    print(
        f"Predicted Peak CPU        : "
        f"{result['peak_cpu']}%"
    )

    print(
        f"Predicted Peak Time       : "
        f"{result['peak_time']}"
    )

    print()

    print(
        f"Baseline Instances        : "
        f"{result['baseline_instances']}"
    )

    print(
        f"Required Instances        : "
        f"{result['required_instances']}"
    )

    print(
        f"Scaling Action            : "
        f"{result['scaling_action']}"
    )

    print(
        f"Prepare Scaling Now       : "
        f"{result['prepare_scaling_now']}"
    )


    print()
    print("-" * 75)
    print("FORECAST SAMPLE")
    print("-" * 75)


    display_columns = [

        "timestamp",

        "predicted_cpu",

        "sales_type",

        "event_name",

        "event_active",

        "event_announced"
    ]


    forecast_display = pd.DataFrame(
        result["forecast"]
    )


    print(
        forecast_display[
            display_columns
        ]
        .head(15)
        .to_string(
            index=False
        )
    )


    print()
    print("=" * 75)
    print()


if __name__ == "__main__":

    main()