import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# E-COMMERCE PREDICTIVE AUTO-SCALING PROJECT
# DATASET GENERATOR
# ============================================================
#
# This program generates hourly e-commerce workload data.
#
# The system automatically identifies:
#
#   Normal Sales
#   Weekend Sales
#   Summer Sale
#   Mega Sale
#   Flash Sale
#   Diwali Sale
#   Black Friday
#   Christmas Sale
#   New Year Sale
#
# Event dates are stored in events.csv.
#
# NOTE:
# The event dates in events.csv are DEMO/SIMULATION dates
# for the college project. They are not intended to represent
# an official real-world sales calendar.
# ============================================================


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

np.random.seed(42)

DATA_DIR = Path(__file__).resolve().parent

EVENT_FILE = DATA_DIR / "events.csv"

OUTPUT_FILE = DATA_DIR / "utilization.csv"

START_DATE = "2026-01-01 00:00:00"

END_DATE = "2026-12-31 23:00:00"

ANNOUNCEMENT_WINDOW_HOURS = 168


# ------------------------------------------------------------
# Check event calendar
# ------------------------------------------------------------

if not EVENT_FILE.exists():
    raise FileNotFoundError(
        f"Event calendar not found:\n{EVENT_FILE}\n\n"
        "Create data\\events.csv before running this program."
    )


# ------------------------------------------------------------
# Load event calendar
# ------------------------------------------------------------

events = pd.read_csv(EVENT_FILE)

required_event_columns = [
    "event_name",
    "event_type",
    "start_date",
    "end_date",
    "intensity"
]

missing_event_columns = [
    column
    for column in required_event_columns
    if column not in events.columns
]

if missing_event_columns:
    raise ValueError(
        "events.csv is missing columns: "
        + ", ".join(missing_event_columns)
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

if events["intensity"].isna().any():
    raise ValueError(
        "Every event must have a valid numeric intensity."
    )


# ------------------------------------------------------------
# Create hourly timestamps
# ------------------------------------------------------------

timestamps = pd.date_range(
    start=START_DATE,
    end=END_DATE,
    freq="1h"
)


# ------------------------------------------------------------
# Normal daily shopping pattern
# ------------------------------------------------------------

def daily_factor(hour):
    """
    Returns the normal traffic multiplier for an hour.
    """

    if 0 <= hour < 6:
        return 0.45

    if 6 <= hour < 9:
        return 0.65

    if 9 <= hour < 13:
        return 0.85

    if 13 <= hour < 17:
        return 0.95

    if 17 <= hour < 21:
        return 1.35

    return 1.05


# ------------------------------------------------------------
# Get active event
# ------------------------------------------------------------

def get_active_event(timestamp):
    """
    Returns the event currently active at the given timestamp.

    The event is considered active from its start date at 00:00
    until its end date at 23:00.
    """

    matching = events[
        (events["start_date"] <= timestamp)
        & (
            events["end_date"]
            + pd.Timedelta(hours=23)
            >= timestamp
        )
    ]

    if matching.empty:
        return None

    # Highest intensity event gets priority
    matching = matching.sort_values(
        "intensity",
        ascending=False
    )

    return matching.iloc[0]


# ------------------------------------------------------------
# Get next upcoming event
# ------------------------------------------------------------

def get_upcoming_event(timestamp):
    """
    Returns the next scheduled special event after timestamp.
    """

    future_events = events[
        events["start_date"] > timestamp
    ]

    if future_events.empty:
        return None

    future_events = future_events.sort_values(
        "start_date"
    )

    return future_events.iloc[0]


# ------------------------------------------------------------
# Determine sales type
# ------------------------------------------------------------

def determine_sales_type(timestamp):
    """
    Automatically identifies the current sales type.

    Priority:

        1. Special event
        2. Weekend
        3. Normal Sales
    """

    active_event = get_active_event(
        timestamp
    )

    if active_event is not None:

        return (
            active_event["event_name"],
            active_event["event_type"],
            float(active_event["intensity"]),
            True
        )

    if timestamp.dayofweek >= 5:

        return (
            "Weekend Sales",
            "Weekend Sales",
            1.15,
            False
        )

    return (
        "Normal Sales",
        "Normal Sales",
        1.00,
        False
    )


# ------------------------------------------------------------
# Generate records
# ------------------------------------------------------------

records = []


for timestamp in timestamps:

    # --------------------------------------------------------
    # Current event
    # --------------------------------------------------------

    active_event = get_active_event(
        timestamp
    )

    (
        sales_name,
        sales_type,
        event_intensity,
        event_active
    ) = determine_sales_type(timestamp)


    # --------------------------------------------------------
    # Time information
    # --------------------------------------------------------

    hour = timestamp.hour

    day_of_week = timestamp.dayofweek

    days_from_start = (
        timestamp - timestamps[0]
    ).days


    # --------------------------------------------------------
    # Normal traffic factors
    # --------------------------------------------------------

    normal_daily_factor = daily_factor(
        hour
    )

    weekend_factor = (
        1.15
        if day_of_week >= 5
        else 1.00
    )


    # --------------------------------------------------------
    # Long-term growth
    # --------------------------------------------------------

    growth_factor = (
        1.0
        + (days_from_start / 365.0) * 0.20
    )


    # --------------------------------------------------------
    # Total workload factor
    # --------------------------------------------------------

    traffic_factor = (
        normal_daily_factor
        * weekend_factor
        * growth_factor
        * event_intensity
    )


    # --------------------------------------------------------
    # Random variation
    # --------------------------------------------------------

    random_factor = np.random.normal(
        1.0,
        0.05
    )


    # --------------------------------------------------------
    # Active users
    # --------------------------------------------------------

    active_users = (
        800
        * traffic_factor
        * random_factor
    )

    active_users = max(
        active_users,
        100
    )


    # --------------------------------------------------------
    # Requests per minute
    # --------------------------------------------------------

    requests_per_min = (
        active_users * 0.62
        + np.random.normal(0, 25)
    )

    requests_per_min = max(
        requests_per_min,
        50
    )


    # --------------------------------------------------------
    # Orders per minute
    # --------------------------------------------------------

    orders_per_min = (
        requests_per_min * 0.055
        + np.random.normal(0, 2)
    )

    orders_per_min = max(
        orders_per_min,
        1
    )


    # --------------------------------------------------------
    # CPU utilization
    # --------------------------------------------------------

    cpu_usage = (
        18
        + requests_per_min * 0.045
        + active_users * 0.008
        + np.random.normal(0, 2.5)
    )

    cpu_usage = np.clip(
        cpu_usage,
        10,
        98
    )


    # --------------------------------------------------------
    # Memory utilization
    # --------------------------------------------------------

    memory_usage = (
        24
        + active_users * 0.017
        + np.random.normal(0, 2)
    )

    memory_usage = np.clip(
        memory_usage,
        15,
        95
    )


    # --------------------------------------------------------
    # Network utilization
    # --------------------------------------------------------

    network_usage = (
        12
        + requests_per_min * 0.025
        + np.random.normal(0, 2)
    )

    network_usage = np.clip(
        network_usage,
        5,
        98
    )


    # --------------------------------------------------------
    # Upcoming event information
    # --------------------------------------------------------

    upcoming_event = get_upcoming_event(
        timestamp
    )

    event_announced = False

    hours_until_event = -1

    upcoming_event_name = ""

    upcoming_event_intensity = 0.0

    upcoming_event_duration_hours = 0


    if upcoming_event is not None:

        delta_hours = (
            upcoming_event["start_date"]
            - timestamp
        ).total_seconds() / 3600


        # ----------------------------------------------------
        # Announcement window
        # ----------------------------------------------------

        if (
            0 < delta_hours
            <= ANNOUNCEMENT_WINDOW_HOURS
        ):

            event_announced = True

            hours_until_event = round(
                delta_hours,
                2
            )

            upcoming_event_name = (
                upcoming_event["event_name"]
            )

            upcoming_event_intensity = float(
                upcoming_event["intensity"]
            )

            upcoming_event_duration_hours = int(
                (
                    upcoming_event["end_date"]
                    - upcoming_event["start_date"]
                ).days + 1
            ) * 24


    # --------------------------------------------------------
    # Give the ML model information about the future event
    # --------------------------------------------------------

    if (
        not event_active
        and event_announced
    ):

        model_event_intensity = (
            upcoming_event_intensity
        )

        model_event_duration = (
            upcoming_event_duration_hours
        )

    else:

        model_event_intensity = (
            float(event_intensity)
            if event_active
            else 1.0
        )

        model_event_duration = 0


    # --------------------------------------------------------
    # Current event duration
    # --------------------------------------------------------

    current_event_duration_hours = 0

    if active_event is not None:

        current_event_duration_hours = int(
            (
                active_event["end_date"]
                - active_event["start_date"]
            ).days + 1
        ) * 24


    # --------------------------------------------------------
    # Store record
    # --------------------------------------------------------

    records.append({

        "timestamp":
            timestamp,

        "active_users":
            round(active_users, 2),

        "requests_per_min":
            round(requests_per_min, 2),

        "orders_per_min":
            round(orders_per_min, 2),

        "cpu_usage":
            round(cpu_usage, 2),

        "memory_usage":
            round(memory_usage, 2),

        "network_usage":
            round(network_usage, 2),

        "sales_type":
            sales_type,

        "event_name":
            sales_name,

        "event_active":
            int(event_active),

        "event_announced":
            int(event_announced),

        "hours_until_event":
            hours_until_event,

        "event_duration_hours":
            model_event_duration,

        "event_intensity":
            round(
                model_event_intensity,
                2
            ),

        "upcoming_event":
            upcoming_event_name
    })


# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------

df = pd.DataFrame(
    records
)


# ------------------------------------------------------------
# Check missing values
# ------------------------------------------------------------

if df.isnull().sum().sum() > 0:

    print(
        "Warning: missing values detected."
    )

    print(
        df.isnull().sum()
    )


# ------------------------------------------------------------
# Save dataset
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# Display result
# ------------------------------------------------------------

print()
print("=" * 70)
print("E-COMMERCE DATASET CREATED")
print("=" * 70)

print(
    f"Records       : {len(df)}"
)

print(
    f"Start         : {df['timestamp'].min()}"
)

print(
    f"End           : {df['timestamp'].max()}"
)

print()
print("Sales Type Distribution:")

print(
    df["sales_type"].value_counts()
)

print()
print("Event Distribution:")

print(
    df["event_name"].value_counts()
)

print()
print("Dataset Columns:")

for column in df.columns:

    print(
        f"  - {column}"
    )

print()
print("=" * 70)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print("=" * 70)
print()