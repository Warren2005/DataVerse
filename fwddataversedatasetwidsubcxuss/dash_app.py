# dash_app.py
# Interactive analytics dashboard for Brew & Beans datathon case
# - Lets judges pick date range, x-axis, metric, and filters
# - Shows dynamic plots + summary KPIs
# - Designed as an appendix / Q&A tool for your presentation

import pandas as pd
import numpy as np
from datetime import datetime

from dash import Dash, dcc, html, Input, Output
import plotly.express as px

# ============================================================
# 1. LOAD & PREP DATA  (SAME LOGIC AS YOUR NOTEBOOK)
# ============================================================

# Assumes sales.csv and items.csv are in the same folder
sales_df = pd.read_csv("sales.csv")
items_df = pd.read_csv("items.csv")

# Merge item info onto transactions
merged_df = pd.merge(sales_df, items_df, on="item_name", how="left")

# Parse datetime
merged_df["datetime"] = pd.to_datetime(
    merged_df["date"] + " " + merged_df["time"]
)

# Rename for clarity
merged_df = merged_df.rename(
    columns={"own_cup": "used_own_cup", "surcharge": "surcharge_applied"}
)

# Convert to boolean where applicable
merged_df["used_own_cup"] = merged_df["used_own_cup"].astype("boolean")
merged_df["surcharge_applied"] = merged_df["surcharge_applied"].astype("boolean")

# Time features
merged_df["date_only"] = merged_df["datetime"].dt.date
merged_df["year"] = merged_df["datetime"].dt.year
merged_df["month"] = merged_df["datetime"].dt.month
merged_df["hour"] = merged_df["datetime"].dt.hour
merged_df["weekday"] = merged_df["datetime"].dt.day_name()

# Policy / construction periods (same as earlier)
construction_start = pd.to_datetime("2022-09-04")
construction_end   = pd.to_datetime("2022-12-12")
policy_start       = pd.to_datetime("2023-11-01")

merged_df["period"] = np.where(
    (merged_df["datetime"] >= construction_start)
    & (merged_df["datetime"] <= construction_end),
    "construction",
    np.where(
        merged_df["datetime"] < policy_start,
        "pre_policy",
        "post_policy",
    ),
)

# Profit calculation (price - cost + surcharge)
merged_df["surcharge_amount"] = merged_df["surcharge_applied"] * 0.50
merged_df["profit"] = (
    merged_df["price"] - merged_df["production_cost"] + merged_df["surcharge_amount"]
)

# Define "loyal" customers (e.g., 10+ visits)
visits_per_cust = merged_df.groupby("customer_id").size()
loyal_customers = visits_per_cust[visits_per_cust >= 10].index
merged_df["loyal_segment"] = np.where(
    merged_df["customer_id"].isin(loyal_customers),
    "Loyal (10+ visits)",
    "Normal",
)

# For convenience, restrict to drinks for some metrics
drinks_df = merged_df[merged_df["item_type"] == "Drink"].copy()

# For date picker limits
min_date = merged_df["datetime"].min().date()
max_date = merged_df["datetime"].max().date()

# ============================================================
# 2. DASH APP SETUP
# ============================================================

app = Dash(__name__)
server = app.server  # in case you later deploy

# Options for x-axis and metrics
x_axis_options = [
    {"label": "Hour of Day", "value": "hour"},
    {"label": "Weekday", "value": "weekday"},
    {"label": "Month", "value": "month"},
    {"label": "Period (Pre / Construction / Post)", "value": "period"},
    {"label": "Drink Type", "value": "drink_type"},
    {"label": "Drink Temperature", "value": "drink_temperature"},
    {"label": "Transaction Type (Dine-in vs Takeout)", "value": "transaction_type"},
    {"label": "Loyal vs Normal Customers", "value": "loyal_segment"},
]

metric_options = [
    {"label": "Number of Transactions", "value": "count_txn"},
    {"label": "Reusable Cup Share", "value": "reuse_share"},
    {"label": "Average Profit per Transaction", "value": "avg_profit"},
    {"label": "Total Profit", "value": "total_profit"},
]

# ============================================================
# 3. LAYOUT
# ============================================================

app.layout = html.Div(
    style={"fontFamily": "Arial", "padding": "20px"},
    children=[
        html.H2("Brew & Beans – Interactive Analytics Dashboard"),
        html.P(
            "Use this dashboard in Q&A to answer judge questions live. "
            "Filter by date, choose a dimension for the x-axis, select a metric, "
            "and optionally focus on drinks only, loyal customers, periods, etc."
        ),

        # ---- Controls row ----
        html.Div(
            style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
            children=[
                # Date range
                html.Div(
                    children=[
                        html.Label("Date range"),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            start_date=min_date,
                            end_date=max_date,
                        ),
                    ]
                ),

                # X-axis
                html.Div(
                    children=[
                        html.Label("X-axis"),
                        dcc.Dropdown(
                            id="x-axis",
                            options=x_axis_options,
                            value="hour",
                            clearable=False,
                        ),
                    ],
                    style={"minWidth": "220px"},
                ),

                # Metric
                html.Div(
                    children=[
                        html.Label("Metric"),
                        dcc.Dropdown(
                            id="metric",
                            options=metric_options,
                            value="count_txn",
                            clearable=False,
                        ),
                    ],
                    style={"minWidth": "260px"},
                ),

                # Item type filter
                html.Div(
                    children=[
                        html.Label("Scope"),
                        dcc.RadioItems(
                            id="scope",
                            options=[
                                {"label": "All Items", "value": "all"},
                                {"label": "Drinks Only", "value": "drinks"},
                            ],
                            value="all",
                            inline=True,
                        ),
                    ]
                ),

                # Period filter
                html.Div(
                    children=[
                        html.Label("Periods"),
                        dcc.Checklist(
                            id="period-filter",
                            options=[
                                {"label": "Pre-policy", "value": "pre_policy"},
                                {"label": "Construction", "value": "construction"},
                                {"label": "Post-policy", "value": "post_policy"},
                            ],
                            value=["pre_policy", "construction", "post_policy"],
                            inline=True,
                        ),
                    ],
                    style={"maxWidth": "400px"},
                ),
            ],
        ),

        html.Hr(),

        # ---- KPIs + Plot ----
        html.Div(
            children=[
                html.Div(
                    id="kpi-cards",
                    style={
                        "display": "flex",
                        "gap": "40px",
                        "flexWrap": "wrap",
                        "marginBottom": "20px",
                    },
                ),
                dcc.Graph(id="main-graph"),
            ]
        ),
    ],
)

# ============================================================
# 4. CALLBACKS
# ============================================================

@app.callback(
    [Output("main-graph", "figure"), Output("kpi-cards", "children")],
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("x-axis", "value"),
        Input("metric", "value"),
        Input("scope", "value"),
        Input("period-filter", "value"),
    ],
)
def update_dashboard(start_date, end_date, x_axis, metric, scope, period_values):
    # ---- Filter by date ----
    mask = (merged_df["date_only"] >= pd.to_datetime(start_date).date()) & (
        merged_df["date_only"] <= pd.to_datetime(end_date).date()
    )
    df = merged_df.loc[mask].copy()

    # ---- Scope: drinks only or all items ----
    if scope == "drinks":
        df = df[df["item_type"] == "Drink"]

    # ---- Filter by selected periods ----
    if period_values:
        df = df[df["period"].isin(period_values)]

    if df.empty:
        fig = px.scatter(title="No data for the selected filters.")
        return fig, [html.Div("No data in this range.", style={"fontWeight": "bold"})]

    # ---- Aggregation logic based on chosen metric ----
    group_cols = [x_axis]

    if metric == "count_txn":
        agg_df = df.groupby(group_cols).size().reset_index(name="value")
        y_label = "Number of Transactions"

    elif metric == "reuse_share":
        # Only meaningful for drinks
        agg_df = df.groupby(group_cols)["used_own_cup"].mean().reset_index(name="value")
        y_label = "Reusable Cup Share"

    elif metric == "avg_profit":
        agg_df = df.groupby(group_cols)["profit"].mean().reset_index(name="value")
        y_label = "Average Profit per Transaction ($)"

    elif metric == "total_profit":
        agg_df = df.groupby(group_cols)["profit"].sum().reset_index(name="value")
        y_label = "Total Profit ($)"

    else:
        agg_df = df.groupby(group_cols).size().reset_index(name="value")
        y_label = "Value"

    # ---- Build figure (bar or line depending on x-axis) ----
    # If x is numeric-like (hour / month), use line; else bar
    if x_axis in ["hour", "month"]:
        fig = px.line(agg_df, x=x_axis, y="value", markers=True)
    else:
        fig = px.bar(agg_df, x=x_axis, y="value")

    fig.update_layout(
        title=f"{y_label} by {x_axis}",
        xaxis_title=x_axis,
        yaxis_title=y_label,
    )

    # ---- KPI cards (high-level summary for judges) ----
    total_txn = len(df)
    unique_customers = df["customer_id"].nunique()
    avg_profit_txn = df["profit"].mean()
    reuse_rate = df["used_own_cup"].mean()

    kpis = [
        html.Div(
            children=[
                html.Div("Total Transactions", style={"fontWeight": "bold"}),
                html.Div(f"{total_txn:,}"),
            ]
        ),
        html.Div(
            children=[
                html.Div("Unique Customers", style={"fontWeight": "bold"}),
                html.Div(f"{unique_customers:,}"),
            ]
        ),
        html.Div(
            children=[
                html.Div("Avg Profit / Transaction", style={"fontWeight": "bold"}),
                html.Div(f"${avg_profit_txn:0.2f}"),
            ]
        ),
        html.Div(
            children=[
                html.Div("Reusable Cup Share (Drinks)", style={"fontWeight": "bold"}),
                html.Div(f"{reuse_rate*100:0.1f}%"),
            ]
        ),
    ]

    return fig, kpis


# ============================================================
# 5. MAIN
# ============================================================
server = app.server

if __name__ == "__main__":
    app.run(debug=True)
