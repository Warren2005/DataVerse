"""
analysis_and_plots.py

Loads `sales.csv` and `items.csv`, cleans/merges them, computes revenue and profit per sale,
and produces interactive Plotly HTML visualizations saved to `outputs/`.

Run:
  python3 analysis_and_plots.py

Requirements: see requirements.txt
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path('/Users/wld/Downloads/fwddataversedatasetwidsubcxuss')
SALES_CSV = ROOT / 'sales.csv'
ITEMS_CSV = ROOT / 'items.csv'
OUT_DIR = ROOT / 'outputs'
OUT_DIR.mkdir(exist_ok=True)

print('Reading', SALES_CSV)
print('Reading', ITEMS_CSV)

df_sales = pd.read_csv(SALES_CSV, parse_dates=False)
df_items = pd.read_csv(ITEMS_CSV)

# Quick harmonization
# Some files may contain BOM or stray whitespace in column names; standardize
df_sales.columns = df_sales.columns.str.strip()
df_items.columns = df_items.columns.str.strip()

# Combine date + time into single datetime
if 'date' in df_sales.columns and 'time' in df_sales.columns:
    df_sales['datetime'] = pd.to_datetime(df_sales['date'].astype(str) + ' ' + df_sales['time'].astype(str), errors='coerce')
else:
    # fallback if a single timestamp column exists
    for c in ['timestamp', 'datetime']: 
        if c in df_sales.columns:
            df_sales['datetime'] = pd.to_datetime(df_sales[c], errors='coerce')
            break

# Merge item metadata
# Normalize item name strings
df_sales['item_name'] = df_sales['item_name'].str.strip()
df_items['item_name'] = df_items['item_name'].str.strip()

df = df_sales.merge(df_items, on='item_name', how='left')

# Ensure price and production_cost numeric
for col in ['price', 'production_cost']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# For rows where price is missing (unlikely), assume price 0
if 'price' not in df.columns:
    df['price'] = np.nan

# Revenue per row: if price available use price, else NaN
df['revenue'] = df['price']

# Profit per row (simple): price - production_cost
if 'production_cost' in df.columns:
    df['profit'] = df['price'] - df['production_cost']
else:
    df['profit'] = np.nan

# Set some helpful derived columns
df['date_only'] = pd.to_datetime(df['datetime'].dt.date)
# Extract hour and weekday
df['hour'] = df['datetime'].dt.hour

# weekday as name
df['weekday'] = df['datetime'].dt.day_name()

# Replace 'N/A' or 'n/a' with NaN for own_cup when necessary
if 'own_cup' in df.columns:
    df['own_cup'] = df['own_cup'].replace({'N/A': np.nan, 'n/a': np.nan})
    # Attempt to coerce to boolean
    df['own_cup_bool'] = df['own_cup'].astype('boolean')
else:
    df['own_cup_bool'] = pd.NA

# Basic aggregates
# Time series: daily counts and revenue
daily = df.groupby('date_only').agg(count=('item_name', 'count'), revenue=('revenue', 'sum'), profit=('profit','sum')).reset_index()
# Rolling 7-day average of count
daily = daily.sort_values('date_only')
daily['count_7d_roll'] = daily['count'].rolling(7, min_periods=1).mean()

def save_fig(fig, name):
    path = OUT_DIR / f"{name}.html"
    fig.write_html(str(path))
    print('Wrote', path)

# 1) Time series: daily count + rolling avg and revenue (secondary y)
fig_ts = go.Figure()
fig_ts.add_trace(go.Bar(x=daily['date_only'], y=daily['count'], name='Daily Count', marker_color='lightseagreen'))
fig_ts.add_trace(go.Line(x=daily['date_only'], y=daily['count_7d_roll'], name='7-day rolling avg', line=dict(color='darkblue')))
fig_ts.add_trace(go.Line(x=daily['date_only'], y=daily['revenue'], name='Daily Revenue', yaxis='y2', line=dict(color='orange')))
fig_ts.update_layout(title='Daily Sales Count and Revenue', xaxis_title='Date', yaxis=dict(title='Count'), yaxis2=dict(title='Revenue', overlaying='y', side='right'))
save_fig(fig_ts, 'time_series_daily')

# 2) Top items by revenue and quantity
top_by_revenue = df.groupby('item_name').agg(count=('item_name','count'), revenue=('revenue','sum')).reset_index().sort_values('revenue', ascending=False)
top_by_count = top_by_revenue.sort_values('count', ascending=False)

fig_top_rev = px.bar(top_by_revenue.head(20).sort_values('revenue'), x='revenue', y='item_name', orientation='h', title='Top items by revenue (top 20)', labels={'revenue':'Revenue','item_name':'Item'})
save_fig(fig_top_rev, 'top_items_by_revenue')

fig_top_count = px.bar(top_by_count.head(20).sort_values('count'), x='count', y='item_name', orientation='h', title='Top items by count (top 20)', labels={'count':'Count','item_name':'Item'})
save_fig(fig_top_count, 'top_items_by_count')

# 3) Item type comparison (drinks vs merchandise) — revenue & profit margin
if 'item_type' in df.columns:
    type_summary = df.groupby('item_type').agg(revenue=('revenue','sum'), profit=('profit','sum')).reset_index()
    type_summary['margin'] = type_summary['profit'] / type_summary['revenue']
    fig_type = px.bar(type_summary, x='item_type', y='revenue', color='margin', title='Revenue by Item Type (colored by margin)', labels={'margin':'Profit Margin'})
    save_fig(fig_type, 'item_type_revenue_margin')

# 4) Hour vs weekday heatmap of sales volume
# Create pivot table: weekday order
weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
heat = df.groupby(['weekday','hour']).size().reset_index(name='count')
heat['weekday'] = pd.Categorical(heat['weekday'], categories=weekday_order, ordered=True)
heat_pivot = heat.pivot(index='weekday', columns='hour', values='count').fillna(0).loc[weekday_order]
fig_heat = px.imshow(heat_pivot, labels=dict(x='Hour of day', y='Weekday', color='Sales count'), x=heat_pivot.columns, y=heat_pivot.index, title='Sales volume heatmap (weekday vs hour)')
save_fig(fig_heat, 'heatmap_weekday_hour')

# 5) own_cup adoption over time
if 'own_cup_bool' in df.columns:
    own = df.dropna(subset=['own_cup_bool']).groupby('date_only').agg(total=('own_cup_bool','count'), own_cup_count=('own_cup_bool', lambda s: (s==True).sum())).reset_index()
    if not own.empty:
        own['own_pct'] = own['own_cup_count'] / own['total']
        fig_own = px.line(own, x='date_only', y='own_pct', title='Own cup adoption over time', labels={'own_pct':'% using own cup','date_only':'Date'})
        fig_own.update_yaxes(tickformat='.0%')
        save_fig(fig_own, 'own_cup_adoption')

# Save cleaned merged csv for reproducibility
cleaned_path = OUT_DIR / 'cleaned_merged_sales.csv'
df.to_csv(cleaned_path, index=False)
print('Saved cleaned merged CSV to', cleaned_path)

print('All done. Open the HTML files in the `outputs/` folder.')
