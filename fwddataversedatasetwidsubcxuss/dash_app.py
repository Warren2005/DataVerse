"""
Dash app for interactive exploration of the coffee shop dataset.
Run with:
  /Users/wld/Downloads/fwddataversedatasetwidsubcxuss/.venv/bin/python dash_app.py
Then open http://127.0.0.1:8050 in your browser.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path('/Users/wld/Downloads/fwddataversedatasetwidsubcxuss')
CLEANED = ROOT / 'outputs' / 'cleaned_merged_sales.csv'

# Load cleaned data (fallback to merging original CSVs if cleaned file missing)
if CLEANED.exists():
    df = pd.read_csv(CLEANED, parse_dates=['datetime', 'date_only'])
else:
    # Fallback: merge on the fly
    SALES = ROOT / 'sales.csv'
    ITEMS = ROOT / 'items.csv'
    df_sales = pd.read_csv(SALES)
    df_items = pd.read_csv(ITEMS)
    df_sales['item_name'] = df_sales['item_name'].str.strip()
    df_items['item_name'] = df_items['item_name'].str.strip()
    df = df_sales.merge(df_items, on='item_name', how='left')
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
    df['date_only'] = pd.to_datetime(df['datetime'].dt.date)

# Ensure types
if 'price' in df.columns:
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
if 'production_cost' in df.columns:
    df['production_cost'] = pd.to_numeric(df['production_cost'], errors='coerce')
if 'revenue' not in df.columns:
    df['revenue'] = df.get('price')
if 'profit' not in df.columns and 'production_cost' in df.columns:
    df['profit'] = df['price'] - df['production_cost']

# Time features
df['date_only'] = pd.to_datetime(df['date_only'])
df['hour'] = pd.to_datetime(df['datetime']).dt.hour
try:
    df['weekday'] = pd.to_datetime(df['datetime']).dt.day_name()
except Exception:
    df['weekday'] = pd.to_datetime(df['date_only']).dt.day_name()

# Precompute item summary
item_summary = df.groupby('item_name').agg(count=('item_name','count'), revenue=('revenue','sum')).reset_index().fillna(0)
# Lightweight pre-aggregations to speed up common queries
daily_agg = df.groupby('date_only').agg(count=('item_name','count'), revenue=('revenue','sum')).reset_index()
customer_agg = df.dropna(subset=['customer_id']).groupby('customer_id').agg(first_purchase=('date_only','min'), purchases=('item_name','count')).reset_index() if 'customer_id' in df.columns else pd.DataFrame()

# Cohort: assign cohort (month of first purchase) and prepare retention table
cohort_retention = None
if 'customer_id' in df.columns and not customer_agg.empty:
    cust_first = customer_agg[['customer_id','first_purchase']].copy()
    cust_first['cohort'] = pd.to_datetime(cust_first['first_purchase']).dt.to_period('M').dt.to_timestamp()
    df_c = df.dropna(subset=['customer_id']).copy()
    df_c['purchase_month'] = pd.to_datetime(df_c['date_only']).dt.to_period('M').dt.to_timestamp()
    df_c = df_c.merge(cust_first[['customer_id','cohort']], on='customer_id', how='left')
    # active customers per cohort per month
    cohort_pivot = df_c.groupby(['cohort','purchase_month'])['customer_id'].nunique().reset_index()
    cohort_sizes = cust_first.groupby('cohort')['customer_id'].nunique().reset_index(name='cohort_size')
    cohort_retention = cohort_pivot.merge(cohort_sizes, on='cohort', how='left')
    cohort_retention['retention'] = cohort_retention['customer_id'] / cohort_retention['cohort_size']
# App
app = Dash(__name__, title='Coffee shop sales explorer')
server = app.server

# Layout
min_date = df['date_only'].min().date() if not df['date_only'].isna().all() else datetime.now().date()
max_date = df['date_only'].max().date() if not df['date_only'].isna().all() else datetime.now().date()

item_types = ['All'] + sorted(df['item_type'].dropna().unique().tolist()) if 'item_type' in df.columns else ['All']

app.layout = html.Div([
    html.H2('Coffee shop interactive dashboard'),
    html.Div([
        html.Div([
            html.Label('Date range', title='Select start and end dates to filter the dataset'),
            dcc.DatePickerRange(id='date-range', start_date=min_date, end_date=max_date, display_format='YYYY-MM-DD')
        ], style={'display':'inline-block', 'margin-right':'24px'}),
        html.Div([
            html.Label('Item type', title='Filter by item_type (or choose All)'),
            dcc.Dropdown(id='item-type', options=[{'label':t,'value':t} for t in item_types], value='All', clearable=False, style={'width':'200px'})
        ], style={'display':'inline-block', 'margin-right':'24px'}),
        html.Div([
            html.Label('Segment by', title='Choose a column to segment or color the visualizations'),
            dcc.Dropdown(id='segment-col', options=[
                {'label':'None','value':'None'},
                {'label':'transaction_type','value':'transaction_type'},
                {'label':'drink_type','value':'drink_type'},
                {'label':'drink_temperature','value':'drink_temperature'},
                {'label':'item_type','value':'item_type'},
                {'label':'own_cup','value':'own_cup_bool'},
                {'label':'surcharge','value':'surcharge'}
            ], value='None', clearable=False, style={'width':'220px'})
        ], style={'display':'inline-block', 'margin-right':'24px'}),
        html.Div([
            html.Label('Segment values (multi-select)', title='Choose which values of the selected segment to include'),
            dcc.Dropdown(id='segment-values', options=[], value=[], multi=True, placeholder='Select values to filter', style={'width':'320px'})
        ], style={'display':'inline-block', 'margin-right':'24px'}),
        html.Div([
            html.Label('Top N (for items)', title='Limit categorical charts to the top-N categories for readability'),
            dcc.Slider(id='top-n', min=5, max=50, step=1, value=10, marks={5:'5',10:'10',20:'20',50:'50'})
        ], style={'display':'inline-block', 'width':'300px', 'verticalAlign':'top'})
        ,
        html.Div([
            html.Label('Transaction type', title='Filter by transaction_type (e.g., sale, refund)'),
            dcc.Dropdown(id='transaction-type', options=[{'label':t,'value':t} for t in sorted(df['transaction_type'].dropna().unique())] if 'transaction_type' in df.columns else [], value=None, clearable=True, style={'width':'220px'})
        ], style={'display':'inline-block', 'margin-left':'12px', 'margin-right':'24px'}),
        html.Div([
            html.Label('Chart type', title='Choose visualization type: Bar, Histogram, Pie, Line, Scatter'),
            dcc.Dropdown(id='chart-type', options=[
                {'label':'Bar','value':'bar'},
                {'label':'Histogram','value':'histogram'},
                {'label':'Pie','value':'pie'},
                {'label':'Line','value':'line'},
                {'label':'Scatter','value':'scatter'}
            ], value='bar', clearable=False, style={'width':'160px'})
        ], style={'display':'inline-block', 'margin-right':'12px'}),
        html.Div([
            html.Label('X axis', title='Choose the X-axis field for the custom visualization'),
            dcc.Dropdown(id='x-axis', options=[
                {'label':'Item name','value':'item_name'},
                {'label':'Item type','value':'item_type'},
                {'label':'Drink type','value':'drink_type'},
                {'label':'Drink temp','value':'drink_temperature'},
                {'label':'Hour','value':'hour'},
                {'label':'Weekday','value':'weekday'},
                {'label':'Date','value':'date_only'}
            ], value='item_name', clearable=False, style={'width':'180px'})
        ], style={'display':'inline-block', 'margin-right':'12px'}),
        html.Div([
            html.Label('Y metric', title='Choose the metric shown on Y-axis (numeric) or Count for frequency)'),
            dcc.Dropdown(id='y-metric', options=[
                {'label':'Count','value':'count'},
                {'label':'Revenue','value':'revenue'},
                {'label':'Profit','value':'profit'}
            ], value='count', clearable=False, style={'width':'140px'})
        ], style={'display':'inline-block', 'margin-right':'12px'}),
        html.Div([
            html.Label('Agg', title='Aggregation function to apply to the Y metric (sum, mean, count)'),
            dcc.Dropdown(id='agg-func', options=[{'label':'Sum','value':'sum'},{'label':'Mean','value':'mean'},{'label':'Count','value':'count'}], value='sum', clearable=False, style={'width':'120px'})
        ], style={'display':'inline-block'}),
    ], style={'marginBottom':'20px'}),

    dcc.Tabs(id='tabs', value='tab-ts', children=[
        dcc.Tab(label='Time series', value='tab-ts'),
        dcc.Tab(label='Top items', value='tab-top'),
    dcc.Tab(label='Segments', value='tab-seg'),
    dcc.Tab(label='Price vs Hour', value='tab-pricehour'),
    dcc.Tab(label='Custom Viz', value='tab-custom'),
    dcc.Tab(label='Customers', value='tab-customers'),
        dcc.Tab(label='Heatmap (weekday vs hour)', value='tab-heat'),
        dcc.Tab(label='Own-cup adoption', value='tab-own')
    ]),
    html.Div([
        html.Button('Download filtered data (CSV)', id='download-btn', n_clicks=0, style={'marginRight':'12px'}),
        dcc.Download(id='download-data')
    ], style={'marginTop':'8px'}),

    html.Div(id='tab-content', style={'marginTop':'20px'}),

    html.Div(style={'marginTop':'20px','fontSize':'12px','color':'#666'}, children=[
        html.Div('App reads cleaned data at: {}'.format(str(CLEANED))),
        html.Div('If app seems slow with many rows, consider using the cleaned aggregated CSV or sampling.')
    ])
])


# Callbacks
@app.callback(
    Output('segment-values','options'),
    Output('segment-values','value'),
    Input('segment-col','value')
)
def update_segment_values(col):
    """Populate the segment-values dropdown based on chosen segment column and clear previous selection."""
    if not col or col == 'None' or col not in df.columns:
        return [], []
    vals = sorted(df[col].dropna().unique().tolist())
    options = [{'label': str(v), 'value': v} for v in vals]
    # clear current selection when segment column changes
    return options, []

def filter_df(start_date, end_date, item_type, transaction_type, segment_col, segment_values):
    d = df.copy()
    if start_date:
        d = d[d['date_only'] >= pd.to_datetime(start_date)]
    if end_date:
        d = d[d['date_only'] <= pd.to_datetime(end_date)]
    if item_type and item_type!='All' and 'item_type' in d.columns:
        d = d[d['item_type']==item_type]
    if transaction_type and 'transaction_type' in d.columns:
        d = d[d['transaction_type'] == transaction_type]
    if segment_col and segment_col!='None' and segment_values:
        # guard against values not present
        vals = [v for v in segment_values if v in d[segment_col].unique()]
        if vals:
            d = d[d[segment_col].isin(vals)]
    return d


@app.callback(
    Output('tab-content','children'),
    Input('tabs','value'),
    Input('date-range','start_date'),
    Input('date-range','end_date'),
    Input('item-type','value'),
    Input('top-n','value'),
    Input('segment-col','value'),
    Input('segment-values','value'),
    Input('transaction-type','value'),
    Input('chart-type','value'),
    Input('x-axis','value'),
    Input('y-metric','value'),
    Input('agg-func','value')
)
def render_tab(tab, start_date, end_date, item_type, top_n, segment_col, segment_values, transaction_type, chart_type, x_axis, y_metric, agg_func):
    # filter df by dates and other controls using helper
    d = filter_df(start_date, end_date, item_type, transaction_type, segment_col, segment_values)

    if tab=='tab-ts':
        daily = d.groupby('date_only').agg(count=('item_name','count'), revenue=('revenue','sum')).reset_index().sort_values('date_only')
        daily['count_7d'] = daily['count'].rolling(7, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily['date_only'], y=daily['count'], name='Daily count', marker_color='lightseagreen'))
        fig.add_trace(go.Scatter(x=daily['date_only'], y=daily['count_7d'], name='7-day avg', line=dict(color='darkblue')))
        fig.add_trace(go.Scatter(x=daily['date_only'], y=daily['revenue'], name='Revenue', yaxis='y2', line=dict(color='orange')))
        fig.update_layout(title='Daily count and revenue', xaxis_title='Date', yaxis=dict(title='Count'), yaxis2=dict(title='Revenue', overlaying='y', side='right'))
        return dcc.Graph(figure=fig)

    if tab=='tab-top':
        summary = d.groupby('item_name').agg(count=('item_name','count'), revenue=('revenue','sum')).reset_index().fillna(0)
        metric = 'revenue'
        top = summary.sort_values(metric, ascending=False).head(top_n).sort_values(metric)
        fig = px.bar(top, x=metric, y='item_name', orientation='h', title=f'Top {top_n} items by {metric}')
        fig.update_layout(height=600)
        return dcc.Graph(figure=fig)

    if tab=='tab-heat':
        heat = d.groupby(['weekday','hour']).size().reset_index(name='count')
        weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        heat['weekday'] = pd.Categorical(heat['weekday'], categories=weekday_order, ordered=True)
        pivot = heat.pivot(index='weekday', columns='hour', values='count').fillna(0).reindex(weekday_order)
        fig = px.imshow(pivot, labels=dict(x='Hour', y='Weekday', color='Sales count'), x=pivot.columns, y=pivot.index, title='Sales heatmap (weekday vs hour)')
        return dcc.Graph(figure=fig)

    if tab=='tab-seg':
        # Time series per segment (stacked area)
        if segment_col and segment_col!='None':
            seg_col = segment_col
            # If user selected specific values, filter to them
            if segment_values:
                dseg = d[d[seg_col].isin(segment_values)]
            else:
                dseg = d.copy()
            grp = dseg.groupby(['date_only', seg_col]).size().reset_index(name='count')
            if grp.empty:
                return html.Div('No data for selected segmentation and filters')
            fig = px.area(grp, x='date_only', y='count', color=seg_col, title=f'Sales over time segmented by {seg_col}')
            fig.update_layout(height=600)
            return dcc.Graph(figure=fig)
        else:
            return html.Div('Select a segment column to view segmented timeseries.')

    if tab=='tab-pricehour':
        # Scatter: price vs hour, colored by segment if set
        if 'price' in d.columns:
            plotdf = d.dropna(subset=['price','hour'])
            if segment_col and segment_col!='None' and segment_values:
                plotdf = plotdf[plotdf[segment_col].isin(segment_values)]
            color = segment_col if (segment_col and segment_col!='None') else None
            # sample for speed
            sample_n = min(len(plotdf), 2000)
            if sample_n <= 0:
                return html.Div('No price/hour data to plot for selected filters')
            fig = px.scatter(plotdf.sample(sample_n), x='hour', y='price', color=color, hover_data=['item_name','transaction_type'], title='Price vs Hour (sampled)')
            fig.update_layout(height=600)
            return dcc.Graph(figure=fig)
        else:
            return html.Div('Price column not available in data')

    if tab=='tab-custom':
        # Render a placeholder that will be populated by a dedicated custom-viz callback.
        # This ensures chart-type and other custom controls update the chart immediately
        # via a separate callback without re-rendering the whole tab content.
        return html.Div(id='custom-viz-content')

    if tab=='tab-customers':
        # Customer-focused metrics: daily unique customers and purchases-per-customer distribution
        if 'customer_id' not in d.columns:
            return html.Div('No customer_id column available in data')
        cust = d.dropna(subset=['customer_id']).copy()
        if cust.empty:
            return html.Div('No customer data for selected filters')

        daily_unique = cust.groupby('date_only')['customer_id'].nunique().reset_index(name='unique_customers')
        purchases_per_customer = cust.groupby('customer_id').size().reset_index(name='purchases')

        fig1 = px.line(daily_unique, x='date_only', y='unique_customers', title='Daily unique customers')
        freq = purchases_per_customer['purchases'].value_counts().reset_index().sort_values('index')
        freq.columns = ['purchases','count']
        fig2 = px.bar(freq, x='purchases', y='count', title='Distribution of purchases per customer')
        # Cohort retention (if precomputed cohort_retention exists)
        cohort_div = None
        if cohort_retention is not None and not cohort_retention.empty:
            # filter cohort_retention to cohorts present in the current date range if applicable
            crm = cohort_retention.copy()
            # pivot for heatmap-like retention
            pivot = crm.pivot(index='cohort', columns='purchase_month', values='retention').fillna(0)
            # sort columns chronologically
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)
            fig3 = px.imshow(pivot, labels=dict(x='Purchase month', y='Cohort month', color='Retention'), x=[c.strftime('%Y-%m') for c in pivot.columns], y=[c.strftime('%Y-%m') for c in pivot.index], title='Cohort retention (fraction of cohort active)')
            cohort_div = dcc.Graph(figure=fig3)

        return html.Div([dcc.Graph(figure=fig1), dcc.Graph(figure=fig2), cohort_div])


    @app.callback(
        Output('custom-viz-content','children'),
        Input('tabs','value'),
        Input('date-range','start_date'),
        Input('date-range','end_date'),
        Input('item-type','value'),
        Input('transaction-type','value'),
        Input('segment-col','value'),
        Input('segment-values','value'),
        Input('chart-type','value'),
        Input('x-axis','value'),
        Input('y-metric','value'),
        Input('agg-func','value'),
        Input('top-n','value')
    )
    def render_custom_viz(tab, start_date, end_date, item_type, transaction_type, segment_col, segment_values, chart_type, x_axis, y_metric, agg_func, top_n):
        # Only render when Custom Viz tab is active
        if tab != 'tab-custom':
            return html.Div('Custom Viz: select the tab to show charts')

        d = filter_df(start_date, end_date, item_type, transaction_type, segment_col, segment_values)
        if d is None or d.empty:
            return html.Div('No data for selected filters')

        color = segment_col if (segment_col and segment_col!='None' and segment_col in d.columns) else None

        try:
            if chart_type == 'bar':
                if y_metric == 'count' or agg_func == 'count':
                    grp = d.groupby(x_axis).size().reset_index(name='count')
                    grp = grp.sort_values('count', ascending=False).head(top_n)
                    fig = px.bar(grp, x=x_axis, y='count', color=color, title=f'Bar: count by {x_axis}')
                else:
                    if y_metric not in d.columns:
                        return html.Div(f'Metric "{y_metric}" not available')
                    if agg_func == 'sum':
                        grp = d.groupby(x_axis)[y_metric].sum().reset_index()
                    elif agg_func == 'mean':
                        grp = d.groupby(x_axis)[y_metric].mean().reset_index()
                    else:
                        grp = d.groupby(x_axis)[y_metric].count().reset_index(name='count')
                    grp = grp.sort_values(grp.columns[-1], ascending=False).head(top_n)
                    fig = px.bar(grp, x=x_axis, y=grp.columns[-1], color=color, title=f'Bar: {y_metric} ({agg_func}) by {x_axis}')

            elif chart_type == 'histogram':
                fig = px.histogram(d, x=x_axis, color=color, title=f'Histogram of {x_axis}')

            elif chart_type == 'pie':
                if y_metric == 'count' or agg_func == 'count':
                    grp = d.groupby(x_axis).size().reset_index(name='count')
                    grp = grp.sort_values('count', ascending=False).head(top_n)
                    fig = px.pie(grp, names=x_axis, values='count', title=f'Pie: count by {x_axis}')
                else:
                    if y_metric not in d.columns:
                        return html.Div(f'Metric "{y_metric}" not available')
                    grp = d.groupby(x_axis)[y_metric].sum().reset_index()
                    grp = grp.sort_values(y_metric, ascending=False).head(top_n)
                    fig = px.pie(grp, names=x_axis, values=y_metric, title=f'Pie: {y_metric} by {x_axis}')

            elif chart_type == 'line':
                if x_axis == 'date_only':
                    grp = d.groupby('date_only').agg(val=(y_metric if y_metric in d.columns else 'item_name','sum' if y_metric in d.columns else 'count')).reset_index()
                    fig = px.line(grp, x='date_only', y='val', title=f'Line: {y_metric} over time')
                else:
                    # aggregate by x_axis
                    if y_metric in d.columns:
                        grp = d.groupby(x_axis)[y_metric].sum().reset_index()
                        fig = px.line(grp, x=x_axis, y=y_metric, title=f'Line: {y_metric} by {x_axis}')
                    else:
                        grp = d.groupby(x_axis).size().reset_index(name='val')
                        fig = px.line(grp, x=x_axis, y='val', title=f'Line: count by {x_axis}')

            elif chart_type == 'scatter':
                if y_metric == 'count' or y_metric not in d.columns:
                    return html.Div('Scatter requires a numeric Y metric present in data')
                s = d.dropna(subset=[x_axis, y_metric])
                sample_n = min(len(s), 2000)
                if sample_n <= 0:
                    return html.Div('No data for scatter')
                fig = px.scatter(s.sample(sample_n), x=x_axis, y=y_metric, color=color, title=f'Scatter: {y_metric} vs {x_axis}')

            else:
                return html.Div('Unknown chart type')

            fig.update_layout(height=650)
            return dcc.Graph(figure=fig)
        except Exception as e:
            return html.Div(f'Error rendering custom viz: {str(e)}')


    @app.callback(
        Output('y-metric','disabled'),
        Output('agg-func','disabled'),
        Output('top-n','disabled'),
        Input('chart-type','value')
    )
    def update_control_disabled(chart_type):
        """Disable or enable controls depending on the selected chart type."""
        # Defaults: enabled (False)
        if not chart_type:
            return False, False, False
        if chart_type == 'bar':
            return False, False, False
        if chart_type == 'histogram':
            return True, True, True
        if chart_type == 'pie':
            return False, False, False
        if chart_type == 'line':
            return False, False, True
        if chart_type == 'scatter':
            return False, True, True
        # fallback: enable all
        return False, False, False

    if tab=='tab-own':
        if 'own_cup_bool' in d.columns and d['own_cup_bool'].notna().any():
            own = d.dropna(subset=['own_cup_bool']).groupby('date_only').agg(total=('own_cup_bool','count'), own_cup_count=('own_cup_bool', lambda s: (s==True).sum())).reset_index()
            own['own_pct'] = own['own_cup_count'] / own['total']
            fig = px.line(own, x='date_only', y='own_pct', title='Own cup adoption over time', labels={'own_pct':'% using own cup'})
            fig.update_yaxes(tickformat='.0%')
            return dcc.Graph(figure=fig)
        else:
            return html.Div('No own_cup data available for the selected filters.')

    return html.Div('Unknown tab')


@app.callback(
    Output('download-data','data'),
    Input('download-btn','n_clicks'),
    State('date-range','start_date'),
    State('date-range','end_date'),
    State('item-type','value'),
    State('transaction-type','value'),
    State('segment-col','value'),
    State('segment-values','value'),
    prevent_initial_call=True
)
def download_filtered(n_clicks, start_date, end_date, item_type, transaction_type, segment_col, segment_values):
    # prepare filtered dataframe and send as CSV
    d = filter_df(start_date, end_date, item_type, transaction_type, segment_col, segment_values)
    if d is None or d.empty:
        return dcc.send_data_frame(pd.DataFrame().to_csv, 'filtered_sales.csv')
    return dcc.send_data_frame(d.to_csv, 'filtered_sales.csv', index=False)


if __name__ == '__main__':
    # Run server. If you want to expose to network change host to '0.0.0.0'
    # Newer Dash versions use app.run instead of app.run_server
    try:
        app.run(host='127.0.0.1', port=8050, debug=False)
    except TypeError:
        # Fallback for older Dash versions that still expect run_server
        app.run_server(host='127.0.0.1', port=8050, debug=False)
