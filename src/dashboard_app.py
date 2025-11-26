import streamlit as st
import pandas as pd
import numpy as np
import os
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Config
st.set_page_config(page_title="OPSD PowerDesk Dashboard", layout="wide")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
OUTPUT_DIR = config['data']['output_dir']
COUNTRIES = config['countries']

@st.cache_data
def load_data(country_code):
    # 1. Try loading Online Simulation (Best for Live Country)
    sim_path = os.path.join(OUTPUT_DIR, f'{country_code}_online_simulation.csv')
    
    if os.path.exists(sim_path):
        df = pd.read_csv(sim_path, parse_dates=['timestamp'])
        df['source'] = 'simulation'
    else:
        # 2. Fallback to Anomalies File (Test Set)
        anom_path = os.path.join(OUTPUT_DIR, f'{country_code}_anomalies.csv')
        if os.path.exists(anom_path):
            df = pd.read_csv(anom_path, parse_dates=['timestamp'])
            df['source'] = 'test_set'
        else:
            return pd.DataFrame(), pd.DataFrame()

    # Set Index
    df.set_index('timestamp', inplace=True)
    
    # Load Updates Log
    update_path = os.path.join(OUTPUT_DIR, f'{country_code}_online_updates.csv')
    if os.path.exists(update_path):
        df_updates = pd.read_csv(update_path, parse_dates=['timestamp'])
        df_updates.set_index('timestamp', inplace=True)
    else:
        df_updates = pd.DataFrame()

    return df, df_updates

# --- Sidebar ---
st.sidebar.title("⚡ PowerDesk")
selected_country = st.sidebar.selectbox("Select Country", COUNTRIES, index=0) # DE default

df, df_updates = load_data(selected_country)

if df.empty:
    st.error(f"No data found for {selected_country}.")
    st.stop()

# Time Travel Slider
min_time = df.index.min()
max_time = df.index.max()

# Default: End of data
default_time = max_time

st.sidebar.markdown("---")
st.sidebar.header("Simulation Control")
current_time = st.sidebar.slider(
    "Current Time (UTC)",
    min_value=min_time.to_pydatetime(),
    max_value=max_time.to_pydatetime(),
    value=default_time.to_pydatetime(),
    step=pd.Timedelta(hours=1),
    format="MM-DD HH:mm"
)
# Ensure timezone awareness (UTC)
current_time = pd.Timestamp(current_time).tz_localize('UTC') if pd.Timestamp(current_time).tz is None else pd.Timestamp(current_time)

# --- Data Slicing ---
# History: Last 14 days
history_start = current_time - pd.Timedelta(days=14)
mask_history = (df.index >= history_start) & (df.index <= current_time)
df_history = df.loc[mask_history]

# Future: Next 24h (Forecast Cone)
# We take the rows immediately following current_time
# In our data structure, these rows contain the forecast made for that time
mask_future = (df.index > current_time) & (df.index <= current_time + pd.Timedelta(hours=24))
df_future = df.loc[mask_future]

# --- KPI Tiles ---
st.markdown(f"### 🇪🇺 {selected_country} Power Grid Status")
st.markdown(f"**Current Time:** {current_time.strftime('%Y-%m-%d %H:%M UTC')}")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# 1. Rolling 7d MASE
# We calculate this on the fly for the *visible* history window (last 7 days of it)
history_7d = df_history.last('7D')
mase_val = "N/A"
if len(history_7d) > 24:
    mae = (history_7d['y_true'] - history_7d['yhat']).abs().mean()
    # Naive seasonal error (approximate)
    naive_err = (history_7d['y_true'].diff(24)).abs().mean()
    if naive_err > 0:
        mase = mae / naive_err
        mase_val = f"{mase:.3f}"

kpi1.metric("Rolling 7d MASE", mase_val)

# 2. 7d PI Coverage
cov_val = "N/A"
if len(history_7d) > 0:
    inside = (history_7d['y_true'] >= history_7d['lo']) & (history_7d['y_true'] <= history_7d['hi'])
    cov = inside.mean() * 100
    cov_val = f"{cov:.1f}%"

kpi2.metric("7d PI Coverage (80%)", cov_val)

# 3. Anomalies Today
history_24h = df_history.last('24H')
anom_count = 0
if len(history_24h) > 0:
    anom_count = history_24h['flag_z'].sum()
    
kpi3.metric("Anomalies (Last 24h)", int(anom_count), delta_color="inverse")

# 4. Last Update
last_update_str = "Offline"
last_reason = ""
if not df_updates.empty:
    # Find last update <= current_time
    past_updates = df_updates[df_updates.index <= current_time]
    if not past_updates.empty:
        last_upd = past_updates.iloc[-1]
        last_update_str = last_upd.name.strftime('%H:%M')
        last_reason = last_upd['reason']

kpi4.metric("Last Online Update", last_update_str, delta=last_reason)

# --- Main Chart ---
tab1, tab2 = st.tabs(["Live Forecast", "Data View"])

with tab1:
    fig = go.Figure()

    # Actuals
    fig.add_trace(go.Scatter(
        x=df_history.index, y=df_history['y_true'],
        mode='lines', name='Actual Load',
        line=dict(color='black', width=2)
    ))

    # Forecast (History)
    fig.add_trace(go.Scatter(
        x=df_history.index, y=df_history['yhat'],
        mode='lines', name='Forecast (Past)',
        line=dict(color='blue', width=1, dash='dot')
    ))

    # Anomalies
    anoms = df_history[df_history['flag_z'] == 1]
    if not anoms.empty:
        fig.add_trace(go.Scatter(
            x=anoms.index, y=anoms['y_true'],
            mode='markers', name='Anomaly',
            marker=dict(color='red', size=8, symbol='x')
        ))

    # Forecast Cone (Future)
    if not df_future.empty:
        # Mean
        fig.add_trace(go.Scatter(
            x=df_future.index, y=df_future['yhat'],
            mode='lines', name='Forecast (Next 24h)',
            line=dict(color='blue', width=2)
        ))
        # PI
        fig.add_trace(go.Scatter(
            x=df_future.index.tolist() + df_future.index[::-1].tolist(),
            y=df_future['hi'].tolist() + df_future['lo'][::-1].tolist(),
            fill='toself', fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='80% PI', showlegend=True
        ))

    fig.update_layout(
        title="Load Forecast & Anomalies",
        xaxis_title="Time (UTC)",
        yaxis_title="Load (MW)",
        height=500,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Anomaly Tape ---
    st.subheader("Anomaly Tape (Last 14 Days)")
    
    tape_fig = go.Figure()
    
    # Background
    tape_fig.add_trace(go.Scatter(
        x=df_history.index, y=[1]*len(df_history),
        mode='lines', line=dict(color='#f0f2f6', width=15),
        hoverinfo='skip', showlegend=False
    ))
    
    # Anomalies (Z-score or CUSUM)
    # Check if flag_cusum exists
    if 'flag_cusum' in df_history.columns:
        tape_anoms = df_history[(df_history['flag_z'] == 1) | (df_history['flag_cusum'] == 1)]
    else:
        tape_anoms = df_history[df_history['flag_z'] == 1]
        
    if not tape_anoms.empty:
        tape_fig.add_trace(go.Scatter(
            x=tape_anoms.index, y=[1]*len(tape_anoms),
            mode='markers',
            marker=dict(color='red', size=10, symbol='square'),
            name='Anomaly',
            text=tape_anoms.index.strftime('%Y-%m-%d %H:%M'),
            hovertemplate='Anomaly: %{text}<extra></extra>'
        ))
        
    tape_fig.update_layout(
        height=100,
        yaxis=dict(visible=False, range=[0.9, 1.1]),
        xaxis=dict(visible=True, title="Time"),
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(tape_fig, use_container_width=True)

with tab2:
    st.dataframe(df_history.sort_index(ascending=False))
