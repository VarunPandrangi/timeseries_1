import pandas as pd
import numpy as np
import os
import time
import warnings
import yaml
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
OUTPUT_DIR = config['data']['output_dir']
COUNTRY = config['countries'][0] # Default to first country (DE)
SIMULATION_HOURS = config['live_simulation']['simulation_hours']
START_HISTORY_HOURS = config['live_simulation']['history_days'] * 24
ROLLING_WINDOW_HOURS = 90 * 24 # 90 days for refit (hardcoded in strategy description)
DRIFT_WINDOW_HOURS = config['live_simulation']['drift_window_days'] * 24
METRIC_WINDOW_HOURS = 7 * 24 # 7 days for metrics

# Model Order for DE from Config
ORDER = tuple(config['forecasting']['model_orders'][COUNTRY]['order'])
SEASONAL_ORDER = tuple(config['forecasting']['model_orders'][COUNTRY]['seasonal_order'])

def load_data():
    # Load cleaned data
    clean_path = os.path.join(OUTPUT_DIR, f'{COUNTRY}_cleaned.csv')
    df = pd.read_csv(clean_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.asfreq('H')
    df['load'] = df['load'].ffill()
    
    # Exogenous
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    
    hour_dummies = pd.get_dummies(df['hour'], prefix='h', drop_first=True).astype(float)
    dow_dummies = pd.get_dummies(df['dayofweek'], prefix='d', drop_first=True).astype(float)
    exog = pd.concat([hour_dummies, dow_dummies], axis=1)
    
    if 'wind' in df.columns:
        df['wind'] = df['wind'].fillna(0).astype(float)
        exog['wind'] = df['wind']
    if 'solar' in df.columns:
        df['solar'] = df['solar'].fillna(0).astype(float)
        exog['solar'] = df['solar']
        
    exog.index = df.index
    
    return df, exog

def get_splits(df):
    n = len(df)
    train_end = int(n * 0.8)
    dev_end = int(n * 0.9)
    
    # We start simulation at the beginning of Test set
    # But we need history (Train+Dev) for the initial model
    # Requirement: Live start history 120d
    history_df = df.iloc[:dev_end]
    if len(history_df) > START_HISTORY_HOURS:
        history_df = history_df.iloc[-START_HISTORY_HOURS:]
        
    test_df = df.iloc[dev_end:]
    
    return history_df, test_df

def calculate_mase(y_true, y_pred, y_train_history, seasonality=24):
    if len(y_true) == 0: return np.nan
    mae = mean_absolute_error(y_true, y_pred)
    
    # Naive seasonal error on history
    # We use a fixed slice of history for denominator stability or rolling?
    # Standard MASE uses in-sample naive error.
    # We'll use the last 90 days of history for the denominator to be relevant.
    y_hist = y_train_history[-ROLLING_WINDOW_HOURS:]
    naive_errors = np.abs(y_hist[seasonality:] - y_hist[:-seasonality])
    d = np.mean(naive_errors)
    
    return mae / d if d != 0 else np.nan

def calculate_coverage(y_true, lo, hi):
    if len(y_true) == 0: return np.nan
    inside = (y_true >= lo) & (y_true <= hi)
    return inside.mean() * 100

def run_simulation():
    print(f"Starting Online Simulation for {COUNTRY}...")
    
    df, exog = load_data()
    history_df, test_df = get_splits(df)
    history_exog = exog.loc[history_df.index]
    test_exog = exog.loc[test_df.index]
    
    # Limit simulation to 2000 hours + buffer
    sim_df = test_df.iloc[:SIMULATION_HOURS + 48] 
    sim_exog = test_exog.iloc[:SIMULATION_HOURS + 48]
    
    # Initial Model Fit
    # Fit on last 90 days of history to be consistent with strategy
    print("Initializing model...")
    init_train_data = history_df.iloc[-ROLLING_WINDOW_HOURS:]
    init_train_exog = history_exog.iloc[-ROLLING_WINDOW_HOURS:]
    
    model = SARIMAX(init_train_data['load'], 
                    exog=init_train_exog,
                    order=ORDER, 
                    seasonal_order=SEASONAL_ORDER,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)
    
    # State variables
    current_history_load = history_df['load'].tolist()
    current_history_exog = history_exog # DataFrame
    
    # Z-score stats (rolling window)
    # We need a buffer of residuals. 
    # Let's assume we start with empty residuals or pre-calculate on Dev?
    # To be safe, let's just collect residuals as we go. 
    # The prompt says "rolling z-score with window=336h, min_periods=168".
    # We can use the end of Dev residuals if we had them, but let's start fresh 
    # and wait for min_periods or use a warm-up.
    # Better: Calculate residuals on the last 336h of Dev using the initial model to warm up.
    
    print("Warming up residuals...")
    warmup_steps = 336
    warmup_data = history_df.iloc[-warmup_steps:]
    warmup_exog = history_exog.iloc[-warmup_steps:]
    
    # We can use get_prediction on in-sample data
    warmup_pred = results.get_prediction(start=warmup_data.index[0], end=warmup_data.index[-1], exog=warmup_exog)
    residuals_buffer = (warmup_data['load'] - warmup_pred.predicted_mean).tolist()
    
    # Drift stats   
    # EWMA of |z|
    ewma_z = 0.0
    alpha = 0.1
    z_score_buffer = [] # For percentile calculation (last 30 days)
    
    # CUSUM State
    cusum_s_pos = 0.0
    cusum_s_neg = 0.0
    
    # Forecast buffer: {timestamp: {yhat, lo, hi}}
    forecast_buffer = {}
    
    # Logs
    update_logs = []
    
    # Simulation Results
    simulation_results = []
    
    # Metrics buffer (for "Before" calculation)
    # We need actuals and predictions for the last 7 days
    metrics_buffer_y_true = []
    metrics_buffer_y_pred = []
    metrics_buffer_lo = []
    metrics_buffer_hi = []
    
    print(f"Simulating {SIMULATION_HOURS} hours...")
    
    start_time = time.time()
    
    for i in range(len(sim_df)):
        if i >= SIMULATION_HOURS:
            break
            
        current_time = sim_df.index[i]
        current_obs = sim_df['load'].iloc[i]
        current_exog_row = sim_exog.iloc[[i]]
        
        # 1. Retrieve Forecast (generated previously)
        # If we don't have a forecast (start of sim), we generate one now?
        # Or we assume we forecasted yesterday.
        # For i=0, we haven't forecasted yet.
        # Let's generate an initial forecast at i=0 for the first 24h?
        # But the loop says "at 00:00 forecast next 24h".
        # If i=0 is not 00:00, we might have an issue.
        # Let's just generate a forecast if missing.
        
        if current_time not in forecast_buffer:
            # Emergency forecast (should only happen at very start)
            pred_res = results.get_forecast(steps=1, exog=current_exog_row)
            yhat = pred_res.predicted_mean.iloc[0]
            conf = pred_res.conf_int(alpha=0.2).iloc[0]
            forecast_buffer[current_time] = {'yhat': yhat, 'lo': conf.iloc[0], 'hi': conf.iloc[1]}
            
        fc = forecast_buffer[current_time]
        yhat = fc['yhat']
        lo = fc['lo']
        hi = fc['hi']
        
        # 2. Compute Residual & Z-score
        resid = current_obs - yhat
        residuals_buffer.append(resid)
        
        # Maintain residual buffer size (need enough for 336h rolling)
        if len(residuals_buffer) > 336 * 2: # Keep some extra
            residuals_buffer.pop(0)
            
        # Compute rolling stats
        # We need last 336 residuals
        if len(residuals_buffer) >= 168:
            window_resids = residuals_buffer[-336:]
            roll_mean = np.mean(window_resids)
            roll_std = np.std(window_resids)
            
            if roll_std == 0: roll_std = 1e-6
            
            z = (resid - roll_mean) / roll_std
        else:
            z = 0.0 # Not enough data
            
        # Compute Flags
        z_val = 0.0 if np.isnan(z) else z
        flag_z = 1 if abs(z_val) >= 3.0 else 0
        
        # CUSUM
        cusum_s_pos = max(0, cusum_s_pos + z_val - 0.5)
        cusum_s_neg = max(0, cusum_s_neg - z_val - 0.5)
        flag_cusum = 1 if (cusum_s_pos > 5.0 or cusum_s_neg > 5.0) else 0
        
        simulation_results.append({
            'timestamp': current_time,
            'y_true': current_obs,
            'yhat': yhat,
            'lo': lo,
            'hi': hi,
            'z_resid': z_val,
            'flag_z': flag_z,
            'flag_cusum': flag_cusum
        })
            
        # 3. Update Drift Stats
        abs_z = abs(z)
        ewma_z = alpha * abs_z + (1 - alpha) * ewma_z
        
        z_score_buffer.append(abs_z)
        if len(z_score_buffer) > DRIFT_WINDOW_HOURS:
            z_score_buffer.pop(0)
            
        # 4. Check Triggers
        trigger_reason = None
        
        # Schedule: 00:00 UTC
        if current_time.hour == 0:
            trigger_reason = 'scheduled'
            
        # Drift
        if len(z_score_buffer) >= 168: # Wait a bit
            threshold = np.percentile(z_score_buffer, 95)
            if ewma_z > threshold:
                # Prioritize drift if both happen? Or just log drift?
                # If scheduled happens, we update anyway.
                if trigger_reason != 'scheduled':
                    trigger_reason = 'drift'
        
        # 5. Adaptation
        if trigger_reason:
            update_start = time.time()
            
            # Metrics Before
            # Last 7 days
            mase_before = np.nan
            cov_before = np.nan
            
            if len(metrics_buffer_y_true) >= METRIC_WINDOW_HOURS:
                y_true_7d = metrics_buffer_y_true[-METRIC_WINDOW_HOURS:]
                y_pred_7d = metrics_buffer_y_pred[-METRIC_WINDOW_HOURS:]
                lo_7d = metrics_buffer_lo[-METRIC_WINDOW_HOURS:]
                hi_7d = metrics_buffer_hi[-METRIC_WINDOW_HOURS:]
                
                # History for MASE (last 90d from current history)
                # current_history_load is list
                mase_before = calculate_mase(y_true_7d, y_pred_7d, np.array(current_history_load))
                cov_before = calculate_coverage(np.array(y_true_7d), np.array(lo_7d), np.array(hi_7d))
            
            # Refit
            # Data: last 90 days from NOW (including current_obs? No, usually we update after seeing obs)
            # We append current_obs to history at end of loop, but for refit we should include it?
            # "Online adaptation = after you ingest new data... update your model"
            # So yes, include current_obs.
            
            # Construct training set
            # We need to combine current_history (which doesn't have current_obs yet) + current_obs
            refit_load = current_history_load + [current_obs]
            refit_exog = pd.concat([current_history_exog, current_exog_row])
            
            # Take last 90 days
            train_load_subset = refit_load[-ROLLING_WINDOW_HOURS:]
            train_exog_subset = refit_exog.iloc[-ROLLING_WINDOW_HOURS:]
            
            # Refit Model
            try:
                # Create new model
                new_model = SARIMAX(train_load_subset, 
                                    exog=train_exog_subset,
                                    order=ORDER, 
                                    seasonal_order=SEASONAL_ORDER,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                # Use previous params as start to speed up?
                # results.params can be passed to start_params
                new_results = new_model.fit(disp=False, start_params=results.params, maxiter=50)
                results = new_results
                
                # Metrics After (Hindcast on last 7d)
                # We predict the last 7 days using the NEW model
                # This is in-sample prediction on the training set (tail)
                if len(metrics_buffer_y_true) >= METRIC_WINDOW_HOURS:
                    # The last 7 days are in train_load_subset (at the end)
                    # We can use get_prediction
                    hindcast_start = len(train_load_subset) - METRIC_WINDOW_HOURS
                    hindcast_res = results.get_prediction(start=hindcast_start, end=len(train_load_subset)-1, exog=train_exog_subset.iloc[hindcast_start:])
                    
                    y_pred_after = hindcast_res.predicted_mean
                    conf_after = hindcast_res.conf_int(alpha=0.2)
                    
                    mase_after = calculate_mase(y_true_7d, y_pred_after, np.array(refit_load))
                    cov_after = calculate_coverage(np.array(y_true_7d), conf_after.iloc[:, 0], conf_after.iloc[:, 1])
                else:
                    mase_after = np.nan
                    cov_after = np.nan
                    
                duration = time.time() - update_start
                
                update_logs.append({
                    'timestamp': current_time,
                    'strategy': 'Rolling SARIMA',
                    'reason': trigger_reason,
                    'duration_s': duration,
                    'MASE_before': mase_before,
                    'MASE_after': mase_after,
                    'Cov_before': cov_before,
                    'Cov_after': cov_after
                })
                
                print(f"Update at {current_time}: {trigger_reason} ({duration:.2f}s)")
                
            except Exception as e:
                print(f"Update failed at {current_time}: {e}")
        
        # 6. Forecast (if 00:00)
        # "at 00:00 UTC forecast next 24h"
        # This means we generate forecasts for t+1, t+2, ..., t+24
        if current_time.hour == 0:
            # We need exog for next 24h
            # Check if we have enough exog
            if i + 24 < len(sim_exog):
                next_24_exog = sim_exog.iloc[i+1 : i+25]
                
                # Forecast
                # We need to make sure the model state is up to date
                # If we just refitted, results is up to date (includes current_obs)
                # If we didn't refit, we need to append current_obs to results
                
                if not trigger_reason: # If we didn't refit
                    # Append current observation
                    # Note: append creates a new results object
                    results = results.append([current_obs], exog=current_exog_row, refit=False)
                
                # Now forecast
                pred_res = results.get_forecast(steps=24, exog=next_24_exog)
                fc_mean = pred_res.predicted_mean
                fc_conf = pred_res.conf_int(alpha=0.2)
                
                # Store in buffer
                for step in range(24):
                    # timestamp of forecast
                    fc_ts = sim_df.index[i + 1 + step]
                    forecast_buffer[fc_ts] = {
                        'yhat': fc_mean.iloc[step],
                        'lo': fc_conf.iloc[step, 0],
                        'hi': fc_conf.iloc[step, 1]
                    }
            else:
                # End of simulation data approaching
                pass
        else:
            # If not 00:00, we still need to update the model state with the current observation
            # unless we just refitted (which only happens at 00:00 or drift)
            # Drift can happen at any hour.
            if not trigger_reason:
                results = results.append([current_obs], exog=current_exog_row, refit=False)

        # 7. Update History & Buffers
        current_history_load.append(current_obs)
        current_history_exog = pd.concat([current_history_exog, current_exog_row])
        
        metrics_buffer_y_true.append(current_obs)
        metrics_buffer_y_pred.append(yhat)
        metrics_buffer_lo.append(lo)
        metrics_buffer_hi.append(hi)
        
        if len(metrics_buffer_y_true) > METRIC_WINDOW_HOURS:
            metrics_buffer_y_true.pop(0)
            metrics_buffer_y_pred.pop(0)
            metrics_buffer_lo.pop(0)
            metrics_buffer_hi.pop(0)
            
    # Save Logs
    log_df = pd.DataFrame(update_logs)
    output_file = os.path.join(OUTPUT_DIR, f'{COUNTRY}_online_updates.csv')
    log_df.to_csv(output_file, index=False)
    
    # Save Simulation Results (Time Series)
    sim_res_df = pd.DataFrame(simulation_results)
    sim_res_file = os.path.join(OUTPUT_DIR, f'{COUNTRY}_online_simulation.csv')
    sim_res_df.to_csv(sim_res_file, index=False)
    
    print(f"Simulation Complete. Logs saved to {output_file}")
    print(f"Simulation Time Series saved to {sim_res_file}")
    print(f"Total Updates: {len(log_df)}")

if __name__ == "__main__":
    run_simulation()
