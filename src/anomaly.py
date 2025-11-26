import pandas as pd
import numpy as np
import os
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
OUTPUT_DIR = config['data']['output_dir']
COUNTRIES = config['countries']
Z_WINDOW = config['anomaly_detection']['z_score_window']
Z_THRESH = config['anomaly_detection']['z_score_threshold']
CUSUM_K = config['anomaly_detection']['cusum_k']
CUSUM_H = config['anomaly_detection']['cusum_h']

def load_forecasts(country_code):
    # Load Dev and Test to have enough history for rolling stats
    dev_path = os.path.join(OUTPUT_DIR, f'{country_code}_forecasts_dev.csv')
    test_path = os.path.join(OUTPUT_DIR, f'{country_code}_forecasts_test.csv')
    
    df_dev = pd.read_csv(dev_path, parse_dates=['timestamp'])
    df_test = pd.read_csv(test_path, parse_dates=['timestamp'])
    
    df_dev['set'] = 'dev'
    df_test['set'] = 'test'
    
    # Concatenate
    df = pd.concat([df_dev, df_test]).sort_values('timestamp').reset_index(drop=True)
    return df

def compute_cusum(series, k=0.5, h=5.0):
    # CUSUM for detecting mean shifts
    # S_t+ = max(0, S_{t-1}+ + x_t - k)
    # S_t- = max(0, S_{t-1}- - x_t - k)
    # Alarm if S+ > h or S- > h
    
    s_pos = np.zeros(len(series))
    s_neg = np.zeros(len(series))
    alarms = np.zeros(len(series))
    
    series_vals = series.values
    
    for i in range(1, len(series)):
        s_pos[i] = max(0, s_pos[i-1] + series_vals[i] - k)
        s_neg[i] = max(0, s_neg[i-1] - series_vals[i] - k)
        
        if s_pos[i] > h or s_neg[i] > h:
            alarms[i] = 1
            
    return alarms

def detect_anomalies_unsupervised(df, country_code):
    # 3.1.i Compute residuals
    # et = yt - yhat
    df['resid'] = df['y_true'] - df['yhat']
    
    # 3.1.ii Rolling z-score
    # window = 336h (14d), min_periods = 168
    window = Z_WINDOW
    min_periods = window // 2
    
    # We compute rolling stats on the whole series (Dev+Test) to warm up for Test
    roll_mean = df['resid'].rolling(window=window, min_periods=min_periods).mean()
    roll_std = df['resid'].rolling(window=window, min_periods=min_periods).std()
    
    df['z_resid'] = (df['resid'] - roll_mean) / roll_std
    
    # 3.1.iii Flag anomaly if |zt| >= 3.0
    df['flag_z'] = (df['z_resid'].abs() >= Z_THRESH).astype(int)
    
    # 3.1.iv Optional CUSUM
    # k=0.5, h=5.0 on zt
    # Fill NaNs in z_resid with 0 for CUSUM calculation or skip
    z_clean = df['z_resid'].fillna(0)
    df['flag_cusum'] = compute_cusum(z_clean, k=CUSUM_K, h=CUSUM_H).astype(int)
    
    # Filter for Test set only for output
    df_test = df[df['set'] == 'test'].copy()
    
    # Save outputs/<CC>_anomalies.csv
    output_cols = ['timestamp', 'y_true', 'yhat', 'z_resid', 'flag_z', 'flag_cusum', 'lo', 'hi']
    output_file = os.path.join(OUTPUT_DIR, f'{country_code}_anomalies.csv')
    df_test[output_cols].to_csv(output_file, index=False)
    print(f"Saved unsupervised anomalies to {output_file}")
    
    return df_test

def main():
    for cc in COUNTRIES:
        print(f"Processing anomalies for {cc}...")
        df = load_forecasts(cc)
        detect_anomalies_unsupervised(df, cc)

if __name__ == "__main__":
    main()
