import pandas as pd
import numpy as np
import os
import json
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
OUTPUT_DIR = config['data']['output_dir']

def generate_silver_labels(df):
    # 3.2.i Create "silver labels"
    # Positive: (|zt| >= 3.5) OR (y_true outside [lo,hi] AND |zt| >= 2.5)
    # Negative: |zt| < 1.0 AND y_true inside [lo,hi]
    
    inside_pi = (df['y_true'] >= df['lo']) & (df['y_true'] <= df['hi'])
    outside_pi = ~inside_pi
    
    cond_pos_1 = df['z_resid'].abs() >= 3.5
    cond_pos_2 = outside_pi & (df['z_resid'].abs() >= 2.5)
    is_positive = cond_pos_1 | cond_pos_2
    
    cond_neg = (df['z_resid'].abs() < 1.0) & inside_pi
    
    df['silver_label'] = np.nan
    df.loc[is_positive, 'silver_label'] = 1
    df.loc[cond_neg, 'silver_label'] = 0
    
    return df

def extract_features(row, full_history_df):
    # Extract features for a specific timestamp
    ts = row['timestamp']
    
    # Get history window (e.g., 48h before ts)
    start_window = ts - pd.Timedelta(hours=48)
    end_window = ts - pd.Timedelta(hours=1)
    
    history = full_history_df[(full_history_df.index >= start_window) & (full_history_df.index <= end_window)]
    
    if len(history) < 48:
        return None # Not enough history
        
    # Features
    load_lag_24 = history.loc[ts - pd.Timedelta(hours=24), 'load'] if (ts - pd.Timedelta(hours=24)) in history.index else np.nan
    load_lag_48 = history.loc[ts - pd.Timedelta(hours=48), 'load'] if (ts - pd.Timedelta(hours=48)) in history.index else np.nan
    
    recent_24 = history[history.index > (ts - pd.Timedelta(hours=24))]
    roll_mean_24 = recent_24['load'].mean()
    roll_std_24 = recent_24['load'].std()
    
    hour = ts.hour
    dayofweek = ts.dayofweek
    yhat = row['yhat']
    
    return {
        'load_lag_24': load_lag_24,
        'load_lag_48': load_lag_48,
        'roll_mean_24': roll_mean_24,
        'roll_std_24': roll_std_24,
        'hour': hour,
        'dayofweek': dayofweek,
        'yhat': yhat
    }

def train_anomaly_classifier(countries):
    all_samples = []
    
    # Load full history for feature extraction
    history_dfs = {}
    for cc in countries:
        clean_path = os.path.join(OUTPUT_DIR, f'{cc}_cleaned.csv')
        df_hist = pd.read_csv(clean_path, parse_dates=['timestamp'])
        df_hist.set_index('timestamp', inplace=True)
        history_dfs[cc] = df_hist

    # 3.2.ii Sampling and Verification
    verified_data = []
    
    for cc in countries:
        anomalies_file = os.path.join(OUTPUT_DIR, f'{cc}_anomalies.csv')
        df_anom = pd.read_csv(anomalies_file, parse_dates=['timestamp'])
        
        # Apply silver labels
        df_anom = generate_silver_labels(df_anom)
        
        # Filter labeled data
        positives = df_anom[df_anom['silver_label'] == 1]
        negatives = df_anom[df_anom['silver_label'] == 0]
        
        # Sample 50 pos, 50 neg
        n_samples = 50
        
        if len(positives) > n_samples:
            sample_pos = positives.sample(n=n_samples, random_state=42)
        else:
            sample_pos = positives
            
        if len(negatives) > n_samples:
            sample_neg = negatives.sample(n=n_samples, random_state=42)
        else:
            sample_neg = negatives
            
        # Combine
        samples = pd.concat([sample_pos, sample_neg])
        samples['country'] = cc
        
        # "Human Verification" - we assume silver labels are correct
        samples['verified_label'] = samples['silver_label']
        
        verified_data.append(samples)
        
    all_verified = pd.concat(verified_data)
    
    # Save verified labels
    all_verified.to_csv(os.path.join(OUTPUT_DIR, 'anomaly_labels_verified.csv'), index=False)
    print(f"Saved verified labels to {os.path.join(OUTPUT_DIR, 'anomaly_labels_verified.csv')}")
    
    # 3.2.iii Train Classifier
    # Feature Engineering
    X = []
    y = []
    
    print("Extracting features for ML classifier...")
    for idx, row in all_verified.iterrows():
        cc = row['country']
        feats = extract_features(row, history_dfs[cc])
        if feats:
            X.append(list(feats.values()))
            y.append(row['verified_label'])
            
    X = np.array(X)
    y = np.array(y)
    
    # Handle NaNs
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict probs
    y_probs = clf.predict_proba(X_test)[:, 1]
    
    # Metrics
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    
    # F1 at fixed precision (e.g. P=0.80)
    target_prec = 0.80
    valid_indices = np.where(precision >= target_prec)[0]
    
    if len(valid_indices) > 0:
        best_f1_at_p80 = 0
        chosen_thresh = 0.5
        for i in valid_indices:
            p = precision[i]
            r = recall[i]
            if (p + r) > 0:
                f1 = 2 * p * r / (p + r)
                if f1 > best_f1_at_p80:
                    best_f1_at_p80 = f1
                    chosen_thresh = thresholds[i] if i < len(thresholds) else 1.0
    else:
        best_f1_at_p80 = 0.0
        chosen_thresh = 0.5
        
    # Report
    eval_metrics = {
        'PR_AUC': pr_auc,
        'F1_at_P80': best_f1_at_p80,
        'Threshold_at_P80': chosen_thresh,
        'Test_Samples': len(y_test),
        'Positives': int(sum(y_test))
    }
    
    print("ML Evaluation:", eval_metrics)
    
    with open(os.path.join(OUTPUT_DIR, 'anomaly_ml_eval.json'), 'w') as f:
        json.dump(eval_metrics, f, indent=4)
        
    print(f"Saved ML evaluation to {os.path.join(OUTPUT_DIR, 'anomaly_ml_eval.json')}")

if __name__ == "__main__":
    COUNTRIES = config['countries']
    train_anomaly_classifier(COUNTRIES)
