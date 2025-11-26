import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import warnings
import yaml

warnings.filterwarnings("ignore")

# Set device for PyTorch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
OUTPUT_DIR = config['data']['output_dir']
COUNTRIES = config['countries']
MODEL_ORDERS = {
    cc: {
        'order': tuple(config['forecasting']['model_orders'][cc]['order']),
        'seasonal_order': tuple(config['forecasting']['model_orders'][cc]['seasonal_order'])
    } for cc in COUNTRIES
}

def load_data(country_code):
    file_path = os.path.join(OUTPUT_DIR, f'{country_code}_cleaned.csv')
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.asfreq('H')
    df['load'] = df['load'].ffill()
    
    # Add exogenous features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    
    # Hour one-hots (drop first to avoid dummy trap)
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
    df['load'] = df['load'].astype(float)
    
    return df, exog

def split_data(df):
    n = len(df)
    train_end = int(n * 0.8)
    dev_end = int(n * 0.9)
    
    train = df.iloc[:train_end]
    dev = df.iloc[train_end:dev_end]
    test = df.iloc[dev_end:]
    
    return train, dev, test

def calculate_metrics(y_true, y_pred, y_train_history, seasonality=24):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    smape = 200 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))
    
    y_train_vals = y_train_history.values if hasattr(y_train_history, 'values') else y_train_history
    naive_errors = np.abs(y_train_vals[seasonality:] - y_train_vals[:-seasonality])
    d = np.mean(naive_errors)
    
    mae = mean_absolute_error(y_true, y_pred)
    mase = mae / d if d != 0 else np.nan
    
    return {
        'MASE': mase,
        'sMAPE': smape,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def run_sarima_backtest(train_df, eval_df, train_exog, eval_exog, order, seasonal_order):
    print(f"Fitting SARIMAX{order}x{seasonal_order} on training data...")
    
    fit_start_idx = max(0, len(train_df) - 2000)
    train_subset = train_df.iloc[fit_start_idx:]
    exog_subset = train_exog.iloc[fit_start_idx:]
    
    model = SARIMAX(train_subset['load'], 
                    exog=exog_subset,
                    order=order, 
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)
    
    n_steps = len(eval_df)
    stride = 24
    
    print("Starting SARIMA backtest...")
    
    current_results = results
    predictions = []
    
    for i in range(0, n_steps, stride):
        if i + stride > n_steps:
            break
            
        horizon_start = i
        horizon_end = i + stride
        
        exog_forecast = eval_exog.iloc[horizon_start:horizon_end]
        
        pred_res = current_results.get_forecast(steps=stride, exog=exog_forecast)
        yhat = pred_res.predicted_mean
        conf_int = pred_res.conf_int(alpha=0.2)
        
        chunk_dates = eval_df.index[horizon_start:horizon_end]
        chunk_actuals = eval_df['load'].iloc[horizon_start:horizon_end]
        
        for j in range(stride):
            predictions.append({
                'timestamp': chunk_dates[j],
                'y_true': chunk_actuals.iloc[j],
                'yhat': yhat.iloc[j],
                'lo': conf_int.iloc[j, 0],
                'hi': conf_int.iloc[j, 1],
                'horizon': j + 1,
                'train_end': str(train_df.index[-1])
            })
        
        new_obs = chunk_actuals
        new_exog = eval_exog.iloc[horizon_start:horizon_end]
        
        current_results = current_results.append(new_obs, exog=new_exog, refit=False)
        
    return pd.DataFrame(predictions)

def create_sequences(data, target, seq_length, horizon):
    X, y = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i:(i + seq_length)])
        y.append(target[i + seq_length : i + seq_length + horizon])
    return np.array(X), np.array(y)

# ============================================================
# PyTorch GRU Model for Direct Multi-Horizon Forecasting
# ============================================================

class GRUForecaster(nn.Module):
    """
    GRU model for direct multi-horizon forecasting.
    Input: (batch, seq_length, n_features) -> Output: (batch, horizon)
    """
    def __init__(self, n_features, hidden_size=128, num_layers=2, horizon=24, dropout=0.2):
        super(GRUForecaster, self).__init__()
        
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, horizon)
        )
        
    def forward(self, x):
        # x: (batch, seq_length, n_features)
        gru_out, _ = self.gru(x)  # gru_out: (batch, seq_length, hidden_size)
        # Take the last time step output
        last_hidden = gru_out[:, -1, :]  # (batch, hidden_size)
        output = self.fc(last_hidden)  # (batch, horizon)
        return output

def run_gru_backtest(train_df, eval_df):
    """
    GRU direct multi-horizon forecasting (168h input -> 24h output).
    MANDATORY per assignment requirements.
    """
    seq_length = 168
    horizon = 24
    
    print(f"Preparing data for GRU model (PyTorch, device={DEVICE})...")
    
    # Prepare features: load + cyclic time features (efficient for NN)
    train_data = train_df[['load']].copy()
    eval_data = eval_df[['load']].copy()
    
    # Add wind/solar if available
    for col in ['wind', 'solar']:
        if col in train_df.columns:
            train_data[col] = train_df[col].fillna(0)
            eval_data[col] = eval_df[col].fillna(0)
    
    # Add cyclic time features (better than one-hot for NNs)
    train_data['hour_sin'] = np.sin(2 * np.pi * train_df.index.hour / 24)
    train_data['hour_cos'] = np.cos(2 * np.pi * train_df.index.hour / 24)
    train_data['dow_sin'] = np.sin(2 * np.pi * train_df.index.dayofweek / 7)
    train_data['dow_cos'] = np.cos(2 * np.pi * train_df.index.dayofweek / 7)
    
    eval_data['hour_sin'] = np.sin(2 * np.pi * eval_df.index.hour / 24)
    eval_data['hour_cos'] = np.cos(2 * np.pi * eval_df.index.hour / 24)
    eval_data['dow_sin'] = np.sin(2 * np.pi * eval_df.index.dayofweek / 7)
    eval_data['dow_cos'] = np.cos(2 * np.pi * eval_df.index.dayofweek / 7)
    
    # Scale data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    
    # Create sequences for training
    X_train, y_train = create_sequences(train_scaled, train_scaled[:, 0], seq_length, horizon)
    
    n_features = X_train.shape[2]
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Build Model
    print(f"Training GRU model (input: {seq_length}h -> output: {horizon}h, features: {n_features})...")
    model = GRUForecaster(n_features=n_features, hidden_size=128, num_layers=2, horizon=horizon).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    epochs = 30
    best_loss = float('inf')
    patience_counter = 0
    patience = 5
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    
    # Backtest
    print("Starting GRU backtest...")
    eval_scaled = scaler.transform(eval_data)
    full_seq = np.concatenate([train_scaled[-seq_length:], eval_scaled], axis=0)
    
    predictions = []
    n_eval_steps = len(eval_df)
    stride = 24
    
    # Batch prediction for efficiency
    all_inputs = []
    all_indices = []
    
    for i in range(0, n_eval_steps, stride):
        if i + stride > n_eval_steps:
            break
        current_X = full_seq[i : i + seq_length]
        all_inputs.append(current_X)
        all_indices.append(i)
    
    if len(all_inputs) == 0:
        print("No valid sequences for GRU backtest.")
        return pd.DataFrame()
    
    # Predict all at once
    all_inputs = np.array(all_inputs)
    all_inputs_t = torch.FloatTensor(all_inputs).to(DEVICE)
    
    with torch.no_grad():
        all_preds_scaled = model(all_inputs_t).cpu().numpy()  # (n_batches, 24)
    
    # Inverse transform predictions
    for batch_idx, i in enumerate(all_indices):
        pred_scaled = all_preds_scaled[batch_idx]
        
        # Create dummy array for inverse transform (load is column 0)
        dummy = np.zeros((horizon, train_scaled.shape[1]))
        dummy[:, 0] = pred_scaled
        pred_inv = scaler.inverse_transform(dummy)[:, 0]
        
        # Store results
        chunk_dates = eval_df.index[i : i + stride]
        chunk_actuals = eval_df['load'].iloc[i : i + stride]
        
        for j in range(stride):
            predictions.append({
                'timestamp': chunk_dates[j],
                'y_true': chunk_actuals.iloc[j],
                'yhat': pred_inv[j],
                'lo': np.nan,  # NN doesn't provide PI natively
                'hi': np.nan,
                'horizon': j + 1,
                'train_end': str(train_df.index[-1])
            })
    
    print(f"GRU backtest complete. Generated {len(predictions)} predictions.")
    return pd.DataFrame(predictions)

def main():
    summary_metrics = []
    
    for cc in COUNTRIES:
        print(f"\n{'='*60}")
        print(f"Processing {cc}...")
        print(f"{'='*60}")
        
        df, exog = load_data(cc)
        train_df, dev_df, test_df = split_data(df)
        train_exog, dev_exog, test_exog = split_data(exog)
        
        # SARIMA
        order = MODEL_ORDERS[cc]['order']
        seasonal_order = MODEL_ORDERS[cc]['seasonal_order']
        
        # Combined Backtest (Dev+Test) - single run for efficiency
        print(f"\nRunning SARIMA Combined Backtest (Dev+Test) for {cc}...")
        combined_eval_df = pd.concat([dev_df, test_df])
        combined_eval_exog = pd.concat([dev_exog, test_exog])
        
        sarima_full_res = run_sarima_backtest(train_df, combined_eval_df, train_exog, combined_eval_exog, order, seasonal_order)
        
        # Split results back to Dev and Test
        test_start = test_df.index[0]
        
        sarima_dev_res = sarima_full_res[sarima_full_res['timestamp'] < test_start]
        sarima_test_res = sarima_full_res[sarima_full_res['timestamp'] >= test_start]
        
        # Save SARIMA results
        sarima_dev_res.to_csv(os.path.join(OUTPUT_DIR, f'{cc}_forecasts_dev.csv'), index=False)
        sarima_test_res.to_csv(os.path.join(OUTPUT_DIR, f'{cc}_forecasts_test.csv'), index=False)
        
        # SARIMA Metrics
        sarima_metrics_dev = calculate_metrics(sarima_dev_res['y_true'], sarima_dev_res['yhat'], train_df['load'])
        sarima_metrics_test = calculate_metrics(sarima_test_res['y_true'], sarima_test_res['yhat'], train_df['load'])
        
        # PI Coverage
        in_pi_dev = ((sarima_dev_res['y_true'] >= sarima_dev_res['lo']) & (sarima_dev_res['y_true'] <= sarima_dev_res['hi'])).mean() * 100
        in_pi_test = ((sarima_test_res['y_true'] >= sarima_test_res['lo']) & (sarima_test_res['y_true'] <= sarima_test_res['hi'])).mean() * 100
        
        sarima_metrics_dev['PI_Coverage_80'] = in_pi_dev
        sarima_metrics_test['PI_Coverage_80'] = in_pi_test
        
        print(f"\n{cc} SARIMA Dev Metrics: MASE={sarima_metrics_dev['MASE']:.4f}, sMAPE={sarima_metrics_dev['sMAPE']:.2f}")
        print(f"{cc} SARIMA Test Metrics: MASE={sarima_metrics_test['MASE']:.4f}, sMAPE={sarima_metrics_test['sMAPE']:.2f}")
        
        # Add SARIMA metrics to summary (Dev and Test)
        summary_metrics.append({
            'Country': cc,
            'Set': 'Dev',
            'Model': 'SARIMA',
            **sarima_metrics_dev
        })
        summary_metrics.append({
            'Country': cc,
            'Set': 'Test',
            'Model': 'SARIMA',
            **sarima_metrics_test
        })
        
        # GRU Neural Network (MANDATORY)
        print(f"\nRunning GRU Neural Network Backtest for {cc}...")
        gru_full_res = run_gru_backtest(train_df, combined_eval_df)
        
        if gru_full_res.empty:
            raise RuntimeError(f"GRU model failed for {cc}. This is mandatory!")
        
        # Split GRU results into Dev and Test
        gru_dev_res = gru_full_res[gru_full_res['timestamp'] < test_start]
        gru_test_res = gru_full_res[gru_full_res['timestamp'] >= test_start]
        
        # GRU Metrics
        gru_metrics_dev = calculate_metrics(gru_dev_res['y_true'], gru_dev_res['yhat'], train_df['load'])
        gru_metrics_test = calculate_metrics(gru_test_res['y_true'], gru_test_res['yhat'], train_df['load'])
        
        print(f"\n{cc} GRU Dev Metrics: MASE={gru_metrics_dev['MASE']:.4f}, sMAPE={gru_metrics_dev['sMAPE']:.2f}")
        print(f"{cc} GRU Test Metrics: MASE={gru_metrics_test['MASE']:.4f}, sMAPE={gru_metrics_test['sMAPE']:.2f}")
        
        # Add GRU metrics to summary
        summary_metrics.append({
            'Country': cc,
            'Set': 'Dev',
            'Model': 'GRU',
            **gru_metrics_dev
        })
        summary_metrics.append({
            'Country': cc,
            'Set': 'Test',
            'Model': 'GRU',
            **gru_metrics_test
        })
        
        # Save GRU results
        gru_dev_res.to_csv(os.path.join(OUTPUT_DIR, f'{cc}_forecasts_dev_gru.csv'), index=False)
        gru_test_res.to_csv(os.path.join(OUTPUT_DIR, f'{cc}_forecasts_test_gru.csv'), index=False)

    # Save Summary Table
    summary_df = pd.DataFrame(summary_metrics)
    
    print(f"\n{'='*60}")
    print("FORECAST COMPARISON TABLE (All Countries, Dev & Test)")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics_summary.csv'), index=False)
    print(f"\nSaved metrics summary to {os.path.join(OUTPUT_DIR, 'metrics_summary.csv')}")

if __name__ == "__main__":
    main()
