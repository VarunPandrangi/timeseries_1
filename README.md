# OPSD PowerDesk

Day-ahead electric load forecasting, anomaly detection, live monitoring, and dashboarding for European power grids.

## Overview

This project builds a 24-step (day-ahead) forecasting system using Open Power System Data (OPSD) hourly time series. It covers three European countries: Germany (DE), France (FR), and Spain (ES).

The pipeline includes:
- Classical SARIMA forecasting with automatic order selection
- GRU neural network for direct multi-horizon prediction
- Anomaly detection using residual z-scores and CUSUM
- Machine learning classifier for anomaly verification
- Live simulation with online model adaptation
- Interactive Streamlit dashboard

## Countries

| Code | Country |
|------|---------|
| DE | Germany |
| FR | France |
| ES | Spain |

## Project Structure

```
OPSD_PowerDesk/
├── config.yaml              # Configuration file (countries, thresholds, model parameters)
├── requirements.txt         # Python dependencies
├── run_pipeline.py          # Pipeline orchestrator (runs all scripts in sequence)
├── README.md                # This file
│
├── data/
│   ├── time_series_60min_singleindex.csv   # Raw OPSD data (not in repo, too large)
│   └── processed/                          # Preprocessed country data and plots
│       ├── DE_processed.csv
│       ├── FR_processed.csv
│       ├── IT_processed.csv
│       └── *.png                           # Sanity check and analysis plots
│
├── src/
│   ├── load_opsd.py           # Data loading and cleaning
│   ├── decompose_acf_pacf.py  # STL decomposition, ACF/PACF, SARIMA order selection
│   ├── forecast.py            # SARIMA and GRU backtesting
│   ├── anomaly.py             # Z-score and CUSUM anomaly detection
│   ├── anomaly_ml.py          # ML-based anomaly classifier
│   ├── live_loop.py           # Live simulation with Rolling SARIMA adaptation
│   ├── dashboard_app.py       # Streamlit dashboard
│   └── metrics.py             # MASE, sMAPE, MSE, RMSE, MAPE, coverage functions
│
└── outputs/
    ├── model_orders.txt                # Selected SARIMA orders for each country
    ├── metrics_summary.csv             # All forecasting metrics (Dev and Test)
    ├── anomaly_labels_verified.csv     # Human-verified anomaly labels
    ├── anomaly_ml_eval.json            # ML classifier evaluation (PR-AUC, F1)
    │
    ├── DE_cleaned.csv                  # Cleaned data for Germany
    ├── DE_forecasts_dev.csv            # SARIMA Dev set forecasts
    ├── DE_forecasts_test.csv           # SARIMA Test set forecasts
    ├── DE_forecasts_dev_gru.csv        # GRU Dev set forecasts
    ├── DE_forecasts_test_gru.csv       # GRU Test set forecasts
    ├── DE_anomalies.csv                # Detected anomalies with z-scores
    ├── DE_online_simulation.csv        # Live simulation results
    ├── DE_online_updates.csv           # Model update log (timestamp, reason, duration)
    │
    ├── FR_cleaned.csv, FR_forecasts_*, FR_anomalies.csv   # France outputs
    ├── ES_cleaned.csv, ES_forecasts_*, ES_anomalies.csv   # Spain outputs
    │
    └── plots/
        ├── DE_stl_decomposition.png    # STL trend/seasonal/residual
        ├── DE_acf_pacf_diff_1_24.png   # ACF/PACF for order selection
        ├── DE_sanity_check.png         # Last 14 days validation plot
        └── (same for FR and ES)
```

## Source Code Description

### load_opsd.py
Reads the raw OPSD CSV and creates tidy dataframes for each country. Renames columns (timestamp, load), drops missing values, and saves cleaned data.

### decompose_acf_pacf.py
Performs exploratory analysis:
- STL decomposition with period=24 (daily seasonality)
- Stationarity testing with ADF
- ACF/PACF plots up to lag 48
- Grid search over SARIMA parameters using BIC
- Saves selected orders to model_orders.txt

### forecast.py
Runs expanding-origin backtests:
- SARIMA: Uses orders from config, generates 24h forecasts with 80% prediction intervals
- GRU: PyTorch model with 168h input, 24h output, 2 layers, 128 hidden units
- Saves forecasts for Dev (10%) and Test (10%) sets
- Computes all metrics: MASE, sMAPE, MSE, RMSE, MAPE, PI coverage

### anomaly.py
Detects anomalies on Test set residuals:
- Rolling z-score with 336h window (14 days)
- Flags anomalies when |z| >= 3.0
- CUSUM detection with k=0.5, h=5.0
- Saves results with flag_z and flag_cusum columns

### anomaly_ml.py
Trains a classifier to reduce false positives:
- Creates silver labels based on z-score and PI coverage
- Samples 100 points per country for verification
- Trains Logistic Regression on lag features and calendar variables
- Reports PR-AUC and F1 at 80% precision

### live_loop.py
Simulates 2000+ hours of live operation for Germany:
- Processes data hour by hour
- Generates 24h forecasts at 00:00 UTC
- Monitors drift using EWMA of |z|
- Triggers Rolling SARIMA refit (90-day window) on schedule or drift
- Logs all updates with before/after metrics

### dashboard_app.py
Streamlit dashboard with:
- Country selector
- Last 7-14 days of actual vs predicted load
- Forecast cone with 80% prediction interval
- Anomaly tape highlighting flagged hours
- KPI tiles: rolling 7-day MASE, PI coverage, anomaly count, last update

### metrics.py
Helper functions for:
- MASE (Mean Absolute Scaled Error, seasonality=24)
- sMAPE (Symmetric Mean Absolute Percentage Error)
- MSE, RMSE, MAPE
- 80% Prediction Interval coverage

## Setup

### Requirements
- Python 3.8 or higher
- See requirements.txt for dependencies

### Installation
```bash
pip install -r requirements.txt
```

### Data
Download the OPSD hourly time series CSV and place it in:
```
data/time_series_60min_singleindex.csv
```

## Usage

### Run Full Pipeline
```bash
python run_pipeline.py
```

This runs all scripts in order:
1. load_opsd.py
2. decompose_acf_pacf.py
3. forecast.py
4. anomaly.py
5. anomaly_ml.py
6. live_loop.py

### Run Individual Scripts
```bash
python src/load_opsd.py
python src/decompose_acf_pacf.py
python src/forecast.py
python src/anomaly.py
python src/anomaly_ml.py
python src/live_loop.py
```

### Launch Dashboard
```bash
streamlit run src/dashboard_app.py
```

## Results

### SARIMA Orders (Selected via BIC)

| Country | Order (p,d,q) | Seasonal (P,D,Q,s) |
|---------|---------------|-------------------|
| DE | (2, 1, 2) | (1, 1, 1, 24) |
| FR | (1, 1, 2) | (1, 1, 1, 24) |
| ES | (2, 1, 2) | (1, 1, 1, 24) |

### Forecasting Performance

#### Test Set Results

| Country | Model | MASE | sMAPE | RMSE | PI Coverage (80%) |
|---------|-------|------|-------|------|-------------------|
| DE | SARIMA | 0.539 | 5.19% | 3496 | 90.4% |
| DE | GRU | 0.524 | 4.65% | 3474 | - |
| FR | SARIMA | 0.705 | 4.84% | 3469 | 80.6% |
| FR | GRU | 0.569 | 3.97% | 2906 | - |
| ES | SARIMA | 0.637 | 4.52% | 1567 | 78.8% |
| ES | GRU | 0.664 | 4.63% | 1560 | - |

MASE below 1.0 means the model outperforms a seasonal naive baseline. Both SARIMA and GRU achieve this for all countries.

#### Dev Set Results

| Country | Model | MASE | sMAPE | RMSE | PI Coverage (80%) |
|---------|-------|------|-------|------|-------------------|
| DE | SARIMA | 0.608 | 5.24% | 3796 | 86.6% |
| DE | GRU | 0.626 | 4.91% | 3662 | - |
| FR | SARIMA | 0.828 | 4.60% | 3508 | 72.7% |
| FR | GRU | 0.753 | 4.09% | 2977 | - |
| ES | SARIMA | 0.722 | 4.68% | 1841 | 76.2% |
| ES | GRU | 0.650 | 3.97% | 1545 | - |

### Anomaly Detection

#### ML Classifier Performance
- PR-AUC: 0.822
- F1 at 80% Precision: 0.649
- Test Samples: 88 (44 positives, 44 negatives)

### Live Simulation (Germany)

- Total Hours Simulated: 2001
- Adaptation Strategy: Rolling SARIMA (90-day window)
- Update Triggers: Scheduled (daily at 00:00) and drift-based
- Total Updates: 132 (scheduled + drift triggers)

The system successfully detects drift events and triggers retraining to maintain forecast accuracy.

## Configuration

All parameters are in config.yaml:

```yaml
countries: [DE, FR, ES]
forecasting:
  horizon: 24
  train_ratio: 0.8
  dev_ratio: 0.1
  test_ratio: 0.1
anomaly:
  z_score_window: 336
  z_score_threshold: 3.0
  cusum_k: 0.5
  cusum_h: 5.0
live:
  start_history_days: 120
  min_simulation_hours: 2000
  drift_alpha: 0.1
  drift_percentile: 95
```

## Limitations

- GRU does not provide prediction intervals (only point forecasts)
- Live simulation runs on historical data, not real-time feeds
- SARIMA refitting is computationally expensive
- Wind and solar features are optional and may have missing values
- Model performance may degrade during holidays or extreme weather events

## Dependencies

Key packages:
- pandas, numpy
- statsmodels (SARIMAX)
- torch (PyTorch GRU)
- scikit-learn
- streamlit
- matplotlib

See requirements.txt for full list.
