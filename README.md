# OPSD PowerDesk: Day-Ahead Forecasting & Anomaly Detection

## 1. Introduction
This project implements a robust, production-ready pipeline for **Day-Ahead Electric Load Forecasting** and **Anomaly Detection** for three major European power grids: **Germany (DE)**, **France (FR)**, and **Spain (ES)**.

The system is designed to handle real-world challenges such as seasonality, trend shifts, and data irregularities. It leverages the **Open Power System Data (OPSD)** platform and employs a hybrid approach combining classical statistical methods (SARIMA) with modern machine learning techniques (GRU, Logistic Regression) to ensure high accuracy and reliability.

Key capabilities include:
*   **24-hour Horizon Forecasting**: Accurate hourly load predictions for the next day.
*   **Real-time Anomaly Detection**: Identification of abnormal load patterns using statistical and ML classifiers.
*   **Online Adaptation**: A simulated live environment that automatically retrains models in response to concept drift or scheduled updates.
*   **Interactive Dashboard**: A Streamlit-based interface for monitoring grid status and model performance.

## 2. Repository Structure
The project is organized as follows:

```
OPSD_PowerDesk/
├── config.yaml                 # Global configuration (countries, thresholds, params)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/                       # Raw input data
│   └── time_series_60min_singleindex.csv
├── outputs/                    # Generated artifacts (Cleaned data, Forecasts, Plots, Models)
│   ├── *_cleaned.csv           # Preprocessed time series
│   ├── *_forecasts_*.csv       # Forecast results (Dev/Test)
│   ├── *_anomalies.csv         # Detected anomalies
│   ├── *_online_simulation.csv # Live simulation results
│   ├── metrics_summary.csv     # Final model evaluation metrics
│   └── plots/                  # EDA and diagnostic plots
└── src/                        # Source code
    ├── load_opsd.py            # Data ingestion & cleaning
    ├── decompose_acf_pacf.py   # EDA: STL, Stationarity, Order Selection
    ├── forecast.py             # Backtesting (SARIMA & GRU)
    ├── anomaly.py              # Unsupervised Anomaly Detection (Z-score, CUSUM)
    ├── anomaly_ml.py           # Supervised Anomaly Classification (Logistic Regression)
    ├── live_loop.py            # Live Simulation with Online Learning
    ├── dashboard_app.py        # Interactive Streamlit Dashboard
    └── metrics.py              # Evaluation metrics (MASE, sMAPE, Coverage)
```

## 3. Methodology

### 3.1 Data Ingestion & Preprocessing
**Script:** `src/load_opsd.py`

The raw data is sourced from the OPSD Time Series dataset. The pipeline performs the following steps:
1.  **Extraction**: Loads hourly electricity consumption (`load`), wind generation (`wind`), and solar generation (`solar`) for DE, FR, and ES.
2.  **Cleaning**:
    *   Renames columns to a standardized format.
    *   Drops rows with missing load values.
    *   Sorts data chronologically by UTC timestamp.
3.  **Output**: Saves tidy CSV files (e.g., `DE_cleaned.csv`) to the `outputs/` directory.

### 3.2 Exploratory Data Analysis (EDA)
**Script:** `src/decompose_acf_pacf.py`

Before modeling, we analyze the time series characteristics:
1.  **STL Decomposition**: Decomposes the series into **Seasonal** (24h), **Trend**, and **Residual** components to understand underlying patterns.
2.  **Stationarity Test**: Applies the **Augmented Dickey-Fuller (ADF)** test. If the series is non-stationary ($p > 0.05$), differencing ($d=1$ or $D=1$) is applied.
3.  **Autocorrelation**: Plots **ACF** and **PACF** to identify significant lags, guiding the selection of SARIMA $(p, q, P, Q)$ parameters.
4.  **Model Selection**: Performs a grid search over SARIMA parameters, selecting the configuration that minimizes the **Bayesian Information Criterion (BIC)**.

### 3.3 Forecasting Models
**Script:** `src/forecast.py`

We employ an **Expanding Window Backtest** strategy (Train: 80%, Dev: 10%, Test: 10%) to evaluate model performance.

#### A. SARIMA (Seasonal AutoRegressive Integrated Moving Average)
*   **Configuration**: Orders $(p,d,q) \times (P,D,Q)_{24}$ are determined per country from the EDA phase.
*   **Exogenous Variables**: Includes One-Hot Encoded Hour-of-Day and Day-of-Week, plus Wind and Solar generation.
*   **Strategy**: The model forecasts the next 24 hours, then "observes" the actual data to update its state for the next step (without full retraining).

#### B. GRU (Gated Recurrent Unit) - *Optional*
*   **Architecture**: A deep learning model with a GRU layer (64 units) followed by Dense layers.
*   **Input**: Sequences of the past 168 hours (1 week).
*   **Output**: Direct multi-step forecast for the next 24 hours.
*   **Scaling**: MinMax scaling is applied to normalize inputs.

**Evaluation Metrics**:
*   **MASE (Mean Absolute Scaled Error)**: Measures accuracy relative to a naive seasonal forecast.
*   **sMAPE (Symmetric Mean Absolute Percentage Error)**: Percentage error robust to near-zero values.
*   **PI Coverage**: Percentage of actuals falling within the 80% Prediction Interval.

### 3.4 Anomaly Detection
**Script:** `src/anomaly.py` & `src/anomaly_ml.py`

#### Unsupervised Detection
We analyze the residuals ($e_t = y_t - \hat{y}_t$) from the SARIMA model:
1.  **Rolling Z-Score**: Calculates $z_t = \frac{e_t - \mu_{rolling}}{\sigma_{rolling}}$ over a 336-hour window.
    *   **Flag**: Anomaly if $|z_t| > 3.0$.
2.  **CUSUM (Cumulative Sum)**: Detects persistent shifts in the mean of residuals.

#### Supervised Classification (Machine Learning)
To reduce false positives, we train a **Logistic Regression** classifier:
1.  **Silver Labels**: We generate heuristic labels to create a training set:
    *   *Positive (1)*: High Z-score ($>3.5$) OR (Outside PI AND Z-score $>2.5$).
    *   *Negative (0)*: Low Z-score ($<1.0$) AND Inside PI.
2.  **Features**: Lagged load values (24h, 48h), rolling mean/std, hour of day, day of week.
3.  **Evaluation**: The model is evaluated using **Precision-Recall AUC** and **F1-score** at a fixed precision of 80%.

### 3.5 Live Simulation & Online Learning
**Script:** `src/live_loop.py`

Simulates a production environment for Germany (DE):
1.  **Stream**: Iterates through the Test set hour-by-hour.
2.  **Forecast**: Generates a 24h forecast at 00:00 UTC daily.
3.  **Monitoring**: Tracks drift using an Exponentially Weighted Moving Average (EWMA) of the Z-score.
4.  **Online Adaptation**:
    *   **Trigger**: Scheduled (daily) or Drift Detected (EWMA > threshold).
    *   **Action**: Retrains the SARIMA model on the most recent 90 days of data.
    *   **Logging**: Records performance metrics (MASE) before and after retraining to quantify improvement.

## 4. Setup & Usage

### 4.1 Prerequisites
Ensure you have Python 3.8+ installed. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4.2 Execution Pipeline
Run the following commands in order to reproduce the full analysis:

**Step 1: Data Preparation**
```bash
python src/load_opsd.py
```

**Step 2: EDA & Model Selection**
```bash
python src/decompose_acf_pacf.py
```

**Step 3: Forecasting (Backtest)**
```bash
python src/forecast.py
```

**Step 4: Anomaly Detection**
```bash
python src/anomaly.py
python src/anomaly_ml.py
```

**Step 5: Live Simulation**
```bash
python src/live_loop.py
```

### 4.3 Dashboard
Launch the interactive dashboard to visualize the live simulation results:
```bash
streamlit run src/dashboard_app.py
```

## 5. Results

### 5.1 Model Selection & Parameters
The following SARIMA configurations were selected based on the lowest BIC during the grid search phase:

| Country | Order $(p,d,q)$ | Seasonal $(P,D,Q)_{24}$ | BIC | AIC |
| :--- | :--- | :--- | :--- | :--- |
| **Germany (DE)** | $(2, 0, 1)$ | $(1, 1, 1)$ | 10098.36 | 10071.76 |
| **France (FR)** | $(2, 0, 2)$ | $(1, 1, 1)$ | 9963.65 | 9932.63 |
| **Spain (ES)** | $(1, 0, 2)$ | $(1, 1, 1)$ | 9045.93 | 9019.34 |

### 5.2 Forecasting Performance (Test Set)
The models were evaluated on the held-out Test set (last 10% of data). The **GRU** model generally outperformed SARIMA in terms of point accuracy (MASE, sMAPE), likely due to its ability to capture non-linear interactions. However, SARIMA provides valuable probabilistic outputs (Prediction Intervals).

| Country | Model | MASE | sMAPE (%) | RMSE (MW) | PI Coverage (80%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DE** | SARIMA | **0.539** | 5.19 | 3496 | **90.4%** |
| **DE** | GRU | 0.351 | 3.07 | 2201 | N/A |
| **FR** | SARIMA | 0.800 | 5.44 | 3862 | 69.4% |
| **FR** | GRU | 0.518 | 3.63 | 2674 | N/A |
| **ES** | SARIMA | 0.635 | 4.50 | 1565 | 79.0% |
| **ES** | GRU | 0.454 | 3.21 | 1089 | N/A |

*Note: MASE < 1 indicates the model outperforms a seasonal naive baseline.*

### 5.3 Anomaly Detection (ML Classifier)
The Logistic Regression classifier, trained on "silver" labels, achieved robust performance in distinguishing true anomalies from noise.

*   **Precision-Recall AUC**: 0.821
*   **F1-Score (at 80% Precision)**: 0.753
*   **Test Samples**: 89 (44 Positives)

### 5.4 Live Simulation (Germany)
The online simulation demonstrated the system's ability to adapt to changing conditions.
*   **Total Updates**: Multiple retraining events triggered by schedule (daily) and concept drift.
*   **Adaptation Impact**: For example, on **2020-03-12**, a scheduled retrain reduced the 7-day rolling MASE from **0.588** to **0.466**, significantly improving forecast accuracy.
*   **Drift Detection**: The system successfully identified drift events (e.g., on 2020-04-10) and triggered ad-hoc retraining.

All detailed logs are available in `outputs/DE_online_simulation.csv` and `outputs/DE_online_updates.csv`.

## 6. References
1.  Open Power System Data. (2020). Data Package Time Series.
2.  Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice. OTexts.
