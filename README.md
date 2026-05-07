# 📈 GDP Growth Prediction

> Regression modelling and statistical inference on World Bank panel data across 217 countries (1960–2023).


## Overview

This project builds a machine learning pipeline to predict annual **GDP growth rates** for countries worldwide using historical World Bank indicators. Four regression models are trained on pre-2023 data and evaluated on a completely held-out 2023 test set. Statistical inference is performed using OLS regression to identify significant economic predictors and quantify their effects.

---

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Models](#models)
- [Results](#results)
- [Statistical Inference](#statistical-inference)
- [Installation](#installation)
- [Usage](#usage)
- [Key Design Decisions](#key-design-decisions)

---

## Dataset

| Property | Value |
|---|---|
| Source | [World Bank Open Data](https://data.worldbank.org/) |
| Rows | 13,888 |
| Columns | 18 |
| Coverage | 217 countries · 1960–2023 |
| Target variable | `gdp_growth` — annual GDP growth rate (%) |

**Features include:** population growth, GNI per capita, life expectancy, school enrollment, power consumption, inflation rate, agriculture/industry value added, trade flows, capital formation, FDI, and engineered lag/rolling features.

---

## Project Structure

```
gdp-prediction/
│
├── data/
│   └── gdp_data.csv                   # World Bank panel dataset
│
├── gdp-prediction.ipynb               # Main notebook (final submission)
├── gdp_growth_predictions_2023.csv    # Model predictions output
└── README.md
```

---

## Pipeline

```
Raw CSV  →  Data Cleaning  →  Feature Engineering  →  EDA
                                                         ↓
                              Export CSV  ←  Inference  ←  Modelling & Evaluation
```

### 1 · Data Cleaning
- World Bank missing value marker `".."` replaced with `NaN`
- Forward-fill → back-fill within each country's time series
- String columns (commas, `%` signs) converted to `float64`
- GDP growth capped to `[−50%, +150%]` to remove physically impossible entries
- Inflation clipped to `[−20%, +100%]` to suppress hyperinflation distortion
- Dollar-denominated columns winsorised to 1st–99th percentile

### 2 · Feature Engineering

Raw contemporaneous features had a maximum correlation of **|r| ≈ 0.14** with `gdp_growth`. Lagged features were engineered to introduce genuine predictive signal without data leakage:

| Feature | Description |
|---|---|
| `gdp_growth_lag1/2/3` | Prior 1–3 year GDP growth |
| `gdp_growth_ma3/5` | 3- and 5-year rolling mean of past growth |
| `inflation_lag1` | Prior year inflation |
| `capform_lag1` | Prior year capital formation (% GDP) |
| `trade_openness` | Exports + Imports as % of GDP |
| `gdp_log` | Log-transformed GDP (reduces right skew) |
| `gdp_per_capita` | GDP ÷ population |
| `gni_to_gdp_ratio` | GNI per capita ÷ GDP |

All lag operations use `.shift(1)` — zero leakage into the 2023 test year.

### 3 · Train / Test Split

A **strict temporal split** is used, mirroring a real forecasting scenario:

| Set | Criteria | Size |
|---|---|---|
| Training | `year < 2023` | ~13,700 rows |
| Test | `year == 2023` | ~210 rows (one per country) |

A random split was deliberately avoided because it would leak future observations into training on time-series data.

---

## Models

Four regression models are compared:

| Model | Description | Scaling Required |
|---|---|---|
| **Simple Linear Regression** | Baseline — single best predictor (`gdp_growth_lag1`) | No |
| **Ridge Regression** | L2-regularised linear model across all 24 features | Yes (StandardScaler) |
| **Random Forest** | Ensemble of 300 decision trees (bagging) | No |
| **Gradient Boosting** | Sequential ensemble of 300 shallow trees (boosting) | No |

### Key Hyperparameters

**Ridge:** `alpha=1.0`

**Random Forest:**
```python
RandomForestRegressor(n_estimators=300, max_depth=8,
                      min_samples_leaf=8, max_features=0.6,
                      random_state=42)
```

**Gradient Boosting:**
```python
GradientBoostingRegressor(n_estimators=300, max_depth=4,
                          learning_rate=0.03, subsample=0.8,
                          min_samples_leaf=10, random_state=42)
```

---

## Results

Models are evaluated on the held-out **2023 test set** using RMSE, MSE, and R². 5-fold cross-validation R² is also reported for ensemble models.

| Model | RMSE | MSE | R² (Test) |
|---|---|---|---|
| Simple LR |  6.3137 | 39.8630 | 0.0547 |
| Ridge Regression | 5.9878 | 39.8630 | 0.01498 |
| Random Forest | 5.5417 | 30.7104 | 0.2718 |
| Gradient Boosting | 4.4745 | 20.0214 | 0.5252 |

**Gradient Boosting** achieves the best overall performance. Both ensemble models substantially outperform linear models, confirming non-linear relationships between economic indicators and GDP growth.

---

## Statistical Inference

OLS regression is performed using `statsmodels` on the 7 most informative features:

```python
top_features = [
    'gdp_growth_lag1', 'gdp_growth_ma3', 'gdp_growth_ma5',
    'inflation_rate', 'capital_formation_gdp_percent',
    'population_growth', 'industry_value_added'
]
```

The full regression summary reports coefficient estimates, standard errors, t-statistics, p-values, and 95% confidence intervals. Features with **p < 0.05** are considered statistically significant.

**Key findings:**
- `gdp_growth_lag1` — strong positive effect; growth has significant autocorrelation
- `capital_formation_gdp_percent` — significant positive predictor; investment drives future output (consistent with Solow growth theory)
- `inflation_rate` — negative coefficient expected; high inflation destabilises growth

---

## Installation

```bash
git clone https://github.com/Silver-balls111/gdp-prediction.git
cd gdp-prediction
pip install -r requirements.txt
```

**Requirements:**

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
statsmodels
jupyter
```

---

## Usage

1. Place `gdp_data.csv` inside a `data/` folder in the project root
2. Launch Jupyter:
   ```bash
   jupyter notebook gdp-prediction.ipynb
   ```
3. Run all cells top-to-bottom (`Kernel → Restart & Run All`)
4. Predictions are exported to `gdp_growth_predictions_2023.csv`

---

## Key Design Decisions

**Why lagged features?**  
Contemporaneous indicators (same-year inflation, trade values) have near-zero correlation with GDP growth. Lagged features introduce genuine predictive signal — `gdp_growth_lag1` alone has |r| ≈ 0.36, more than double any raw feature.

**Why Ridge over plain Multiple Linear Regression?**  
The 24-feature set contains highly correlated variables (multiple lag and rolling features). Plain OLS becomes unstable under multicollinearity; Ridge's L2 penalty stabilises coefficient estimates while retaining all features.

**Why a temporal split?**  
Random splits allow future observations to appear in the training set, which would be data leakage. Using all pre-2023 data for training and 2023 strictly as the test year reflects a realistic forecasting scenario with no lookahead bias.

**Why statsmodels for inference?**  
`scikit-learn` provides predictions but no p-values, standard errors, or F-statistics. `statsmodels.OLS` produces a full inference summary table, which is required to assess which features are statistically significant predictors.

---

## License

This project was completed as a university assignment and is shared for educational purposes.
