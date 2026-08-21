# Cryptocurrency Volatility Forecasting: ARCH vs Deep Learning

## Overview

This repository contains a daily cryptocurrency volatility-forecasting study comparing classical conditional-volatility models with deep-learning sequence models across:

- Bitcoin (`BTCUSDT`)
- Ethereum (`ETHUSDT`)
- Solana (`SOLUSDT`)
- XRP (`XRPUSDT`)

The main research question is:

> **Do deep-learning models improve one-day-ahead cryptocurrency volatility forecasts relative to ARCH-family models when all methods are evaluated on the same daily data, forecast dates and loss functions?**

The final model set is:

- GARCH(1,1)
- GJR-GARCH(1,1)
- EGARCH(1,1)
- MLP 30D
- GRU 7D
- LSTM 14D

QLIKE is the primary volatility-forecasting loss. MAE and RMSE are reported as complementary metrics.

---

## Data

Daily OHLCV candles are pulled from the public Binance market-data endpoint.

Approximate source histories:

| Asset | Start | End | Daily rows |
|---|---|---|---:|
| BTC | 2017-08-17 | 2026-08-01 | 3,272 |
| ETH | 2017-08-17 | 2026-08-01 | 3,272 |
| XRP | 2018-05-04 | 2026-08-01 | 3,012 |
| SOL | 2020-08-11 | 2026-08-01 | 2,182 |

The project originally considered intraday volatility construction, but the final research design is fully **daily** so that all model families are evaluated under one consistent framework.

---

## Primary Forecasting Target

Daily close-to-close log return:

```text
r_t = log(C_t / C_{t-1})
```

Primary daily variance proxy:

```text
RV_t = r_t^2
```

The deep-learning target is:

```text
log(RV_t + epsilon)
```

DL predictions are exponentiated back to the variance scale before evaluation.

---

## Temporal Split

All models follow the same chronological split, assigned by **forecast target date**:

| Split | Target dates |
|---|---|
| Train | through 2024-07-31 |
| Validation | 2024-08-01 to 2025-07-31 |
| Test | 2025-08-01 to 2026-07-31 |

Validation and Test each contain 365 daily target dates per asset.

The Test period was kept unopened throughout feature engineering, lookback selection, architecture selection and robustness design. It was evaluated only in the final Train/Validation/Test notebook.

---

## Exploratory Data Analysis

The daily-return EDA supports the use of conditional-volatility models:

- returns are stationary under ADF/KPSS diagnostics;
- return distributions are strongly heavy-tailed;
- full-sample excess kurtosis is approximately 15.8 for BTC, 12.0 for ETH, 8.3 for SOL and 18.8 for XRP;
- squared and absolute returns display volatility clustering;
- Student-t innovations are therefore used for the ARCH-family models.

---

## Deep-Learning Features

The primary MLP, GRU and LSTM models use seven daily features:

```text
log_realized_variance
daily_return
absolute_daily_return
high_low_range
log_volume
rv_mean_7d
rv_mean_30d
```

### Leakage control

A sequence ending on day `t` forecasts day `t+1`.

For example, a forecast for 10 August uses features only through 9 August. Therefore `rv_mean_7d` and `rv_mean_30d` used for that forecast also end on 9 August and do **not** contain the 10 August return.

Additional controls:

- sequences require consecutive calendar days;
- `target_date = sequence_end_date + 1 day` is enforced;
- feature scalers are fitted on Train only;
- global target scalers are fitted on Train only;
- Validation/Test are never used to fit scalers;
- final Train comparisons use a common target-date intersection across the selected DL lookbacks.

---

## Deep-Learning Models

### MLP

Standalone architecture:

```text
Flatten
Dense(64, ReLU)
Dropout(0.20)
Dense(32, ReLU)
Dropout(0.10)
Dense(1)
```

### GRU

```text
GRU(32)
Dropout(0.20)
Dense(16, ReLU)
Dropout(0.10)
Dense(1)
```

### LSTM

```text
LSTM(32)
Dropout(0.20)
Dense(16, ReLU)
Dropout(0.10)
Dense(1)
```

All DL models optimize MSE on log variance with Adam and use early stopping, learning-rate reduction and restoration of the best validation weights.

---

## Why the DL Models Use Different Lookbacks

BTC was used for architecture-specific lookback selection. Each architecture was evaluated with 7D, 14D and 30D input windows.

| Architecture | Selected lookback | Best BTC validation QLIKE |
|---|---:|---:|
| MLP | 30D | 4.627 |
| GRU | 7D | 4.925 |
| LSTM | 14D | 4.739 |

The lookback is treated as an **architecture-specific hyperparameter**, not as a parameter that must be identical across architectures.

This is intentional: an MLP consumes a flattened fixed window, while GRU and LSTM explicitly model sequential state and may react differently to longer memory.

Once selected, these lookbacks were frozen. They were **not re-selected** on ETH, SOL, XRP or later robustness exercises.

---

## ARCH-Family Models

The econometric specifications are:

- GARCH(1,1)
- GJR-GARCH(1,1)
- EGARCH(1,1)

All use:

```text
mean = Constant
innovation distribution = Student-t
```

Returns are scaled by 100 during estimation and forecast variance is converted back to decimal-return units before evaluation.

For Validation and Test, model parameters are estimated using Train only, then held fixed while realized subsequent returns recursively update the conditional-volatility state.

---

## Global Multi-Asset Deep Learning

After the BTC lookbacks were frozen, MLP-30D, GRU-7D and LSTM-14D were trained on pooled BTC/ETH/SOL/XRP sequences.

The pooled setup uses:

- shared neural-network parameters across assets;
- one-hot asset identity;
- per-asset feature scalers fitted on Train only;
- per-asset target scalers fitted on Train only;
- asset-level evaluation after pooled training.

A fusion/ensemble stage was explored but was not retained because it did not consistently improve the core DL results.

---

## Evaluation Metrics

### MAE

```text
MAE = mean(|y - y_hat|)
```

### RMSE

```text
RMSE = sqrt(mean((y - y_hat)^2))
```

### QLIKE

```text
QLIKE = mean(y/y_hat - log(y/y_hat) - 1)
```

QLIKE is evaluated on the positive variance scale and is the primary ranking metric.

The project deliberately keeps MAE and RMSE as supporting metrics because a smooth model can achieve low MAE while still being poorly calibrated for variance spikes.

---

# Primary Validation Results

Best Global DL model by asset:

| Asset | Best Global DL | Validation QLIKE |
|---|---|---:|
| BTC | MLP 30D | 3.7513 |
| ETH | LSTM 14D | 5.1951 |
| SOL | GRU 7D | 2.8200 |
| XRP | LSTM 14D | 4.8030 |

The ARCH family materially outperformed Global DL under QLIKE.

Among the earlier baselines, ETH's 30-Day Mean RV baseline (`1.855280`) narrowly beat GARCH(1,1) (`1.855979`) on Validation, while ARCH models led the other assets.

---

# Final Held-Out Test Results

Notebook 14 evaluates the frozen six-model set on the previously untouched Test period.

## Test QLIKE

| Asset | Winning model | QLIKE |
|---|---|---:|
| BTC | GARCH(1,1) | 1.6059 |
| ETH | GARCH(1,1) | 1.9176 |
| SOL | EGARCH(1,1) | 1.5935 |
| XRP | GJR-GARCH(1,1) | 1.8588 |

**An ARCH-family model wins Test QLIKE for all four assets.**

## Test MAE

The ranking changes under MAE:

| Asset | Lowest-MAE model | MAE |
|---|---|---:|
| BTC | MLP 30D | 0.000465 |
| ETH | LSTM 14D | 0.001096 |
| SOL | GRU 7D | 0.001218 |
| XRP | MLP 30D | 0.001039 |

DL models therefore achieve the lowest Test MAE for all four assets, while the ARCH family also provides the lowest Test RMSE for all four assets.

This metric split is a central empirical finding: neural forecasts appear smoother and minimize typical absolute error, while ARCH models provide stronger variance calibration and large-error control.

---

## Overfitting / Underfitting Interpretation

There is no broad evidence of classical DL overfitting.

For most asset/model combinations, Test QLIKE is similar to or better than Validation QLIKE. Examples:

```text
BTC MLP   3.751 -> 3.730
BTC GRU   4.825 -> 3.839
SOL GRU   2.820 -> 2.438
XRP MLP   6.106 -> 4.796
```

The clearest weaker-generalization case is ETH MLP:

```text
Train QLIKE      4.353
Validation       5.834
Test             6.434
```

This is better described as model/regime sensitivity than project-wide overfitting.

The DL models are also not simply “underfit”: they win MAE on the final Test set. However, their QLIKE is materially worse even on Train. That points to **forecast calibration / objective mismatch** rather than insufficient network capacity. The networks were optimized with MSE on log variance, which naturally encourages smoother forecasts and may underreact to volatility spikes that QLIKE penalizes strongly.

---

# Robustness Analyses

Two distinct robustness exercises were performed.

## A. Rolling-30 Target Retraining — Notebooks 09-12

An exploratory robustness pipeline retrained models using next-day updated 30-day rolling historical variance as the target.

This exercise is valid as an alternative-target experiment and can be retained as supplementary material or an appendix.

However, subsequent supervisor clarification established that the preferred paper robustness check was to **keep the original primary models frozen** and evaluate their forecasts against historical-volatility proxies.

## B. Final Proxy-Based Robustness — Notebook 13

The final supervisor-aligned robustness analysis evaluates the original frozen forecasts against ex-post historical volatility:

```text
HV_W,t = Std(r_{t-W+1}, ..., r_t)
```

with:

```text
W = 7, 14, 30 days
```

The proxy includes the realized target-day return and is therefore an **evaluation reference only**, not a model feature or predictive baseline.

For the proxy evaluation:

- model variance is square-rooted for volatility-scale MAE/RMSE and time-series figures;
- QLIKE remains on the variance scale using `HV_W,t^2`.

Across all **12 asset x proxy-horizon combinations**, the best QLIKE model belongs to the ARCH family.

### Best QLIKE model by proxy horizon

| Proxy | BTC | ETH | SOL | XRP |
|---|---|---|---|---|
| 7D | GARCH | GARCH | EGARCH | GJR-GARCH |
| 14D | GARCH | GJR-GARCH | EGARCH | GJR-GARCH |
| 30D | GJR-GARCH | GJR-GARCH | EGARCH | EGARCH |

This strengthens the main conclusion because ARCH dominance is not tied only to the noisy single-day squared-return proxy.

---

## 30-Day Mean RV vs 30-Day Historical Volatility

These quantities are related but not identical.

30-Day Mean RV is:

```text
mean(r^2)
```

whereas historical variance is the sample variance around the rolling mean return:

```text
s^2 = W/(W-1) * (mean(r^2) - mean(r)^2)
```

The historical-volatility proxy is `sqrt(s^2)`.

There is also a timing distinction in the forecasting setup: a predictive 30-Day Mean RV baseline for 10 August uses information available through 9 August, while the ex-post 10 August historical-volatility proxy includes the realized 10 August return.

---

# Notebook Workflow

## Primary pipeline

1. `01_Data_Creation_QP.ipynb` — Binance daily OHLCV collection
2. `01_EDA_Daily.ipynb` — daily-return EDA and diagnostics
3. `02_data_processing.ipynb` — target/features, sequences, splits and scalers
4. `03_baseline_model.ipynb` — persistence and rolling-mean baselines
5. `04_ARCH_models_train.ipynb` — ARCH-family training diagnostics
6. `04_ARCH_models_validation.ipynb` — validation forecasting and ARCH selection
7. `04_MLP_model.ipynb` — BTC MLP 7D/14D/30D
8. `05_GRU_model.ipynb` — BTC GRU 7D/14D/30D
9. `06_LSTM_model.ipynb` — BTC LSTM 7D/14D/30D
10. `07_Global_MultiAsset_Models_SelectedLookbacks.ipynb` — pooled MLP-30D / GRU-7D / LSTM-14D
11. `08_Final_Model_Comparison.ipynb` — primary validation comparison

## Alternative-target robustness

12. `09_Rolling30_Data_Processing.ipynb`
13. `10_Rolling30_ARCH_Models.ipynb`
14. `11_Rolling30_DL_Models.ipynb`
15. `12_Final_Robustness_Comparison.ipynb`

## Final paper robustness and held-out evaluation

16. `13_Primary_Models_vs_Dynamic_Rolling_Volatility_Proxy_FINAL.ipynb` — frozen models evaluated against 7D/14D/30D historical-volatility proxies; dynamic metric tables and figures
17. `14_All_Models_Train_Validation_Test_Errors_FINAL.ipynb` — final Train/Validation/Test MAE, RMSE and QLIKE; opens the held-out Test period

---

# Recommended Paper Positioning

The final empirical story is:

1. Daily crypto returns are stationary but heavy-tailed and volatility-clustered.
2. Architecture-specific lookback selection gives MLP-30D, GRU-7D and LSTM-14D.
3. Global pooling increases the DL training sample but does not overturn the ARCH advantage under QLIKE.
4. On the final held-out Test set, ARCH-family models win QLIKE and RMSE for all four assets.
5. DL models win MAE for all four assets, indicating smoother forecasts with lower typical absolute error but weaker variance-risk calibration.
6. ARCH dominance under QLIKE remains robust when frozen forecasts are evaluated against 7D, 14D and 30D historical-volatility proxies.

The conclusion should therefore emphasize **forecast calibration versus smooth average-error minimization**, rather than presenting the study as a simplistic “ARCH beats deep learning” exercise.

---

# Future Work

Useful extensions include:

- higher-frequency realized-volatility measures rather than daily squared-return proxies;
- HAR-RV and realized-GARCH benchmarks;
- derivatives variables such as funding rates, open interest and liquidations;
- macro, order-book and sentiment features;
- regime-dependent or state-switching models;
- probabilistic volatility intervals and calibration;
- neural models trained directly with a QLIKE-style objective;
- broader multi-asset universes for pooled learning;
- transformer or attention-based sequence models.

Because the Test set has now been opened, any future architecture or feature tuning should be treated as a **new experiment with a new holdout period**, not as continued optimization of the current final models.

---

## Project Status

**Daily data pipeline:** complete  
**EDA:** complete  
**Primary ARCH and DL modeling:** complete  
**Multi-asset modeling:** complete  
**Validation comparison:** complete  
**Rolling-variance exploratory robustness:** complete  
**7D/14D/30D proxy robustness:** complete  
**Held-out Test evaluation:** complete  
**Current modeling project:** closed / ready for paper writing
