# Cryptocurrency Volatility Forecasting: ARCH vs Deep Learning

## Overview

This repository contains an empirical study of **daily cryptocurrency volatility forecasting** using classical conditional-volatility models and deep-learning models.

The project compares:

- **GARCH(1,1)**
- **GJR-GARCH(1,1)**
- **EGARCH(1,1)**
- **MLP**
- **GRU**
- **LSTM**
- Simple volatility baselines

across four major crypto assets:

- Bitcoin (`BTCUSDT`)
- Ethereum (`ETHUSDT`)
- Solana (`SOLUSDT`)
- XRP (`XRPUSDT`)

The core research question is:

> **Do deep-learning models improve one-day-ahead cryptocurrency volatility forecasts relative to ARCH-family models and simple historical-volatility baselines when all methods are evaluated on the same data, target dates and loss functions?**

A second robustness experiment tests whether the conclusion changes when volatility is defined using a **30-day rolling standard deviation of daily returns** instead of squared daily returns.

---

## Project Motivation

Cryptocurrency returns exhibit several features commonly associated with financial volatility:

- volatility clustering;
- heavy-tailed return distributions;
- rapidly changing market regimes;
- large jumps and extreme observations;
- nonlinear temporal dependence.

These characteristics make crypto a natural setting for comparing traditional econometric volatility models with neural-network approaches.

Rather than assuming that a more complex model should outperform a classical model, this project evaluates all model families under a common out-of-sample framework.

---

# Data

## Source

Daily OHLCV market data is collected from the public **Binance** market-data endpoint.

Assets:

| Asset | Trading Pair |
|---|---|
| Bitcoin | BTCUSDT |
| Ethereum | ETHUSDT |
| Solana | SOLUSDT |
| XRP | XRPUSDT |

The data is stored as one daily observation per asset.

Approximate available history in the project dataset:

| Asset | Start |
|---|---|
| BTC | 2017-08 |
| ETH | 2017-08 |
| XRP | 2018-05 |
| SOL | 2020-08 |

The common project data extends through **2026-08-01**.

---

# Return Definition

Daily close-to-close log returns are defined as

\[
r_t = \ln\left(\frac{C_t}{C_{t-1}}\right)
\]

where \(C_t\) is the daily closing price.

Returns are calculated only across consecutive calendar days.

---

# Volatility Definitions

The project contains two separate experiments.

## 1. Primary Volatility Target — Squared Daily Return

The primary daily variance proxy is

\[
RV_t = r_t^2
\]

with corresponding volatility proxy

\[
|r_t|.
\]

For deep-learning models, the modeled target is

\[
\log(RV_t + \epsilon).
\]

Predictions are exponentiated back to the variance scale before MAE, RMSE and QLIKE are calculated.

This experiment asks:

> **How well can the models forecast the next day's variance proxy?**

---

## 2. Robustness Target — 30-Day Rolling Historical Volatility

The alternative volatility definition is

\[
HV_{30,t}
=
\operatorname{Std}(r_{t-29},\ldots,r_t),
\]

using the sample standard deviation (`ddof=1`).

The variance-scale target is

\[
HV^2_{30,t},
\]

and the deep-learning target is

\[
\log(HV^2_{30,t}+\epsilon).
\]

This experiment asks:

> **How well can the models forecast the next day's updated 30-day historical-variance state?**

Because adjacent 30-day windows share 29 of 30 observations, this target is substantially smoother than \(r_t^2\).

**Metric magnitudes should therefore not be compared directly across the two target definitions.** Model rankings are compared within each target definition.

---

# Experimental Design

All models use the same temporal split.

| Split | Target Dates |
|---|---|
| Train | Through 2024-07-31 |
| Validation | 2024-08-01 to 2025-07-31 |
| Test | 2025-08-01 to 2026-07-31 |

The split is assigned by **forecast target date**, not by feature date.

This ensures a sequence ending on day \(t\) forecasts day \(t+1\).

## Current reporting status

The results documented below are **validation-period results**.

The held-out test period has intentionally remained untouched during:

- feature development;
- lookback selection;
- architecture comparison;
- ARCH specification comparison;
- robustness analysis.

For a paper-grade final performance estimate, the frozen specifications should be evaluated once on the held-out test period without further model selection.

---

# Deep-Learning Features

The primary daily experiment uses seven features:

1. `log_realized_variance`
2. `daily_return`
3. `absolute_daily_return`
4. `high_low_range`
5. `log_volume`
6. `rv_mean_7d`
7. `rv_mean_30d`

For the rolling-30 robustness experiment, the variance-state features are replaced with:

1. `log_rolling_30d_variance`
2. `daily_return`
3. `absolute_daily_return`
4. `high_low_range`
5. `log_volume`
6. `rolling30_var_mean_7d`
7. `rolling30_var_mean_30d`

Rolling features require consecutive calendar observations.

Missing observations required by a sequence are not imputed.

---

# Data Leakage Controls

Several controls are applied throughout the project:

- sequences are constructed chronologically;
- target date is always one day after sequence end;
- feature scaling is fitted on **training data only**;
- validation and test are transformed using the training scaler;
- no future volatility observation enters a feature window;
- test data is not used for lookback or model selection;
- deep-learning training uses `shuffle=False`.

---

# Models

## Baselines

### Persistence

For the primary target:

\[
\widehat{RV}_{t+1}=RV_t.
\]

For the rolling-30 target:

\[
\widehat{HV^2}_{30,t+1}=HV^2_{30,t}.
\]

### Historical means

The project also evaluates:

- 7-day mean variance;
- 30-day mean variance.

For the rolling-30 robustness experiment these are calculated from the rolling-30 variance state.

---

# ARCH-Family Models

Three classical volatility specifications are used:

- GARCH(1,1)
- GJR-GARCH(1,1)
- EGARCH(1,1)

Student-t innovations are used to accommodate heavy-tailed crypto returns.

Daily returns are scaled by 100 during ARCH estimation and converted back to decimal-return variance for evaluation.

## Forecasting Design

Parameters are estimated using the training period.

During validation:

1. parameters remain fixed;
2. observed returns sequentially update the volatility state;
3. one-step-ahead conditional variance is produced.

This provides genuine out-of-sample recursive volatility forecasts.

---

## ARCH Forecasts for the Rolling-30 Target

A standard GARCH forecast is the conditional variance of the next return and is not directly equivalent to 30-day rolling variance.

For the robustness experiment, the GARCH next-return mean and variance are analytically transformed into the expected next 30-day sample variance.

For 29 already-observed returns, let

\[
A=\sum_{i=1}^{29}r_i,
\qquad
B=\sum_{i=1}^{29}r_i^2.
\]

If the ARCH model forecasts next-return mean \(\mu\) and variance \(\sigma^2\), then

\[
E[s^2_{30}]
=
\frac{
B+\sigma^2+\mu^2
-
\frac{
A^2+2A\mu+\sigma^2+\mu^2
}{30}
}{29}.
\]

The implementation includes an identity test showing that replacing the forecast distribution with the actual next return reconstructs the saved rolling-30 sample variance to floating-point precision.

---

# Deep-Learning Models

## MLP

The standalone MLP uses:

- sequence flattening;
- Dense(64, ReLU);
- Dropout;
- Dense(32, ReLU);
- Dropout;
- linear output.

## GRU

The GRU model uses:

- GRU(32);
- Dropout;
- Dense(16, ReLU);
- Dropout;
- linear output.

## LSTM

The LSTM model uses:

- LSTM(32);
- Dropout;
- Dense(16, ReLU);
- Dropout;
- linear output.

All models optimize MSE on the log-variance target using Adam.

Training includes:

- early stopping;
- learning-rate reduction;
- restoration of the best validation weights;
- fixed random seeds.

---

# Lookback Selection

The primary BTC experiment tested:

- 7 days
- 14 days
- 30 days

for MLP, GRU and LSTM.

The selected validation lookbacks were:

| Architecture | Selected Lookback |
|---|---:|
| MLP | 30 days |
| GRU | 7 days |
| LSTM | 14 days |

These specifications were **frozen** for the robustness experiment rather than re-tuned on the alternative volatility target.

---

# Global Multi-Asset Models

A second DL experiment pools BTC, ETH, SOL and XRP observations.

The global models retain the selected architecture-specific lookbacks:

- Global MLP — 30d
- Global GRU — 7d
- Global LSTM — 14d

The pooled framework uses:

- per-asset feature scalers fitted on training observations;
- per-asset target scalers fitted on training targets;
- one-hot asset identity;
- shared model parameters across assets;
- asset-level validation evaluation.

A previously explored fusion/ensemble stage was not retained in the final pipeline because it did not provide consistent improvements over the standalone/global DL models.

---

# Evaluation Metrics

## Mean Absolute Error

\[
MAE=
\frac{1}{N}
\sum_{t=1}^{N}
|y_t-\hat y_t|
\]

## Root Mean Squared Error

\[
RMSE=
\sqrt{
\frac{1}{N}
\sum_{t=1}^{N}
(y_t-\hat y_t)^2
}
\]

## QLIKE

The primary volatility-forecasting loss is

\[
QLIKE
=
\frac{1}{N}
\sum_{t=1}^{N}
\left[
\frac{y_t}{\hat y_t}
-
\log\left(
\frac{y_t}{\hat y_t}
\right)
-1
\right].
\]

Lower values are better.

QLIKE is calculated on the **variance scale**, not the log-variance scale.

---

# Main Validation Results

## Primary Target: Daily Squared Return Variance

Best model by validation QLIKE:

| Asset | Winning Model | QLIKE |
|---|---|---:|
| BTC | GARCH(1,1) | 1.865873 |
| ETH | 30-Day Mean | 1.855280 |
| SOL | GARCH(1,1) | 1.650260 |
| XRP | EGARCH(1,1) | 1.914143 |

For ETH, the 30-day historical-mean baseline narrowly outperforms GARCH(1,1):

- 30-Day Mean: `1.855280`
- GARCH(1,1): `1.855979`

The two are effectively very close in QLIKE terms.

### Best Global DL Models

| Asset | Best Global DL | QLIKE |
|---|---|---:|
| BTC | Global MLP-30d | 3.751286 |
| ETH | Global LSTM-14d | 5.195097 |
| SOL | Global GRU-7d | 2.820048 |
| XRP | Global LSTM-14d | 4.802991 |

The primary experiment therefore provides little evidence that the tested DL architectures improve QLIKE relative to the strongest ARCH/baseline alternatives.

---

# Rolling-30 Robustness Results

Best model by validation QLIKE:

| Asset | Winning Model | QLIKE |
|---|---|---:|
| BTC | GARCH(1,1) | 0.002567 |
| ETH | GARCH(1,1) | 0.003161 |
| SOL | GARCH(1,1) | 0.002666 |
| XRP | EGARCH(1,1) | 0.004998 |

ARCH-family models therefore win **all four assets** under the alternative volatility definition.

### Best Global DL

Global GRU-7d is the strongest Global DL specification for every asset:

| Asset | Global GRU-7d QLIKE |
|---|---:|
| BTC | 0.004830 |
| ETH | 0.005485 |
| SOL | 0.005077 |
| XRP | 0.010544 |

### Persistence Benchmark

| Asset | Rolling-30 Persistence QLIKE |
|---|---:|
| BTC | 0.004915 |
| ETH | 0.005222 |
| SOL | 0.004887 |
| XRP | 0.009537 |

Global GRU slightly improves over persistence for BTC, while persistence remains stronger for ETH, SOL and XRP.

---

# Robustness Conclusion

Across the two target definitions there are eight asset-target combinations.

ARCH-family models win **7 of 8**.

The only exception is ETH under the primary squared-return target, where the 30-day historical-mean baseline narrowly beats GARCH.

Winning-family stability:

| Asset | Primary Target | Rolling-30 Target | Stable? |
|---|---|---|---|
| BTC | ARCH | ARCH | Yes |
| ETH | Baseline | ARCH | No |
| SOL | ARCH | ARCH | Yes |
| XRP | ARCH | ARCH | Yes |

Thus the winning model family is unchanged for **3 of 4 assets**, and the broad conclusion that conditional-volatility models outperform the tested deep-learning approaches is robust to the alternative volatility definition.

The rolling-30 experiment also demonstrates that the conclusion is not solely driven by the noisiness of \(r_t^2\).

---

# Important Interpretation Note

QLIKE values from the two target definitions should **not** be compared directly.

For example, a rolling-30 QLIKE of `0.003` is not evidence that the model is hundreds of times more accurate than a squared-return model with QLIKE near `1.8`.

The rolling-30 target is much smoother because adjacent targets share 29 of 30 return observations.

Cross-target conclusions should therefore be based on:

- model rankings;
- winning model families;
- within-target relative performance;
- robustness of conclusions.

---

# Repository Workflow

A typical execution order is:

## Primary Experiment

1. **Data Creation**  
   Pull daily Binance data for BTC, ETH, SOL and XRP.

2. **Daily Data Processing**  
   Calculate returns, variance proxy, features, sequences and leakage-safe splits.

3. **Baselines**  
   Persistence, 7-day mean and 30-day mean.

4. **EDA / ARCH Diagnostics**  
   Return-distribution and volatility-dependence analysis.

5. **ARCH Training / Validation**  
   GARCH, GJR-GARCH and EGARCH.

6. **MLP**  
   BTC 7d / 14d / 30d.

7. **GRU**  
   BTC 7d / 14d / 30d.

8. **LSTM**  
   BTC 7d / 14d / 30d.

9. **Global Multi-Asset Models**  
   Selected MLP / GRU / LSTM specifications across all four assets.

10. **Final Primary Comparison**  
    Combine ARCH, DL and baseline validation metrics.

## Robustness Experiment

11. **Rolling-30 Data Processing**  
    `09_Rolling30_Data_Processing.ipynb`

12. **Rolling-30 ARCH Models**  
    `10_Rolling30_ARCH_Models.ipynb`

13. **Rolling-30 DL Models**  
    `11_Rolling30_DL_Models.ipynb`

14. **Final Robustness Comparison**  
    `12_Final_Robustness_Comparison.ipynb`

---

# Suggested Repository Structure

```text
.
├── notebooks/
│   ├── primary/
│   │   ├── data_creation/
│   │   ├── data_processing/
│   │   ├── baselines/
│   │   ├── arch/
│   │   ├── mlp/
│   │   ├── gru/
│   │   ├── lstm/
│   │   ├── global_models/
│   │   └── final_comparison/
│   │
│   └── robustness/
│       ├── 09_Rolling30_Data_Processing.ipynb
│       ├── 10_Rolling30_ARCH_Models.ipynb
│       ├── 11_Rolling30_DL_Models.ipynb
│       └── 12_Final_Robustness_Comparison.ipynb
│
├── data/
│   ├── raw/
│   └── model_ready/
│
├── results/
│   ├── primary/
│   └── rolling30_historical_variance/
│
├── models/
├── manifests/
└── README.md
```

Large raw datasets, trained neural-network files and intermediate arrays do not necessarily need to be committed to GitHub. They can be regenerated using the notebooks.

---

# Running the Project

The notebooks were developed for **Google Colab** and use a Google Drive project root similar to:

```python
/content/drive/MyDrive/Quant Research
```

If running locally or from another Drive location, update the project root paths in the notebooks.

## Main Python Dependencies

```text
numpy
pandas
scikit-learn
tensorflow
arch
joblib
matplotlib
pyarrow
openpyxl
requests
```

Install additional notebook-specific dependencies where required.

---

# Reproducibility

The project uses:

- fixed temporal splits;
- fixed model seeds where supported;
- train-only feature scaling;
- deterministic sequence construction;
- explicit dataset configuration files;
- saved validation predictions and metrics;
- data manifests;
- model-specific result directories.

The robustness pipeline is stored separately from the primary pipeline so alternative-target artifacts do not overwrite the original results.

---

# Current Research Takeaway

The empirical evidence from this project favors **parsimonious conditional-volatility models** over the tested deep-learning architectures for one-day-ahead crypto volatility forecasting.

Key observations are:

1. GARCH-family models are consistently strong under QLIKE.
2. Neural models can obtain competitive or lower MAE in some cases while still performing poorly under QLIKE.
3. Pooling multiple crypto assets improves some DL forecasts, but does not overturn the ARCH advantage.
4. The same broad result persists under a much smoother 30-day historical-volatility target.
5. Increased model complexity does not automatically translate into superior volatility forecasts.

This makes the project a useful empirical comparison of **econometric structure versus neural-network flexibility** in cryptocurrency volatility forecasting.

---

# Limitations

Important limitations include:

- only four cryptocurrencies are considered;
- the study uses daily data;
- squared daily returns are a noisy proxy for latent daily variance;
- the rolling-30 target has strong mechanical overlap across adjacent observations;
- the DL architecture search is intentionally limited rather than exhaustive;
- macroeconomic, derivatives, order-book and sentiment variables are not included;
- the currently documented headline tables are validation results rather than final held-out test estimates.

---

# Potential Extensions

Possible future work includes:

- final evaluation on the untouched test period;
- higher-frequency realized-volatility measures;
- HAR-RV and realized-GARCH models;
- exogenous macro / market / sentiment features;
- regime-dependent volatility models;
- transformer-based sequence models;
- broader cross-asset training universes;
- probabilistic volatility forecasts and interval calibration.

---

## Project Status

**Primary modeling:** complete  
**Multi-asset modeling:** complete  
**Rolling-30 robustness analysis:** complete  
**Validation comparison:** complete  
**Held-out test evaluation:** reserved / not yet used
