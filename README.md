# 🏦 Synthetic Financial Data

> Generate synthetic OHLCV (Open, High, Low, Close, Volume) financial time series with four generative model architectures, evaluate them honestly against real data, and generate labeled structural-break datasets for testing break detectors.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**synfin** is a Python package for generating and evaluating synthetic stock market data.
It implements **four architectures** you can train and compare:

| Model | Description | Reference |
|-------|-------------|-----------|
| **TimeGAN** | Recurrent GAN with temporal supervision | Yoon et al., NeurIPS 2019 |
| **Diffusion (DDPM)** | 1D U-Net denoising diffusion model | Ho et al., NeurIPS 2020 |
| **Diffusion-TS** | Transformer diffusion with trend/seasonal decomposition | Yuan & Qiao, ICLR 2024 |
| **VAE + Copula** | Variational Autoencoder with copula dependency modeling | Kingma & Welling 2014 |

The evaluation suite checks whether generated data reproduces the key **stylized facts** of financial returns:
- 📊 Fat-tailed return distributions
- 📈 Volatility clustering
- 🔗 Volume-volatility correlation
- 📉 Leverage effect

---

## Architecture Diagrams

### TimeGAN (3-Phase Training)

```
Phase 1 — Autoencoder:   X ──→ Embedder ──→ H ──→ Recovery ──→ X̂
Phase 2 — Supervisor:         H ──→ Supervisor ──→ Ŝ
Phase 3 — Joint:         Z ──→ Generator ──→ Ê ──→ Supervisor ──→ H_hat
                               Discriminator(H_real vs H_hat) adversarial loss
```

### Diffusion Model (DDPM)

```
Training:   x_0 ──→[add noise t steps]──→ x_t ──→[UNet1D]──→ ε_pred  (MSE loss vs ε)
Sampling:   x_T ~ N(0,I) ──→[denoise T steps]──→ x_0
```

### VAE + Copula

```
Encoder:  X ──→[LSTM]──→ (μ, σ²)  ──→[reparameterize]──→ z
Decoder:  z ──→[LSTM]──→ X̂
Copula:   μ(X_train) ──→ Fit Gaussian/Student-t Copula   (after training)
Generate: Copula.sample() ──→ z ──→ Decoder ──→ X_synthetic
```

---

## Project Structure

```
synthetic-financial-data/
├── pyproject.toml            # Package metadata, dependencies, entry points, tool config
├── requirements.txt          # `-e .` (dependencies live in pyproject.toml)
├── setup.cfg                 # flake8 settings only
├── Makefile                  # Common developer commands
├── .github/workflows/ci.yml  # Lint + tests + smoke test on every push/PR
│
├── configs/                  # YAML configs; model files pull in default.yaml via `defaults:`
│   ├── default.yaml          # Data, training and evaluation defaults
│   ├── timegan.yaml / diffusion.yaml / diffusion_ts.yaml / vae_copula.yaml
│   └── diffusion_ts_quick.yaml   # Short Diffusion-TS run for CPU demos
│
├── data/                     # raw/, processed/, synthetic/, structural_breaks/ (gitignored)
├── notebooks/                # Exploration, training, evaluation, visualization
│
├── src/synfin/
│   ├── cli/                  # synfin-download/-train/-generate/-evaluate/-structural-breaks
│   ├── data/                 # Download, features, leak-free preprocessing, OHLCV rebuild
│   ├── models/               # timegan/, diffusion/, diffusion_ts/, vae_copula/, factory.py
│   ├── training/             # Trainer, EMA, LR schedulers, self-describing checkpoints
│   ├── evaluation/           # Statistical tests, stylized facts, TSTR, privacy
│   ├── structural_breaks/    # Labeled structural-break data generator (ADIA formats)
│   ├── constraints/          # OHLCV post-processing constraints
│   ├── visualization/        # Plotting utilities
│   └── utils/                # Config, logging, seed, device
│
├── scripts/                  # Thin wrappers around synfin.cli + smoke_test.py
└── tests/                    # pytest suite, incl. an end-to-end CLI pipeline test
```

---

## Installation

### Prerequisites
- Python 3.9+
- pip

### Quick Install

```bash
git clone https://github.com/AndrewFSee/synthetic-financial-data.git
cd synthetic-financial-data

# Optional, CPU-only machines: the CPU build of PyTorch is far smaller
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e .            # or: make install
pip install -e ".[dev]"     # with tests, linters, Jupyter (make install-dev)
```

---

## Quick Start

Each step consumes the previous step's output. `make pipeline MODEL=<model> TICKER=<ticker>` runs steps 2–4.

### 1. Download financial data

```bash
synfin-download --tickers AAPL MSFT        # defaults come from configs/default.yaml
```

Writes `data/raw/<TICKER>_1d.parquet`.

### 2. Train a model

```bash
synfin-train --model timegan               # or diffusion, diffusion_ts, vae_copula
synfin-train --model diffusion_ts --config configs/diffusion_ts_quick.yaml --ticker MSFT
```

Writes:
- `data/processed/<TICKER>_windows.npz`: the scaled train/val/test windows, plus the scaler and feature names.
- `checkpoints/<model>.pt`: a self-describing checkpoint with the architecture, config, feature names and scaler.

For models with a validation split, the checkpoint holds the best-validation weights (EMA weights when `ema_decay` is set).

### 3. Generate synthetic data

```bash
synfin-generate --checkpoint checkpoints/timegan.pt --num-samples 1000
```

The model is rebuilt from the checkpoint and its samples are un-scaled to the original feature units. Output goes to `data/synthetic/<model>_<TICKER>.npz` with:
- `windows`: generated feature windows
- `feature_names`
- `ohlcv`: valid OHLCV bars rebuilt from the generated returns

Diffusion sampling and VAE copula/temperature settings come from the config's `generation:` section and can be overridden with flags.

### 4. Evaluate quality

```bash
synfin-evaluate \
    --real-data data/processed/AAPL_windows.npz \
    --synthetic-data data/synthetic/timegan_AAPL.npz \
    --output reports/timegan_AAPL
```

The real training windows are the reference, and the never-seen test windows are the holdout for the TSTR and privacy checks. Every `scripts/*.py` file is equivalent to the matching `synfin-*` command.

---

## Configuration

Model configs extend `configs/default.yaml` through a `defaults:` list (resolved by `synfin.utils.config.load_config`). Every key in the shipped configs is read by the code.

```yaml
# configs/default.yaml (excerpt)
data:
  feature_set: "returns"     # stationary features (recommended) or "levels" (legacy)
  normalization: "zscore"    # scaler fit on training rows only
  window_size: 30
  train_ratio: 0.7
  val_ratio: 0.15
training:
  seed: 42
  device: "auto"             # CUDA > MPS > CPU
evaluation:
  tstr_task: "volatility"    # or "direction"
```

Diffusion and VAE configs also set `gradient_clip`, `lr_scheduler` (`cosine` with `warmup_steps`, or `reduce_on_plateau`) and `ema_decay`. TimeGAN uses `lambda_e`, `lambda_s` and `gamma`.

---

## Data Module

### Feature sets

The default `returns` set is stationary, and generated windows map back to valid OHLCV bars (`synfin.data.preprocess.windows_to_ohlcv`):

| Feature | Definition |
|---------|------------|
| `LogReturn` | `log(C_t / C_{t-1})` |
| `OpenGap` | `log(O_t / C_{t-1})` |
| `HighRange` | `log(H_t / max(O_t, C_t))` ≥ 0 |
| `LowRange` | `log(min(O_t, C_t) / L_t)` ≥ 0 |
| `LogVolumeRel` | `log1p(V_t)` minus its trailing 20-bar mean |

The legacy `levels` set (raw `Open/High/Low/Close/Volume`, `LogReturn`, `LogVolume`, `DollarVolume`) is still available. Price levels trend over the years, though, so models trained on them mostly learn the training period's price level.

Optional scale-free indicators (`use_rsi`, `use_macd`, `use_bollinger`, `use_atr`) add `RSI/100`, `MACD_Hist/Close`, `BB_Width` and `ATR/Close`.

### No look-ahead

Rows are split chronologically into train, validation and test *before* anything is fitted. The scaler sees training rows only, and each split is windowed separately, so no window straddles two splits.

---

## Evaluation Methodology

Pass windows in original units together with their feature names. Columns are found by name, and distance-based metrics use features standardized with the real data's statistics.

### Statistical Tests
- **KS test**: per-feature, on one random timestep per window so rows are close to independent. The realism score uses the KS *statistic*, because p-values from tens of thousands of pooled rows are always 0.
- **MMD**: unbiased MMD² with an RBF kernel, median-heuristic bandwidth and O(N·M) memory.
- **ACF comparison**: autocorrelations pooled *within* windows, never across window boundaries.

### Stylized Facts (computed on unscaled log returns)
- **Fat tails**: excess kurtosis.
- **Volatility clustering**: ACF of |r| and r².
- **Leverage effect**: correlation between r_t and future |r_{t+k}|.
- **Volume-volatility correlation**: correlation between volume and |r|.

### TSTR Benchmark (Train on Synthetic, Test on Real)
A logistic classifier is trained on synthetic windows and tested on held-out real windows, against a train-on-real (TRTR) baseline. The default task predicts whether the next step's |return| is above the real median, which volatility clustering makes learnable. Next-day *direction* is also available, but it is close to unpredictable, so the gap says little. Label thresholds come from the real data, so any monotone scaling works. If a classifier can't be trained on the synthetic data at all, the TSTR score is 0.

### Privacy and Collapse (calibrated against a real holdout)
A generator that collapses toward the dense centre of the data produces samples close to many training records without copying any of them. Raw distance-to-closest-record can't tell that apart from memorization, so the two are measured separately:

- **Memorization rate**: for each sample, the ratio d1/d2 of its distances to the nearest and second-nearest training records. A copy sits on one record (ratio ≈ 0); a collapsed sample sits between many (ratio ≈ 1). The rate is the share of synthetic samples below the 5th percentile of the holdout's ratios, so about 0.05 is expected without memorization.
- **Collapse diagnostics**: spread (median std ratio vs real) and nearest-record coverage (distinct nearest training records, relative to an equal-sized holdout sample). Either below 0.5 flags collapse.
- **Verdict**: `memorization` if the rate is above twice the expected rate, otherwise `collapse` if collapse is flagged, otherwise `ok`.
- **NNDR** (median nearest-record distance, synthetic vs holdout) is still reported for reference.

### Discriminative Score
A gradient-boosted classifier tries to tell real windows from synthetic ones using per-window dynamics features: volatility level, tail weight, autocorrelation of |x|, and leverage. It's scored with cross-validated ROC AUC on contiguous-block folds, so overlapping real windows can't leak between folds. KS and MMD are nearly blind to temporal structure, so this is the component that catches i.i.d. noise with the right mean and variance.

The **realism score** is the mean of five components, each in [0, 1]: `1 − KS statistic`, `1 − √MMD²`, the TSTR gap score, the privacy score (`1 −` excess memorization rate), and the discriminative score `1 − 2·(AUC − 0.5)`. They are all reported under `realism_components`.

As a sanity check on a known GARCH-t process, held-out data from the same process scores 0.98, time-shuffled data 0.89, and i.i.d. noise with matched mean and variance 0.77.

---

## Structural Break Datasets

`synfin.structural_breaks` generates labeled univariate return series for testing structural-break / change-point detectors. Output uses the ADIA Lab competition file layouts, so existing ADIA pipelines can read it directly.

```bash
# 2025 offline layout: break (if any) exactly at the pre/post boundary
python scripts/generate_structural_breaks.py --preset adia_offline --n-series 10000 --report

# 2026 real-time layout: break-free history + online segment, break anywhere online
python scripts/generate_structural_breaks.py --preset adia_realtime --split test --seed 1

# Easier/harder breaks, or real AAPL returns as the no-break process
python scripts/generate_structural_breaks.py --magnitude-scale 2.0
python scripts/generate_structural_breaks.py --bootstrap-from data/raw/AAPL_1d.parquet
```

Files are written to `data/structural_breaks/<preset>/`: `X_<split>.parquet` and `y_<split>.parquet` (plus `y_<split>_index.parquet` for real-time), in the same schema as the competition data. `meta_<split>.parquet` holds the ground truth that real data lacks: break type, effect size, τ, ramp length, decoys and every process parameter before and after the break.

**Model.** Each series is `mu + sigma * x_t`, where `x_t` is a unit-variance AR(1)-GARCH(1,1) process with Student-t shocks and an optional slow log-volatility factor. Breaks move one parameter group, instantly or along a linear ramp: `mean`, `variance`, `ar`, `tails` (shock df), `garch` (clustering intensity), or a `compound` of two. Dynamics breaks preserve variance. Transient decoys (single outliers and decaying volatility bursts) are added to break and no-break series alike, so detectors are tested against realistic false positives.

**Calibration.** The presets were tuned against the public ADIA training sets. The no-break segments reproduce the real data's tail weight, autocorrelation spread, volatility clustering and outlier frequency. Five simple two-sample detectors score within about 0.035 AUC (offline) and 0.015 AUC (real-time) of their AUC on real data.

```python
from synfin.structural_breaks import generate, get_preset, auc_report, ts_auc

ds = generate(5000, get_preset("adia_offline"), seed=0)
scores = my_detector(ds.series, ds.meta["n_hist"])          # one score per series
print(auc_report(scores, ds.meta))                           # AUC by break type / effect size / decoy
# real-time: ts_auc({id: per_step_scores}, ds.meta) -> time-stratified AUC
```

---

## Running Tests

```bash
# Run all tests
make test

# Run specific test module
python -m pytest tests/test_models/test_timegan.py -v
python -m pytest tests/test_cli/ -v          # end-to-end train -> generate -> evaluate
python scripts/smoke_test.py                 # all four models on dummy data
```

---

## Makefile Reference

`MODEL` (default `timegan`), `TICKER` (default `AAPL`) and `NUM_SAMPLES` can be overridden, e.g. `make pipeline MODEL=diffusion_ts`.

| Command | Description |
|---------|-------------|
| `make install` / `make install-dev` | Install the package (with dev extras) |
| `make download` | Download `TICKER` |
| `make train` | Train `MODEL` on `TICKER` (also `train-timegan`, `train-diffusion`, `train-diffusion-ts`, `train-vae`) |
| `make generate` | Sample from `checkpoints/MODEL.pt` |
| `make evaluate` | Evaluate the generated windows against `TICKER`'s real windows |
| `make pipeline` | train → generate → evaluate |
| `make structural-breaks` | Generate offline + real-time structural-break datasets |
| `make smoke` | End-to-end smoke test on dummy data |
| `make test` | Run unit tests |
| `make lint` / `make format` | flake8, black, isort |
| `make clean` | Clean caches and build artifacts |
| `make all` | install → download → pipeline |

---

## References

1. **TimeGAN**: Yoon, J., Jarrett, D., & van der Schaar, M. (2019). *Time-series Generative Adversarial Networks*. NeurIPS 2019. [arXiv:1906.09592](https://arxiv.org/abs/1906.09592)

2. **DDPM**: Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

3. **DDIM**: Song, J., Meng, C., & Ermon, S. (2020). *Denoising Diffusion Implicit Models*. ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)

4. **VAE**: Kingma, D.P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)

5. **FinDiff**: Sattarov, T. et al. (2023). *FinDiff: Diffusion Models for Financial Tabular Data Generation*. [arXiv:2309.01472](https://arxiv.org/abs/2309.01472)

6. **Quant GANs**: Wiese, M. et al. (2020). *Quant GANs: Deep Generation of Financial Time Series*. Quantitative Finance. [arXiv:1907.04155](https://arxiv.org/abs/1907.04155)

7. **Stylized Facts**: Cont, R. (2001). *Empirical properties of asset returns: stylized facts and statistical issues*. Quantitative Finance, 1(2), 223–236.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.