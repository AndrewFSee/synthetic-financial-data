# 🏦 Synthetic Financial Data

> Generate realistic synthetic OHLCV (Open, High, Low, Close, Volume) financial time-series data using three state-of-the-art generative model architectures.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

**synfin** scaffolds a complete, production-ready Python project for generating synthetic stock market data.  
It implements **three architecture options** you can train, compare, and combine:

| Model | Description | Reference |
|-------|-------------|-----------|
| **TimeGAN** | Recurrent GAN with temporal supervision | Yoon et al., NeurIPS 2019 |
| **Diffusion (DDPM)** | 1D U-Net denoising diffusion model | Ho et al., NeurIPS 2020 |
| **VAE + Copula** | Variational Autoencoder with copula dependency modeling | Kingma & Welling 2014 |

Generated data reproduces key **stylized facts** of financial time series:
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
Copula:   z_train ──→ Fit Gaussian/Student-t Copula
Generate: Copula.sample() ──→ z ──→ Decoder ──→ X_synthetic
```

---

## Project Structure

```
synthetic-financial-data/
├── README.md
├── pyproject.toml            # Modern Python project config
├── requirements.txt          # Pinned dependencies
├── setup.cfg                 # Package configuration
├── .gitignore
├── Makefile                  # Common developer commands
│
├── configs/                  # YAML configuration files
│   ├── default.yaml          # Shared defaults
│   ├── timegan.yaml          # TimeGAN hyperparameters
│   ├── diffusion.yaml        # Diffusion model hyperparameters
│   └── vae_copula.yaml       # VAE+Copula hyperparameters
│
├── data/
│   ├── raw/                  # Downloaded data (gitignored)
│   ├── processed/            # Preprocessed data (gitignored)
│   └── synthetic/            # Generated synthetic data (gitignored)
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_visualization.ipynb
│
├── src/
│   └── synfin/               # Main Python package
│       ├── data/             # Data loading & preprocessing
│       ├── models/
│       │   ├── timegan/      # TimeGAN (Yoon et al.)
│       │   ├── diffusion/    # DDPM diffusion model
│       │   └── vae_copula/   # VAE + Copula
│       ├── training/         # Unified trainer, losses, callbacks
│       ├── evaluation/       # Statistical tests, stylized facts, TSTR, privacy
│       ├── constraints/      # OHLCV post-processing constraints
│       ├── visualization/    # Plotting utilities
│       └── utils/            # Config, logging, seed, device
│
├── scripts/                  # CLI entry points
│   ├── download_data.py
│   ├── train.py
│   ├── generate.py
│   └── evaluate.py
│
└── tests/                    # Unit tests (pytest)
    ├── test_data/
    ├── test_models/
    ├── test_constraints/
    └── test_evaluation/
```

---

## Installation

### Prerequisites
- Python 3.9+
- pip

### Quick Install

```bash
# Clone the repository
git clone https://github.com/AndrewFSee/synthetic-financial-data.git
cd synthetic-financial-data

# Install dependencies and package
pip install -r requirements.txt
pip install -e .
```

### Developer Install

```bash
pip install -e ".[dev]"
```

Or use the Makefile:

```bash
make install        # Production dependencies
make install-dev    # Development dependencies (includes testing, linting)
```

---

## Quick Start

### 1. Download Financial Data

```bash
# Download daily OHLCV data for multiple tickers
python scripts/download_data.py \
    --tickers AAPL MSFT GOOGL AMZN META \
    --start 2015-01-01 \
    --end 2024-12-31 \
    --interval 1d
```

Or with Make:
```bash
make download
```

### 2. Train a Model

```bash
# Train TimeGAN
python scripts/train.py --model timegan --config configs/timegan.yaml

# Train Diffusion Model
python scripts/train.py --model diffusion --config configs/diffusion.yaml

# Train VAE + Copula
python scripts/train.py --model vae_copula --config configs/vae_copula.yaml
```

```bash
make train-timegan
make train-diffusion
make train-vae
```

### 3. Generate Synthetic Data

```bash
python scripts/generate.py \
    --model timegan \
    --checkpoint checkpoints/timegan_best.pt \
    --num-samples 1000 \
    --seq-length 30
```

```bash
make generate
```

### 4. Evaluate Quality

```bash
python scripts/evaluate.py \
    --real-data data/processed/AAPL.parquet \
    --synthetic-data data/synthetic/timegan_synthetic.npy \
    --output reports/
```

```bash
make evaluate
```

---

## Configuration

All configurations live in `configs/`. The hierarchy is:

```
configs/default.yaml        ← shared base configuration
    └── configs/timegan.yaml    ← model-specific overrides
    └── configs/diffusion.yaml
    └── configs/vae_copula.yaml
```

### Key Configuration Options

```yaml
# configs/default.yaml
data:
  tickers: [AAPL, MSFT, GOOGL]
  start_date: "2015-01-01"
  end_date: "2024-12-31"
  window_size: 30
  normalization: "minmax"   # or "zscore"
  features: [Open, High, Low, Close, Volume, LogReturn, LogVolume, DollarVolume]

training:
  seed: 42
  device: "auto"            # auto-selects CUDA > MPS > CPU
  batch_size: 64
  checkpoint_dir: "checkpoints"
```

---

## Data Module

### Features Computed Automatically

| Feature | Description |
|---------|-------------|
| `LogReturn` | `log(Close_t / Close_{t-1})` |
| `LogVolume` | `log(1 + Volume)` |
| `DollarVolume` | `Close × Volume` |
| `RSI` | Relative Strength Index (14-period) |
| `MACD` | Moving Average Convergence Divergence |
| `BB_Upper/Lower` | Bollinger Bands (20-period, 2σ) |
| `ATR` | Average True Range (14-period) |
| `RealizedVol` | Rolling std of log returns (annualized) |

---

## Evaluation Methodology

### Statistical Tests
- **Kolmogorov-Smirnov (KS) test** — per-feature marginal distribution comparison
- **Maximum Mean Discrepancy (MMD)** — distribution distance with RBF kernel
- **ACF comparison** — autocorrelation function similarity at multiple lags

### Stylized Facts
- **Fat tails** — excess kurtosis of return distributions
- **Volatility clustering** — ACF of absolute/squared returns
- **Leverage effect** — negative return-volatility correlation
- **Volume-volatility correlation** — positive volume and volatility correlation

### TSTR Benchmark (Train on Synthetic, Test on Real)
Trains a downstream classifier (next-day return direction) on synthetic data, evaluates on real data. Compares against TRTR (real-to-real) baseline.

### Privacy Metrics
- **NNDR** — Nearest-Neighbor Distance Ratio (memorization detection)
- **DCR** — Distance to Closest Record
- **Membership Inference Risk** — fraction of synthetic samples suspiciously close to training data

---

## Running Tests

```bash
# Run all tests
make test

# Run specific test module
python -m pytest tests/test_models/test_timegan.py -v
python -m pytest tests/test_constraints/ -v
```

---

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make install` | Install production dependencies |
| `make install-dev` | Install dev dependencies |
| `make download` | Download financial data |
| `make train-timegan` | Train TimeGAN model |
| `make train-diffusion` | Train Diffusion model |
| `make train-vae` | Train VAE+Copula model |
| `make generate` | Generate synthetic data |
| `make evaluate` | Run evaluation suite |
| `make test` | Run unit tests |
| `make lint` | Run flake8, black, isort |
| `make format` | Auto-format code |
| `make clean` | Clean caches and build artifacts |
| `make all` | Full pipeline |

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