# synfin — Synthetic Financial Data Generation

> A layered walkthrough of this project: **Part 1** is plain-language and works for any
> interviewer; **Part 2** is the technical deep-dive for an ML/quant audience. Read Part 1 to
> understand *what* and *why*; read Part 2 for *how*.

---

# Part 1 — The high-level story

## The problem: why generate *fake* financial data?

Real market data is scarce, sensitive, and biased toward the one history that actually happened.
That creates four concrete pain points this project addresses:

1. **Privacy / sharing.** Banks and funds can't freely share customer or proprietary trading data.
   A high-quality *synthetic* dataset preserves the statistical character of the real data without
   exposing any real record.
2. **Data augmentation.** Deep models for forecasting, trading, or risk are data-hungry, but we
   only have ~250 trading days a year. Synthetic data expands the training set.
3. **Backtesting & stress testing.** We can generate many *plausible alternative histories* —
   including rare crash-like scenarios — to test a strategy or a risk model against more than the
   single path the market happened to take.
4. **Benchmarking.** Synthetic data with known properties is a controlled test-bed for evaluating
   models.

The hard part is that financial time series have a stubborn statistical fingerprint — the
**"stylized facts"** — that naive models fail to reproduce: returns have **fat tails** (extreme
moves happen far more often than a normal distribution predicts), **volatility clusters** (calm
periods and turbulent periods bunch together), and there's a **leverage effect** (prices tend to
get more volatile after they fall). A generator that gets the averages right but misses these is
useless for risk work.

## What this project does, end to end

```
 Download real OHLCV          Preprocess                 Train a generative model
 (yfinance: AAPL, ...)   ->   log-returns, scaling,  ->   (TimeGAN / Diffusion /
                              30-day sliding windows       Diffusion-TS / VAE+Copula)
                                                                     |
                                                                     v
 Evaluate realism          Post-process to valid OHLCV       Generate synthetic windows
 (stylized facts, MMD,  <-  (High >= Open/Close, etc.)   <-  (sample from the model)
  TSTR, privacy)
```

The pipeline is fully scripted: `synfin-download`, `synfin-train`, `synfin-generate`,
`synfin-evaluate` (plus `scripts/smoke_test.py` to exercise the whole chain on dummy data).

## The four models, in one paragraph each

- **TimeGAN** — Two neural nets play a game: a *generator* invents fake sequences and a
  *discriminator* tries to spot the fakes. Training them against each other pushes the fakes to look
  real. TimeGAN adds an autoencoder and a "supervisor" so the fakes also respect the *time order* of
  the data, not just its snapshot statistics.

- **Diffusion (DDPM)** — Learn to *denoise*. Start from a real sequence, gradually add random noise
  until it's pure static, and train a network to reverse one step of that process. To generate, you
  start from pure noise and run the learned denoiser repeatedly until a realistic sequence emerges.
  This is the same family of method behind modern image generators, and it is currently the
  state-of-the-art for this problem.

- **Diffusion-TS** *(the SOTA upgrade added here)* — A smarter diffusion model. Instead of a
  convolutional network, it uses a **Transformer** (better at long-range patterns) and forces its
  output to decompose into an interpretable **trend + seasonal cycle + residual**. Crucially, it is
  trained partly in the **frequency domain** (via a Fourier transform), which directly targets the
  volatility-clustering and slow-decay behaviour that vanilla diffusion under-fits.

- **VAE + Copula** — A *variational autoencoder* compresses each sequence into a small latent code
  and learns to reconstruct it. A **copula** is then fit on top of those latent codes to capture the
  dependence structure between features (including fat-tailed joint behaviour via a Student-t
  copula). Lightweight and fast, with an explicit handle on cross-feature correlation.

## How we know it worked

We don't just eyeball charts. The evaluation suite scores synthetic data on:
- **Stylized facts** — does it reproduce fat tails, volatility clustering, the leverage effect?
- **Distributional distance** — KS tests per feature, and **MMD** (a kernel measure of how far two
  distributions are).
- **Usefulness (TSTR)** — *Train on Synthetic, Test on Real*: train a downstream classifier purely
  on synthetic data and see how well it does on real data. A small gap means the synthetic data
  carries the real signal.
- **Privacy** — nearest-neighbour checks that the model *generalized* rather than *memorized* real
  records.

---

# Part 2 — Technical deep-dive

## Repository architecture

```
src/synfin/
  data/          download (yfinance) -> preprocess (log-returns, scaling, windowing) -> Dataset
  models/
    timegan/     embedder, recovery, generator, supervisor, discriminator
    diffusion/   DDPM: UNet1D denoiser, noise schedules, DDPM/DDIM samplers
    diffusion_ts/ Diffusion-TS: transformer backbone, seasonal-trend decomposition, samplers
    vae_copula/  encoder, decoder, Gaussian/Student-t copula
  training/      Trainer (grad clip, EMA, LR schedules, best-val restore), checkpoints, losses
  structural_breaks/ labeled structural-break data generator + detector evaluation
  cli/           synfin-download / -train / -generate / -evaluate / -structural-breaks
  evaluation/    statistical_tests, stylized_facts, tstr, privacy, metrics aggregator
  constraints/   OHLCV validity post-processing
  visualization/ candlestick, distributions, correlations, time-series
  utils/         config (YAML merge + `defaults:`), device, logging, seed
configs/         default.yaml + one YAML per model
scripts/         thin wrappers around synfin.cli, plus smoke_test
tests/           pytest suite per module + end-to-end CLI pipeline test
```

Data convention everywhere: tensors are `(batch, seq_len, features)`, default window = 30 days,
5 stationary features (LogReturn, OpenGap, HighRange, LowRange, LogVolumeRel) that map back to
valid OHLCV bars. Rows are split **chronologically** (train→val→test) *before* the scaler is fit
on training rows only, and each split is windowed separately, so nothing leaks across splits.

## Model 1 — TimeGAN (Yoon et al., NeurIPS 2019)

> **Status: baseline only.** On AAPL returns it produces smooth paths, partially
> collapses and exaggerates the leverage effect (see the README's model comparison).
> The VAE + Copula model is the recommended default.

Five RNN modules operating in a learned latent space: **Embedder** `X→H`, **Recovery** `H→X`,
**Generator** `Z→Ê`, **Supervisor** (predicts next latent state), **Discriminator** `H→[0,1]`.
Three-phase training:
1. **Autoencoder** — train Embedder+Recovery to reconstruct (`10·√MSE`).
2. **Supervised** — train Supervisor to predict `h_t` from `h_{t-1}` in latent space.
3. **Joint adversarial** — generator minimizes adversarial losses on both the supervised latents
   `Ĥ` and the raw generator output `Ê` (weighted by `gamma`), plus supervised + moment-matching
   losses; the discriminator is trained against both kinds of fakes (with a `loss > 0.15` guard
   to keep it from overpowering G).

Strength: explicit temporal supervision. Weakness: adversarial training is unstable, and GANs are
prone to mode collapse — which on financial data shows up as *under-dispersed tails*.

## Model 2 — Diffusion / DDPM (Ho et al., NeurIPS 2020)

Forward process adds Gaussian noise on a schedule:
`q(x_t | x_0) = √(ᾱ_t)·x_0 + √(1−ᾱ_t)·ε`. The network is trained with the simplified objective to
**predict the noise** ε:  `L = E_t,ε ‖ε − ε_θ(x_t, t)‖²`. Implementation details:
- **Backbone**: `UNet1D` over the time axis, GroupNorm + SiLU, residual blocks, sinusoidal timestep
  embedding ([unet.py](../src/synfin/models/diffusion/unet.py)).
- **Schedules**: linear and cosine (Nichol & Dhariwal) with precomputed constants
  ([noise_schedule.py](../src/synfin/models/diffusion/noise_schedule.py)).
- **Sampling**: full **DDPM** reverse chain *and* accelerated **DDIM**
  ([sampler.py](../src/synfin/models/diffusion/sampler.py)).

This is the strongest of the three original models — diffusion avoids the adversarial instability of
GANs and the blurry reconstructions of VAEs — but it operates on **raw windows with a conv backbone**,
which is a generation behind the current literature.

## Model 3 — Diffusion-TS (the SOTA upgrade) — Yuan & Qiao, ICLR 2024

Added in `src/synfin/models/diffusion_ts/`. Three differences from the vanilla DDPM, each targeting a
known weakness for financial data:

1. **Transformer encoder–decoder backbone** instead of a conv U-Net
   ([transformer_backbone.py](../src/synfin/models/diffusion_ts/transformer_backbone.py)).
   Self-attention models long-range dependence across the window better than fixed-width
   convolutions. The diffusion timestep is injected as an additive conditioning signal at every
   position; sinusoidal positional encoding handles sequence order.

2. **x0-prediction via interpretable seasonal–trend decomposition**
   ([decomposition.py](../src/synfin/models/diffusion_ts/decomposition.py)). The network predicts the
   *clean* signal `x̂_0 = trend + seasonality + residual`, where **trend** is a low-order polynomial
   in time, **seasonality** is a learned bank of Fourier (sin/cos) harmonics, and **residual** is a
   free linear projection. Predicting `x_0` (not ε) is what makes the frequency loss below
   well-defined, and the decomposition makes the generator interpretable.

3. **Combined time + frequency objective.**
   `L = MSE(x̂_0, x_0) + λ · MSE(rFFT(x̂_0), rFFT(x_0))`. The Fourier term compares the real-FFT
   spectra (real *and* imaginary parts, i.e. amplitude and phase) along the time axis, directly
   penalizing mismatch in the frequency structure that underlies **volatility clustering** and the
   **slow decay of autocorrelations** — exactly the stylized facts a pure time-domain MSE tends to
   wash out. λ is `fourier_loss_weight` in [configs/diffusion_ts.yaml](../configs/diffusion_ts.yaml).

Sampling reuses the schedule constants but is re-derived for an **x0-parameterized** model: the DDPM
step uses the closed-form posterior mean `q(x_{t-1} | x_t, x̂_0)`, and DDIM recovers the implied
noise from `x̂_0` before re-noising ([sampler.py](../src/synfin/models/diffusion_ts/sampler.py)). The
public API (`q_sample`, `training_loss`, `forward`, `sample`) matches the vanilla diffusion model, so
it drops straight into the existing `Trainer` and CLI scripts.

## Model 4 — VAE + Copula

A sequence VAE (LSTM/GRU encoder→`μ, logσ²`, reparameterize, RNN decoder) trained on the ELBO
`recon_loss + β·KL`, with optional **β-annealing** to avoid posterior collapse. At generation time a
**copula** (Gaussian via Cholesky, or **Student-t** for heavier joint tails) is fit on the latent
codes to inject realistic cross-feature dependence before decoding. Cheapest to train and the only
model with an *explicit, inspectable* dependence structure — useful when correlation fidelity matters
more than marginal sharpness.

## Evaluation methodology (and why each metric matters for finance)

`compute_all_metrics` ([metrics.py](../src/synfin/evaluation/metrics.py)) aggregates:

| Metric | What it checks | Why it matters here |
|---|---|---|
| **KS statistic** (per feature, one timestep per window) | Marginal distribution match | Basic distributional fidelity |
| **MMD²** (RBF, median-heuristic bandwidth) | Joint window distribution distance | Single scalar for overall closeness |
| **ACF comparison** (pooled within windows) | Autocorrelation at multiple lags | Volatility clustering lives in the ACF of |returns| |
| **Stylized facts** (on unscaled returns) | Fat tails, vol clustering, leverage, volume–volatility | The financial fingerprint a model must reproduce |
| **TSTR** (next-step volatility task) | Train-on-synthetic / test-on-real AUC gap | Proves the synthetic data is *useful*, not just close |
| **Privacy** (d1/d2 memorization rate, vs. an unseen real holdout) | Whether samples sit on one specific training record | Guards against the model copying real records |
| **Collapse diagnostics** (spread, nearest-record coverage) | Too little spread, or too few distinct regions covered | Separates mode collapse from memorization, which raw distances confuse |
| **Discriminative score** | CV AUC of a real-vs-synthetic classifier on window dynamics | The only check that catches i.i.d. noise with the right moments |

Columns are located by name, distance metrics use real-standardized features, and the five
components (`1 − KS`, `1 − √MMD²`, TSTR gap score, privacy, discriminative) roll up into
`realism_score ∈ [0,1]`. The report also carries a privacy verdict: `ok`, `collapse` or
`memorization`.

## SOTA landscape — what we have vs. what's next

Diffusion is now the consensus SOTA family for this problem, beating GANs (instability, mode
collapse / thin tails) and VAEs (blurry reconstructions). Where the frontier currently sits:

- **Diffusion-TS** (ICLR 2024) — interpretable transformer diffusion with a Fourier objective.
  *Implemented here* as the SOTA-tier model. — https://arxiv.org/abs/2403.01742
- **Financial diffusion with stylized-fact focus** (Quant. Finance 2025) — wavelet-image DDPMs that
  explicitly reproduce fat tails and U-shaped intraday volume. — https://arxiv.org/abs/2410.18897
- **CoFinDiff** (IJCAI 2025) — *controllable* diffusion: condition generation on a target trend or
  volatility. A natural next step for scenario / stress generation. — https://www.ijcai.org/proceedings/2025/1040
- **GBM-noise diffusion** (2025) — inject noise *proportional to price* (geometric Brownian motion)
  so heteroskedasticity is baked into the forward process; sharpens fat tails, clustering, and the
  leverage effect. — https://arxiv.org/pdf/2507.19003
- **Signature / MMD-kernel methods** (Sig-WGAN, Math. Finance 2024) — use the *path signature* as a
  principled feature map for time series, turning the GAN min-max into supervised learning. —
  https://onlinelibrary.wiley.com/doi/abs/10.1111/mafi.12423

**Honest positioning:** the repo now spans GAN, VAE, and *two* diffusion tiers (vanilla DDPM and
Diffusion-TS). The clearest next increments are **conditional generation** (CoFinDiff-style controls
for stress scenarios) and **price-proportional / GBM noise** to push stylized-fact fidelity further.

## How to run

```bash
pip install -e ".[dev]"                 # CPU-only torch is plenty: --index-url .../whl/cpu
pytest -q                               # unit tests (shape/correctness) for every module
python scripts/smoke_test.py            # end-to-end train->generate->evaluate on dummy data

# Real-data path (needs network):
synfin-download --tickers AAPL MSFT --start 2015-01-01 --end 2024-12-31
synfin-train    --model diffusion_ts --ticker AAPL
synfin-generate --checkpoint checkpoints/diffusion_ts.pt
synfin-evaluate --real-data data/processed/AAPL_windows.npz \
                --synthetic-data data/synthetic/diffusion_ts_AAPL.npz
```
