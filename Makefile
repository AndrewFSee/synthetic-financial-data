.PHONY: install install-dev download train train-timegan train-diffusion train-diffusion-ts train-vae generate evaluate pipeline structural-breaks smoke test lint format clean all help

PYTHON := python
PIP := pip
CONFIG_DIR := configs
CHECKPOINT_DIR := checkpoints
REPORT_DIR := reports

# Override on the command line, e.g. `make pipeline MODEL=diffusion_ts TICKER=MSFT`
MODEL ?= timegan
TICKER ?= AAPL
NUM_SAMPLES ?= 1000

help:
	@echo "Available targets:"
	@echo "  install            Install the package"
	@echo "  install-dev        Install with dev dependencies (tests, linters, Jupyter)"
	@echo "  download           Download OHLCV data (tickers/dates from configs/default.yaml)"
	@echo "  train              Train MODEL on TICKER (default: $(MODEL) on $(TICKER))"
	@echo "  train-timegan / train-diffusion / train-diffusion-ts / train-vae"
	@echo "  generate           Sample NUM_SAMPLES windows from checkpoints/MODEL.pt"
	@echo "  evaluate           Compare the generated windows with TICKER's real windows"
	@echo "  pipeline           train -> generate -> evaluate for MODEL on TICKER"
	@echo "  structural-breaks  Generate labeled structural-break datasets (ADIA formats)"
	@echo "  smoke              End-to-end smoke test on dummy data (no network)"
	@echo "  test               Run unit tests"
	@echo "  lint               Run linters (flake8, black, isort)"
	@echo "  format             Auto-format code"
	@echo "  clean              Clean generated files and caches"
	@echo "  all                install -> download -> pipeline"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

download:
	$(PYTHON) scripts/download_data.py --tickers $(TICKER)

train:
	$(PYTHON) scripts/train.py --model $(MODEL) --config $(CONFIG_DIR)/$(MODEL).yaml --ticker $(TICKER)

train-timegan:
	$(MAKE) train MODEL=timegan

train-diffusion:
	$(MAKE) train MODEL=diffusion

train-diffusion-ts:
	$(MAKE) train MODEL=diffusion_ts

train-vae:
	$(MAKE) train MODEL=vae_copula

generate:
	$(PYTHON) scripts/generate.py \
		--checkpoint $(CHECKPOINT_DIR)/$(MODEL).pt \
		--num-samples $(NUM_SAMPLES)

evaluate:
	$(PYTHON) scripts/evaluate.py \
		--real-data data/processed/$(TICKER)_windows.npz \
		--synthetic-data data/synthetic/$(MODEL)_$(TICKER).npz \
		--output $(REPORT_DIR)/$(MODEL)_$(TICKER)

pipeline: train generate evaluate

structural-breaks:
	$(PYTHON) scripts/generate_structural_breaks.py --preset adia_offline --report
	$(PYTHON) scripts/generate_structural_breaks.py --preset adia_realtime

smoke:
	$(PYTHON) scripts/smoke_test.py

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	$(PYTHON) -m flake8 src/ scripts/ tests/
	$(PYTHON) -m black --check src/ scripts/ tests/
	$(PYTHON) -m isort --check-only src/ scripts/ tests/

format:
	$(PYTHON) -m black src/ scripts/ tests/
	$(PYTHON) -m isort src/ scripts/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .mypy_cache/ 2>/dev/null || true

all: install download pipeline
