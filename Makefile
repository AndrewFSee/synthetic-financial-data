.PHONY: install install-dev download train-timegan train-diffusion train-vae generate evaluate test lint clean all help

PYTHON := python
PIP := pip
CONFIG_DIR := configs
DATA_DIR := data
CHECKPOINT_DIR := checkpoints
REPORT_DIR := reports

help:
	@echo "Available targets:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install dev dependencies (includes testing, linting)"
	@echo "  download         Download financial data"
	@echo "  train-timegan    Train the TimeGAN model"
	@echo "  train-diffusion  Train the Diffusion model"
	@echo "  train-vae        Train the VAE+Copula model"
	@echo "  generate         Generate synthetic data"
	@echo "  evaluate         Run full evaluation suite"
	@echo "  test             Run unit tests"
	@echo "  lint             Run linters (flake8, black, isort)"
	@echo "  clean            Clean generated files and caches"
	@echo "  all              Full pipeline: install -> download -> train -> generate -> evaluate"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

download:
	$(PYTHON) scripts/download_data.py \
		--tickers AAPL MSFT GOOGL AMZN META \
		--start 2015-01-01 \
		--end 2024-12-31 \
		--interval 1d

train-timegan:
	$(PYTHON) scripts/train.py \
		--model timegan \
		--config $(CONFIG_DIR)/timegan.yaml

train-diffusion:
	$(PYTHON) scripts/train.py \
		--model diffusion \
		--config $(CONFIG_DIR)/diffusion.yaml

train-vae:
	$(PYTHON) scripts/train.py \
		--model vae_copula \
		--config $(CONFIG_DIR)/vae_copula.yaml

generate:
	$(PYTHON) scripts/generate.py \
		--model timegan \
		--checkpoint $(CHECKPOINT_DIR)/timegan_best.pt \
		--num-samples 1000 \
		--seq-length 30

evaluate:
	$(PYTHON) scripts/evaluate.py \
		--real-data $(DATA_DIR)/processed/AAPL.parquet \
		--synthetic-data $(DATA_DIR)/synthetic/timegan_AAPL.parquet \
		--output $(REPORT_DIR)/

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

all: install download train-timegan generate evaluate
