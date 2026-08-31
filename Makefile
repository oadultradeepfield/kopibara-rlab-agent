MAKEFILE_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
UV := uv
RUN := $(UV) run --directory $(MAKEFILE_DIR)
SOURCES := src tests

.DEFAULT_GOAL := check
.PHONY := install fmt lint types test verify-starter pairwise deepfm lambdarank history history-ensemble history-pipeline bonus-1k bonus-1k-lgbm autonomous check cov install-hooks clean

install:
	cd $(MAKEFILE_DIR) && $(UV) sync --all-groups

fmt:
	$(RUN) ruff format $(SOURCES)
	$(RUN) ruff check --fix $(SOURCES)

lint:
	$(RUN) ruff format --check $(SOURCES)
	$(RUN) ruff check $(SOURCES)

types:
	$(RUN) mypy $(SOURCES)

test:
	$(RUN) pytest

verify-starter:
	$(RUN) python scripts/verify_starter.py

pairwise:
	$(RUN) python experiments/pairwise_fm.py

deepfm:
	$(RUN) python experiments/deepfm_mtl.py --output-dir runs/deepfm-pure/latest

lambdarank:
	$(RUN) python experiments/lambdarank.py --output-dir runs/lambdarank-pure/latest

history:
	$(RUN) python experiments/history_lgbm.py --feedbacks all --output-dir runs/history-lgbm-pure/latest

history-ensemble:
	$(RUN) python experiments/history_lgbm.py --feedbacks all --seed 0 --output-dir runs/history-lgbm-pure/seed0
	$(RUN) python experiments/history_lgbm.py --feedbacks all --seed 1 --output-dir runs/history-lgbm-pure/seed1
	$(RUN) python experiments/history_lgbm.py --feedbacks all --seed 2 --output-dir runs/history-lgbm-pure/seed2
	$(RUN) python experiments/ensemble_submission.py --data-dir kuairand-starter-kit/KuaiRand-Pure/data --scores runs/history-lgbm-pure/seed0/test_scores.npy runs/history-lgbm-pure/seed1/test_scores.npy runs/history-lgbm-pure/seed2/test_scores.npy --output runs/history-lgbm-pure/submission.csv

history-pipeline:
	$(RUN) python experiments/run_history_pipeline.py

bonus-1k:
	$(RUN) python experiments/pairwise_fm.py --data_dir kuairand-starter-kit/KuaiRand-1K/data --output-dir runs/bonus-1k/latest
	$(RUN) python experiments/write_submission.py --data-dir kuairand-starter-kit/KuaiRand-1K/data --scores runs/bonus-1k/latest/test_scores.npy --output runs/bonus-1k/latest/submission.csv

bonus-1k-lgbm:
	$(RUN) python experiments/lambdarank.py --data-dir kuairand-starter-kit/KuaiRand-1K/data --objective binary --seed 0 --output-dir runs/lightgbm-binary-1k/seed0
	$(RUN) python experiments/write_submission.py --data-dir kuairand-starter-kit/KuaiRand-1K/data --scores runs/lightgbm-binary-1k/seed0/test_scores.npy --output runs/lightgbm-binary-1k/seed0/submission.csv

autonomous:
	$(RUN) kopibara-agent --run

check: lint types test verify-starter

cov:
	$(RUN) pytest --cov=src --cov-report=term-missing

install-hooks:
	@bash $(MAKEFILE_DIR)/scripts/install_hooks.sh

clean:
	cd $(MAKEFILE_DIR) && rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov dist build
	cd $(MAKEFILE_DIR) && find . -name __pycache__ -type d -prune -exec rm -rf {} +
