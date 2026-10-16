# Developer entrypoints. Every target here is also what CI runs, so "it passes
# locally" and "it passes in CI" cannot drift apart.
.PHONY: help install lint format typecheck test test-all smoke score score-hub docker clean
.DEFAULT_GOAL := help

DATA ?= data/dataset_production.pkl
WEIGHTS ?= weights

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create the environment from the lockfile (hermetic)
	uv sync --locked --extra dev

lint:  ## ruff check + format check
	uv run --locked ruff check .
	uv run --locked ruff format --check .

format:  ## Apply ruff fixes and formatting
	uv run --locked ruff check --fix .
	uv run --locked ruff format .

typecheck:  ## mypy --strict
	uv run --locked mypy

test:  ## Fast suite (no AntiBERTy weights, no dataset, no network)
	uv run --locked pytest -m "not slow"

test-all:  ## Everything, including tests needing real weights
	uv run --locked pytest

smoke:  ## Real end-to-end run: make smoke DATA=path/to/dataset_production.pkl
	uv run --locked python scripts/smoke_test.py --data $(DATA)

score:  ## Score local checkpoints on fold 0
	uv run --locked rbdaim score --checkpoints $(WEIGHTS) --fold 0 data.path=$(DATA)

score-hub:  ## Download the released weights and score them on fold 0
	uv run --locked rbdaim score --from-hub --fold 0 data.path=$(DATA)

docker:  ## Build the container image
	docker build -t rbdaim-classifier:local .

clean:  ## Remove caches and build artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
