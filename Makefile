.PHONY: help install install-dev test test-unit test-integration test-e2e lint typecheck format run-api run-ui docker-build docker-up clean

PYTHON ?= python3
PIP ?= pip
HOST ?= 127.0.0.1

help:
	@echo "OmniForge - Make Targets"
	@echo "  make install          Install package in editable mode"
	@echo "  make install-dev      Install package with all dev extras"
	@echo "  make test             Run full pytest suite"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-e2e         Run end-to-end tests only"
	@echo "  make lint             Run ruff linter"
	@echo "  make typecheck        Run mypy static type checker"
	@echo "  make format           Format codebase with ruff"
	@echo "  make run-api          Start FastAPI production server"
	@echo "  make run-ui           Start FastAPI + React enterprise studio UI"
	@echo "  make docker-build     Build production container image"
	@echo "  make docker-up        Launch full stack via docker compose"
	@echo "  make clean            Clean build and cache artifacts"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[all]"

test:
	pytest -v

test-unit:
	pytest -v tests/unit

test-integration:
	pytest -v tests/integration

test-e2e:
	pytest -v tests/e2e

lint:
	ruff check src tests

typecheck:
	mypy src/omniforge

format:
	ruff format src tests

run-api:
	uvicorn omniforge.api.main:app --host $(HOST) --port 8000 --reload

run-ui:
	python -m omniforge.launcher --mode ui --host $(HOST) --ui-port 8501 --no-browser

docker-build:
	docker build -f docker/Dockerfile -t omniforge-enterprise:4.0.0 .

docker-up:
	docker compose -f docker/docker-compose.yml up -d --build

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
