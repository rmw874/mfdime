# Common tasks. Python runs through uv (see pyproject.toml); the package is imported from src/ (PYTHONPATH below).
export PYTHONPATH := src
PYTHON ?= uv run python
SMOKE_DIR ?= runs/smoke
.PHONY: setup lint typecheck test faces-check smoke check

setup:  ## create .venv from uv.lock
	uv sync --frozen

lint:
	uv run ruff check .

typecheck:
	uv run ty check

test:  ## synthetic H2 checks: no model downloads or submodules
	$(PYTHON) -P -m unittest discover -s tests -v

faces-check:  ## include the optional OpenCLIP API tests; no model downloads
	uv run --group faces ruff check .
	uv run --group faces ty check
	uv run --group faces python -P -m unittest discover -s tests -v

smoke:  ## a few minutes on the CPU: every script with tiny random models and fake data (no download)
	rm -rf $(SMOKE_DIR)
	$(PYTHON) -P -m mfdime.explain --smoke --seeds 0 1 --out-dir $(SMOKE_DIR)/explain
	$(PYTHON) -P -m mfdime.explain --smoke --dataset celebahq --peek walk_rest --seeds 0 --split held_out --out-dir $(SMOKE_DIR)/explain_faces
	$(PYTHON) -P -m mfdime.explain --smoke --dataset celebahq --peek jump1 --seeds 0 1 --out-dir $(SMOKE_DIR)/h1/jump1
	$(PYTHON) -P -m mfdime.explain --smoke --dataset celebahq --peek euler_rest --seeds 0 1 --out-dir $(SMOKE_DIR)/h1/euler_rest
	$(PYTHON) -P -m mfdime.score --smoke --runs $(SMOKE_DIR)/h1 --out-dir $(SMOKE_DIR)/h1_score

check: lint typecheck test smoke
