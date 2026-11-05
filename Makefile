# Envoltorio fino de `python -m fakenews.tasks` (la lógica vive en src/fakenews/tasks.py).
# En Windows, sin make: ver README.
PY ?= python3.12
TASKS := PYTHONPATH=src $(PY) -m fakenews.tasks

.PHONY: setup data train quickstart evaluate analyze report notebook test test-fast demo all predict clean-artifacts

setup data train quickstart evaluate analyze report notebook test test-fast demo all:
	$(TASKS) $@

predict:          ## Ejemplo: make predict TEXT="..."
	.venv/bin/python -m fakenews.cli predict "$(TEXT)"

clean-artifacts:  ## Borra datos procesados, modelos y figuras generadas (no toca data/raw)
	rm -rf data/interim/* data/processed/* models/*.joblib models/candidates reports/figures/*
