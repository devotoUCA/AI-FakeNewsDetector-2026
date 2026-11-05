# Detector de Fake News — comandos del proyecto.
# Uso: make setup && make all   (o paso a paso: data, train, evaluate, analyze, notebook, test, demo)
PY ?= python3.12
VENV ?= .venv
BIN := $(VENV)/bin
PYTHON := $(BIN)/python
CLI := $(PYTHON) -m fakenews.cli

.PHONY: setup data data-external train evaluate shortcuts panel train-v2 evaluate-v2 quickstart analyze notebook test test-fast demo predict all clean-artifacts

setup:            ## Crea el entorno virtual e instala dependencias fijadas + el paquete
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e . --no-deps

data:             ## Descarga datasets, prepara splits y corre la auditoría
	$(CLI) download
	$(CLI) prepare
	$(CLI) audit

train:            ## Entrena y selecciona modelos (decisiones solo con validación)
	$(CLI) train

evaluate:         ## Evaluación final en test (registrada), LIAR y criterios de éxito
	$(CLI) evaluate

data-external:    ## (it2) Descarga WELFake y lo prepara como externo (dev + holdout)
	$(CLI) download-external
	$(CLI) prepare-external

shortcuts:        ## (it2) Auditoría de atajos de un solo rasgo
	$(CLI) shortcuts

panel:            ## (it2) Panel de evaluación (grupo, tiempo, externo-dev, invarianza) y selección
	$(CLI) panel

train-v2:         ## (it2) Modelo final v2 con OOD y abstención
	$(CLI) train-v2

evaluate-v2:      ## (it2) Evaluación final única: ISOT test + WELFake holdout (registrada)
	$(CLI) evaluate-v2

quickstart:       ## Inicio rápido (~5 min): datos mínimos + SOLO el modelo productivo v2
	$(CLI) quickstart

analyze:          ## Clustering y algoritmo genético (exploratorios)
	$(CLI) analyze

notebook:         ## Ejecuta el notebook de EDA de punta a punta
	$(BIN)/jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_y_preparacion.ipynb --ExecutePreprocessor.timeout=1200

test:             ## Suite completa de tests (incluye notebook y chequeos sobre artefactos)
	$(BIN)/pytest -q

test-fast:        ## Solo tests rápidos (sin notebook ni dataset completo)
	$(BIN)/pytest -q -m "not slow"

demo:             ## Levanta la demo en Streamlit
	$(BIN)/streamlit run app/demo.py

predict:          ## Ejemplo: make predict TEXT="..."
	$(CLI) predict "$(TEXT)"

all: data train evaluate data-external shortcuts panel train-v2 evaluate-v2 analyze notebook test

clean-artifacts:  ## Borra datos procesados, modelos y figuras generadas (no toca data/raw)
	rm -rf data/interim/* data/processed/* models/*.joblib models/candidates reports/figures/*
