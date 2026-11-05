# data/

Nada de esta carpeta se versiona salvo este archivo. Los datos se descargan y procesan con:

```bash
make data    # = python -m fakenews.cli download && ... prepare && ... audit
```

| Carpeta | Contenido | Origen |
|---|---|---|
| `raw/isot_kaggle/` | `Fake.csv`, `True.csv` (+ zip original) | Kaggle `clmentbisaillon/fake-and-real-news-dataset` (endpoint público, sin login) |
| `raw/liar/` | `train.tsv`, `valid.tsv`, `test.tsv`, `README` | https://www.cs.ucsb.edu/~william/data/liar_dataset.zip |
| `interim/` | predicciones intermedias (`val_predictions.npz`, `test_predictions.npz`) | generado |
| `processed/` | `train.csv`, `val.csv`, `test.csv` (texto limpio `doc`, texto mínimo `doc_minimal`) | generado |
| `external/` | reservado para otros datasets externos | — |

Los SHA-256 de los archivos crudos están en `config.yaml` (sección `download`) y los de los splits
en `reports/splits_manifest.json`. Descripción, licencias y sesgos: `docs/data_card.md`.

Si la descarga automática falla (por ejemplo, si Kaggle cambia su endpoint), bajá a mano:
- https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset → dejar `Fake.csv` y `True.csv` en `data/raw/isot_kaggle/`
- https://www.cs.ucsb.edu/~william/data/liar_dataset.zip → descomprimir en `data/raw/liar/`
