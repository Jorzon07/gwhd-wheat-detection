# Detección de espigas de trigo en GWHD 2021

Proyecto de Inteligencia Artificial (Ingeniería de Software, Universidad Surcolombiana).
Autor: Jorge Iván Garzón Sánchez (código 20221203156).

Detección de espigas sobre el conjunto [GWHD 2021](https://zenodo.org/records/5092309) (6 507 imágenes limpias, 47 dominios):
auditoría y limpieza, partición sin fuga por ubicación (70/15/15), cinco detectores con un evaluador común, Hyperband,
validación cruzada, **aprendizaje federado (FedAvg)**, ensambles WBF, aumento con pix2pix y benchmark de inferencia.

**Informe técnico:** [`docs/informe_gwhd2021.pdf`](docs/informe_gwhd2021.pdf)

## Resultados principales (test, 991 imágenes, una semilla por modelo)

| Modelo | mAP50-95 | F1 |
| --- | --- | --- |
| RT-DETR-R18 ajustado (Hyperband) | 0,382 | 0,789 |
| RT-DETR por defecto | 0,368 | 0,774 |
| YOLOv8n | 0,364 | 0,754 |
| Faster R-CNN | 0,343 | 0,770 |
| RetinaNet | 0,320 | 0,736 |
| DETR | 0,252 | 0,679 |
| FedAvg, 1 cliente | 0,377 | 0,791 |
| FedAvg, 21 clientes IID | 0,361 | 0,754 |
| **FedAvg, 21 clientes por dominio** | **0,305** | 0,683 |
| RT-DETR + imágenes sintéticas (pix2pix) | 0,386 | 0,793 |
| Ensamble WBF (RT-DETR + YOLOv8n + Faster R-CNN) | 0,385 | 0,792 |

El ruido de entrenamiento observado es de ~0,015 mAP50-95: las diferencias menores no son interpretables (ver limitaciones en el informe).
La partición agrupa por ubicación; 40–47 % de val/test proceden de países sin imágenes en train.

## Estructura

```
configs/data.yaml      rutas y parámetros de datos
src/data/              descarga, auditoría, limpieza, particiones, dataset
src/models/            fábrica de detectores (RT-DETR, DETR, Faster R-CNN, RetinaNet)
src/training/          entrenamiento, federado (federated.py), YOLO
src/evaluation/        evaluador común, bootstrap, ensambles, prueba final, benchmark
src/gan/               pix2pix condicionado por cajas
scripts/               Hyperband, validación cruzada, cadena GAN
tests/                 pruebas unitarias
results/               informes, tablas y métricas por ejecución (sin pesos ni predicciones)
docs/                  informe en PDF
```

## Reproducir

Python 3.12, PyTorch 2.15 con CUDA (probado en una RTX 4050 de 6 GB, Windows).

```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-phase1.txt -r requirements-phase2.txt -r requirements-phase3.txt -r requirements-phase4.txt
```

Los datos, los pesos y las predicciones **no están en el repositorio** (pesan decenas de GB). Los datos se descargan de Zenodo
(la ruta y el md5 están en `configs/data.yaml`) y las particiones se regeneran con semilla 42 (`src/data/make_splits.py`).
Los resultados de cada experimento están en `results/`.

Federado: `python -m src.training.federated --partition domain --rounds 9` (también `iid` y `single`).
Pruebas: `python -m pytest tests`.

## Notas

- La evaluación en test se ejecutó una sola vez (`results/final/lock.json` guarda los hashes de los pesos).
- `scripts/apply_*.py` son parches históricos aplicados una vez durante el desarrollo; no vuelvas a ejecutarlos.
