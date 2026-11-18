# Informe de partición (domain, semilla 42)

Auditoría de referencia: `20261030_144143`. Generado: 2026-10-31T10:49:05.

## Conjuntos
| split | n_images | n_instances | mean_boxes | n_groups | pct_images | pct_instances | target_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4550 | 192804 | 42.37 | 22 | 69.92 | 70.05 | 70.0 |
| validation | 982 | 41366 | 42.12 | 14 | 15.09 | 15.03 | 15.0 |
| test | 975 | 41079 | 42.13 | 11 | 14.98 | 14.92 | 15.0 |

Objetivo: {'train': 0.7, 'validation': 0.15, 'test': 0.15}. Desviación máxima lograda: 0.091 pp (imágenes), 0.076 pp (espigas); tolerancia pedida 0.1 pp, usada 0.1 pp.
Estado del solver: 0 (Optimization terminated successfully. (HiGHS Status 7: Optimal)).

## Limpieza
- Imágenes en la tabla limpia: 6507
- Cuarentena / exclusiones: {'repeated_name_conflicting_rows': 3, 'orphan_file_no_annotation': 3, 'exact_duplicate_conflicting_labels': 2}
- Eventos de cajas: {'clipped': 12, 'dropped_outside': 7, 'dropped_tiny': 3}
- Imágenes que quedaron sin cajas por la limpieza (antes tenían): 0
- Imágenes sin espigas (incluye negativos originales): 124

## Limitaciones conocidas
- **Agrupación por domain.** 5 ubicaciones tienen imágenes en más de un conjunto (2794 imágenes en total); detalle en `locations_split_across_sets.csv`. Sesiones del mismo sitio pueden compartir campo, variedades o sensor, lo que podría hacer optimistas las métricas de test. No se ha medido ese efecto.
- Esta partición no es la oficial de la competición GWHD 2021 (que separa por dominio con otras proporciones): los resultados no son comparables con su clasificación.
- Las proporciones se miden sobre imágenes y espigas a la vez; la asignación es una de varias particiones válidas dentro de la tolerancia, elegida por la semilla.
