# Informe de partición (location, semilla 42)

Auditoría de referencia: `20261030_144143`. Generado: 2026-11-01T07:51:25.

## Conjuntos
| split | n_images | n_instances | mean_boxes | n_groups | pct_images | pct_instances | target_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4532 | 193128 | 42.61 | 9 | 69.65 | 70.16 | 70.0 |
| validation | 984 | 40275 | 40.93 | 5 | 15.12 | 14.63 | 15.0 |
| test | 991 | 41846 | 42.23 | 7 | 15.23 | 15.2 | 15.0 |

Objetivo: {'train': 0.7, 'validation': 0.15, 'test': 0.15}. Desviación máxima lograda: 0.352 pp (imágenes), 0.368 pp (espigas); tolerancia pedida 0.5 pp, usada 0.5 pp.
Estado del solver: 0 (Optimization terminated successfully. (HiGHS Status 7: Optimal)).

## Limpieza
- Imágenes en la tabla limpia: 6507
- Cuarentena / exclusiones: {'repeated_name_conflicting_rows': 3, 'orphan_file_no_annotation': 3, 'exact_duplicate_conflicting_labels': 2}
- Eventos de cajas: {'clipped': 12, 'dropped_outside': 7, 'dropped_tiny': 3}
- Imágenes que quedaron sin cajas por la limpieza (antes tenían): 0
- Imágenes sin espigas (incluye negativos originales): 124

## Composición
| split | n_images | n_domains | n_locations | n_countries | largest_domain | largest_domain_share_pct |
| --- | --- | --- | --- | --- | --- | --- |
| train | 4532 | 21 | 9 | 6 | ETHZ_1 | 16.5 |
| validation | 984 | 9 | 5 | 4 | Inrae_1 | 17.9 |
| test | 991 | 17 | 7 | 7 | Usask_1 | 20.2 |

Países por conjunto (imágenes): train: France:1870; Japan:1054; Switzerland:747; UK:432; US:355; Australia:74
validation: France:336; US:248; China:220; Norway:180
test: Australia:349; Mexico:204; Canada:200; Japan:120; France:58; Belgium:30; Sudan:30

Imágenes de val/test en países **sin ninguna imagen en train**:
| split | unseen_countries | n_images_unseen | pct_of_set |
| --- | --- | --- | --- |
| validation | China:220; Norway:180 | 400 | 40.7 |
| test | Belgium:30; Canada:200; Mexico:204; Sudan:30 | 464 | 46.8 |

Imágenes de val/test en una ubicación que también tiene imágenes en train: {'validation': 0, 'test': 0}

Restricciones activas: max_domain_share=0.25, min_countries=4 (con >= 30 imágenes), max_unseen_country_share=0.5.

## Limitaciones conocidas
- **Agrupación por ubicación.** Ninguna ubicación aparece en más de un conjunto (verificado). Consecuencia: 40.7 % de las imágenes de validación y 46.8 % de las de test proceden de países sin imágenes en train, de modo que las métricas mezclan generalización a ubicaciones nuevas y a países nuevos. Es una estimación más exigente, no comparable con la de una partición por dominio.
- La metadata asigna a ETHZ_1 la ubicación 'Usask' (distinta de 'Saskatchewan', de Usask_1); se trató como una ubicación propia sin verificar si es un error de la metadata.
- Esta partición no es la oficial de la competición GWHD 2021 (que separa por dominio con otras proporciones): los resultados no son comparables con su clasificación.
- Las proporciones se miden sobre imágenes y espigas a la vez; la asignación es una de varias particiones válidas dentro de la tolerancia, elegida por la semilla.
