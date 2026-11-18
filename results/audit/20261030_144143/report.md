# Informe de auditoría GWHD 2021 (20261030_144143)

**Veredicto:** Sin errores críticos detectados en esta ejecución.

## Errores críticos (deben ser 0 para continuar)
| comprobación | valor |
| --- | --- |
| filas_CSV_sin_archivo_de_imagen | 0 |
| imagenes_ilegibles_o_corruptas | 0 |
| imagenes_con_errores_de_parseo_de_cajas | 0 |
| cajas_no_finitas_o_tamano_no_positivo | 0 |
| image_name_en_varios_CSV_de_splits_distintos | 0 |

## Avisos (a decidir en la Fase 2)
| comprobación | valor |
| --- | --- |
| cajas_fuera_de_limites (recortables en Fase 2) | 19 |
| cajas_diminutas | 3 |
| imagenes_con_tamano_distinto_de_1024x1024 | 0 |
| grupos_duplicados_exactos_md5 | 1 |
| pares_casi_duplicados_pHash | 3 |
| pares_casi_duplicados_entre_splits_oficiales | 0 |
| imagenes_sin_espigas | 125 |
| archivos_huerfanos_en_disco | 3 |

## Totales
- Filas en los 3 CSV de la competición: 6515
- Imágenes únicas auditadas: 6512
- Instancias (espigas) en esas imágenes: 275357
- Dominios: 47

## Por split oficial de la competición (no es nuestra partición 70/15/15)
| competition_split | n_images | n_boxes | mean_boxes |
| --- | --- | --- | --- |
| test | 1381 | 67407 | 48.81 |
| train | 3655 | 163603 | 44.76 |
| val | 1476 | 44347 | 30.05 |

## Espigas por imagen
| variable | count | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n_boxes | 6512.0 | 42.28 | 23.18 | 0.0 | 26.0 | 39.0 | 56.0 | 190.0 |

## Tamaños de imagen
| size | n_images |
| --- | --- |
| 1024x1024 | 6512 |

## Dominios
| domain | n_images | n_boxes | mean_boxes | median_boxes | country | location | development_stage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARC_1 | 30 | 1169 | 38.97 | 38.5 | Sudan | Wad Medani | Filling |
| Arvalis_1 | 66 | 2935 | 44.47 | 44.0 | France | Gréoux | Post-flowering |
| Arvalis_10 | 60 | 1563 | 26.05 | 26.5 | France | Mons | Filling |
| Arvalis_11 | 60 | 2818 | 46.97 | 46.5 | France | VLB | Filling |
| Arvalis_12 | 27 | 1190 | 44.07 | 44.0 | France | Gréoux | Filling |
| Arvalis_2 | 401 | 21003 | 52.38 | 50.0 | France | Gréoux | Filling |
| Arvalis_3 | 588 | 21893 | 37.23 | 36.0 | France | Gréoux | Filling - Ripening |
| Arvalis_4 | 204 | 4270 | 20.93 | 21.0 | France | Gréoux | Filling |
| Arvalis_5 | 448 | 8180 | 18.26 | 16.0 | France | VLB | Filling |
| Arvalis_6 | 160 | 8698 | 54.36 | 53.0 | France | VSC | Filling - Ripening |
| Arvalis_7 | 24 | 1247 | 51.96 | 53.0 | France | VLB | Filling - Ripening |
| Arvalis_8 | 20 | 1062 | 53.1 | 54.0 | France | VLB | Filling - Ripening |
| Arvalis_9 | 32 | 1894 | 59.19 | 59.5 | France | VLB | Ripening |
| CIMMYT_1 | 69 | 2843 | 41.2 | 42.0 | Mexico | Ciudad Obregon | Post-flowering |
| CIMMYT_2 | 77 | 2771 | 35.99 | 37.0 | Mexico | Ciudad Obregon | Post-flowering |
| CIMMYT_3 | 59 | 1537 | 26.05 | 29.0 | Mexico | Ciudad Obregon | Post-flowering |
| ETHZ_1 | 747 | 49603 | 66.4 | 66.0 | Switzerland | Usask | Filling |
| Inrae_1 | 176 | 3634 | 20.65 | 21.0 | France | Toulouse | Filling - Ripening |
| KSU_1 | 100 | 6435 | 64.35 | 64.0 | US | KSU | Post-flowering |
| KSU_2 | 100 | 5302 | 53.02 | 54.0 | US | KSU | Post-flowering |
| KSU_3 | 95 | 5217 | 54.92 | 55.0 | US | KSU | Filling |
| KSU_4 | 60 | 3285 | 54.75 | 55.5 | US | KSU | Ripening |
| NAU_1 | 20 | 1240 | 62.0 | 64.5 | China | Baima | Post-flowering |
| NAU_2 | 100 | 4918 | 49.18 | 48.5 | China | Baima | Post-flowering |
| NAU_3 | 100 | 4596 | 45.96 | 41.5 | China | Baima | Filling |
| NMBU_1 | 82 | 7345 | 89.57 | 90.0 | Norway | NMBU | Filling |
| NMBU_2 | 98 | 5211 | 53.17 | 55.0 | Norway | NMBU | Ripening |
| Rres_1 | 432 | 19210 | 44.47 | 45.0 | UK | Rothamsted | Filling - Ripening |
| Terraref_1 | 144 | 3360 | 23.33 | 23.0 | US | Maricopa,AZ | Ripening |
| Terraref_2 | 106 | 1274 | 12.02 | 9.5 | US |  Maricopa,AZ | Filling |
| ULiège-GxABT_1 | 30 | 1847 | 61.57 | 60.0 | Belgium | Gembloux | Ripening |
| UQ_1 | 22 | 640 | 29.09 | 33.0 | Australia | Gatton | Post-flowering |
| UQ_10 | 106 | 8629 | 81.41 | 80.0 | Australia | Gatton | Filling - Ripening |
| UQ_11 | 84 | 4345 | 51.73 | 48.0 | Australia | Gatton | Post-flowering |
| UQ_2 | 16 | 39 | 2.44 | 0.0 | Australia | Gatton | Post-flowering |
| UQ_3 | 14 | 297 | 21.21 | 20.5 | Australia | Gatton | Filling |
| UQ_4 | 30 | 1039 | 34.63 | 34.0 | Australia | Gatton | Filling |
| UQ_5 | 30 | 3680 | 122.67 | 132.5 | Australia | Gatton | Filling - Ripening |
| UQ_6 | 30 | 1147 | 38.23 | 35.5 | Australia | Gatton | Filling - Ripening |
| UQ_7 | 17 | 1335 | 78.53 | 82.0 | Australia | Gatton | Ripening |
| UQ_8 | 41 | 4835 | 117.93 | 115.0 | Australia | McAllister | Ripening |
| UQ_9 | 33 | 2886 | 87.45 | 93.0 | Australia | Brookstead | Filling - Ripening |
| Ukyoto_1 | 60 | 2670 | 44.5 | 47.0 | Japan | Kyoto | Post-Flowering |
| Usask_1 | 200 | 5985 | 29.92 | 30.0 | Canada | Saskatchewan | Filling - Ripening |
| Utokyo_1 | 538 | 14185 | 26.37 | 26.0 | Japan | NARO-Tsukuba | Ripening |
| Utokyo_2 | 456 | 13010 | 28.53 | 29.0 | Japan | NARO-Tsukuba | Ripening |
| Utokyo_3 | 120 | 3085 | 25.71 | 25.0 | Japan | NARO-Hokkaido | multiple |

## Metadata.csv
- Archivos encontrados: 1
- Ruta: C:\dev\gwhd-project\data\raw\gwhd_2021\metadata_dataset.csv
- Dimensiones: [47, 4]
- Columnas: ['name', 'country', 'location', 'development_stage']
