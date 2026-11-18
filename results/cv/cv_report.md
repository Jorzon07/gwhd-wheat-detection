# Validación cruzada RT-DETR (configuración Hyperband, validación de cada pliegue)

| variante | pliegue | n_train | n_val | ubic. compartidas | mAP50 | mAP50-95 | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 0 | 3677 | 1839 | 14 | 0.923 | 0.535 | 0.898 |
| random | 1 | 3677 | 1839 | 14 | 0.923 | 0.533 | 0.896 |
| random | 2 | 3678 | 1838 | 14 | 0.926 | 0.537 | 0.898 |
| grouped | 0 | 3666 | 1850 | 0 | 0.806 | 0.340 | 0.805 |
| grouped | 1 | 3669 | 1847 | 0 | 0.929 | 0.494 | 0.903 |
| grouped | 2 | 3697 | 1819 | 0 | 0.875 | 0.485 | 0.853 |

**random** (n=3 pliegues): mAP50 0.924 ± 0.002 | mAP50-95 0.535 ± 0.002 | F1 0.897
**grouped** (n=3 pliegues): mAP50 0.870 ± 0.062 | mAP50-95 0.440 ± 0.086 | F1 0.853

Inflación por fuga (aleatoria - agrupada): mAP50 +0.054 | mAP50-95 +0.096 | F1 +0.044

Nota: ± es la desviación típica entre pliegues (pocos pliegues: orientativa). Cada pliegue evalúa sobre sus propias imágenes, con eval cada 2 épocas; la mejor época se elige con el propio pliegue (sesgo optimista, igual en ambas variantes).