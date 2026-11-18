## Remuestreo por imágenes (1000 réplicas, IC 95 % percentil)

| model | metric | point | lo | hi |
| --- | --- | --- | --- | --- |
| frcnn | map50 | 0.791 | 0.774 | 0.806 |
| frcnn | map50_95 | 0.404 | 0.393 | 0.415 |
| frcnn | f1 | 0.808 | 0.797 | 0.819 |
| retinanet | map50 | 0.741 | 0.727 | 0.754 |
| retinanet | map50_95 | 0.389 | 0.377 | 0.399 |
| retinanet | f1 | 0.786 | 0.776 | 0.797 |

Diferencias pareadas (modelo - frcnn):

| a | b | metric | mean_diff | lo | hi | excluye_cero |
| --- | --- | --- | --- | --- | --- | --- |
| retinanet | frcnn | map50 | -0.050 | -0.053 | -0.045 | True |
| retinanet | frcnn | map50_95 | -0.016 | -0.018 | -0.013 | True |
| retinanet | frcnn | map75 | 0.002 | -0.003 | 0.008 | False |
| retinanet | frcnn | precision | -0.010 | -0.014 | -0.006 | True |
| retinanet | frcnn | recall | -0.030 | -0.033 | -0.026 | True |
| retinanet | frcnn | f1 | -0.022 | -0.024 | -0.019 | True |

## Remuestreo por dominios completos (9) (1000 réplicas, IC 95 % percentil)

| model | metric | point | lo | hi |
| --- | --- | --- | --- | --- |
| frcnn | map50 | 0.791 | 0.651 | 0.892 |
| frcnn | map50_95 | 0.404 | 0.321 | 0.474 |
| frcnn | f1 | 0.808 | 0.700 | 0.881 |
| retinanet | map50 | 0.741 | 0.600 | 0.845 |
| retinanet | map50_95 | 0.389 | 0.311 | 0.456 |
| retinanet | f1 | 0.786 | 0.684 | 0.856 |

Diferencias pareadas (modelo - frcnn):

| a | b | metric | mean_diff | lo | hi | excluye_cero |
| --- | --- | --- | --- | --- | --- | --- |
| retinanet | frcnn | map50 | -0.049 | -0.065 | -0.028 | True |
| retinanet | frcnn | map50_95 | -0.015 | -0.023 | -0.006 | True |
| retinanet | frcnn | map75 | 0.002 | -0.012 | 0.018 | False |
| retinanet | frcnn | precision | -0.009 | -0.021 | 0.005 | False |
| retinanet | frcnn | recall | -0.029 | -0.041 | -0.016 | True |
| retinanet | frcnn | f1 | -0.021 | -0.027 | -0.013 | True |

Notas: solo validación. El remuestreo por imagen subestima la incertidumbre porque las imágenes de un dominio están correlacionadas; el de dominios es más conservador pero con pocos dominios es muy ancho. Un solo entrenamiento por modelo: no incluye la variabilidad entre semillas (~0.01 en mAP50).