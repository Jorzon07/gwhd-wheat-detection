## Remuestreo por imágenes (1000 réplicas, IC 95 % percentil)

| model | metric | point | lo | hi |
| --- | --- | --- | --- | --- |
| frcnn | map50 | 0.791 | 0.774 | 0.806 |
| frcnn | map50_95 | 0.404 | 0.393 | 0.415 |
| frcnn | f1 | 0.808 | 0.797 | 0.819 |
| retinanet | map50 | 0.741 | 0.727 | 0.754 |
| retinanet | map50_95 | 0.389 | 0.377 | 0.399 |
| retinanet | f1 | 0.786 | 0.776 | 0.797 |
| rtdetr | map50 | 0.842 | 0.831 | 0.853 |
| rtdetr | map50_95 | 0.449 | 0.439 | 0.458 |
| rtdetr | f1 | 0.821 | 0.812 | 0.830 |
| detr | map50 | 0.729 | 0.715 | 0.744 |
| detr | map50_95 | 0.334 | 0.324 | 0.343 |
| detr | f1 | 0.756 | 0.746 | 0.767 |
| yolov8n | map50 | 0.818 | 0.806 | 0.831 |
| yolov8n | map50_95 | 0.432 | 0.423 | 0.441 |
| yolov8n | f1 | 0.798 | 0.788 | 0.808 |

Diferencias pareadas (modelo - frcnn):

| a | b | metric | mean_diff | lo | hi | excluye_cero |
| --- | --- | --- | --- | --- | --- | --- |
| retinanet | frcnn | map50 | -0.050 | -0.053 | -0.045 | True |
| retinanet | frcnn | map50_95 | -0.016 | -0.018 | -0.013 | True |
| retinanet | frcnn | map75 | 0.002 | -0.003 | 0.008 | False |
| retinanet | frcnn | precision | -0.010 | -0.014 | -0.006 | True |
| retinanet | frcnn | recall | -0.030 | -0.033 | -0.026 | True |
| retinanet | frcnn | f1 | -0.022 | -0.024 | -0.019 | True |
| rtdetr | frcnn | map50 | 0.052 | 0.045 | 0.059 | True |
| rtdetr | frcnn | map50_95 | 0.045 | 0.040 | 0.050 | True |
| rtdetr | frcnn | map75 | 0.062 | 0.054 | 0.070 | True |
| rtdetr | frcnn | precision | -0.004 | -0.010 | 0.001 | False |
| rtdetr | frcnn | recall | 0.026 | 0.021 | 0.032 | True |
| rtdetr | frcnn | f1 | 0.013 | 0.009 | 0.017 | True |
| detr | frcnn | map50 | -0.061 | -0.066 | -0.055 | True |
| detr | frcnn | map50_95 | -0.070 | -0.076 | -0.064 | True |
| detr | frcnn | map75 | -0.103 | -0.115 | -0.092 | True |
| detr | frcnn | precision | -0.059 | -0.064 | -0.054 | True |
| detr | frcnn | recall | -0.046 | -0.051 | -0.040 | True |
| detr | frcnn | f1 | -0.052 | -0.056 | -0.047 | True |
| yolov8n | frcnn | map50 | 0.028 | 0.022 | 0.035 | True |
| yolov8n | frcnn | map50_95 | 0.029 | 0.023 | 0.034 | True |
| yolov8n | frcnn | map75 | 0.045 | 0.035 | 0.055 | True |
| yolov8n | frcnn | precision | -0.009 | -0.015 | -0.003 | True |
| yolov8n | frcnn | recall | -0.010 | -0.015 | -0.005 | True |
| yolov8n | frcnn | f1 | -0.010 | -0.014 | -0.006 | True |

## Remuestreo por dominios completos (9) (1000 réplicas, IC 95 % percentil)

| model | metric | point | lo | hi |
| --- | --- | --- | --- | --- |
| frcnn | map50 | 0.791 | 0.651 | 0.892 |
| frcnn | map50_95 | 0.404 | 0.321 | 0.474 |
| frcnn | f1 | 0.808 | 0.700 | 0.881 |
| retinanet | map50 | 0.741 | 0.600 | 0.845 |
| retinanet | map50_95 | 0.389 | 0.311 | 0.456 |
| retinanet | f1 | 0.786 | 0.684 | 0.856 |
| rtdetr | map50 | 0.842 | 0.740 | 0.910 |
| rtdetr | map50_95 | 0.449 | 0.379 | 0.500 |
| rtdetr | f1 | 0.821 | 0.736 | 0.880 |
| detr | map50 | 0.729 | 0.598 | 0.828 |
| detr | map50_95 | 0.334 | 0.260 | 0.393 |
| detr | f1 | 0.756 | 0.658 | 0.827 |
| yolov8n | map50 | 0.818 | 0.696 | 0.900 |
| yolov8n | map50_95 | 0.432 | 0.359 | 0.478 |
| yolov8n | f1 | 0.798 | 0.699 | 0.864 |

Diferencias pareadas (modelo - frcnn):

| a | b | metric | mean_diff | lo | hi | excluye_cero |
| --- | --- | --- | --- | --- | --- | --- |
| retinanet | frcnn | map50 | -0.049 | -0.065 | -0.028 | True |
| retinanet | frcnn | map50_95 | -0.015 | -0.023 | -0.006 | True |
| retinanet | frcnn | map75 | 0.002 | -0.012 | 0.018 | False |
| retinanet | frcnn | precision | -0.009 | -0.021 | 0.005 | False |
| retinanet | frcnn | recall | -0.029 | -0.041 | -0.016 | True |
| retinanet | frcnn | f1 | -0.021 | -0.027 | -0.013 | True |
| rtdetr | frcnn | map50 | 0.051 | 0.015 | 0.095 | True |
| rtdetr | frcnn | map50_95 | 0.045 | 0.022 | 0.073 | True |
| rtdetr | frcnn | map75 | 0.062 | 0.027 | 0.101 | True |
| rtdetr | frcnn | precision | -0.004 | -0.024 | 0.015 | False |
| rtdetr | frcnn | recall | 0.026 | -0.002 | 0.062 | False |
| rtdetr | frcnn | f1 | 0.014 | -0.005 | 0.039 | False |
| detr | frcnn | map50 | -0.061 | -0.086 | -0.042 | True |
| detr | frcnn | map50_95 | -0.071 | -0.120 | -0.033 | True |
| detr | frcnn | map75 | -0.106 | -0.204 | -0.031 | True |
| detr | frcnn | precision | -0.060 | -0.085 | -0.038 | True |
| detr | frcnn | recall | -0.046 | -0.078 | -0.018 | True |
| detr | frcnn | f1 | -0.052 | -0.066 | -0.039 | True |
| yolov8n | frcnn | map50 | 0.028 | -0.001 | 0.068 | False |
| yolov8n | frcnn | map50_95 | 0.028 | -0.005 | 0.064 | False |
| yolov8n | frcnn | map75 | 0.044 | -0.021 | 0.101 | False |
| yolov8n | frcnn | precision | -0.009 | -0.034 | 0.013 | False |
| yolov8n | frcnn | recall | -0.011 | -0.032 | 0.011 | False |
| yolov8n | frcnn | f1 | -0.010 | -0.025 | 0.008 | False |

Notas: solo validación. El remuestreo por imagen subestima la incertidumbre porque las imágenes de un dominio están correlacionadas; el de dominios es más conservador pero con pocos dominios es muy ancho. Un solo entrenamiento por modelo: no incluye la variabilidad entre semillas (~0.01 en mAP50).