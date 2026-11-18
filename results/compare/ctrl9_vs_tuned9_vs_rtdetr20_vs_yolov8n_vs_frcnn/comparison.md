## Remuestreo por imágenes (1000 réplicas, IC 95 % percentil)

| model | metric | point | lo | hi |
| --- | --- | --- | --- | --- |
| ctrl9 | map50 | 0.840 | 0.829 | 0.851 |
| ctrl9 | map50_95 | 0.434 | 0.425 | 0.444 |
| ctrl9 | f1 | 0.819 | 0.810 | 0.828 |
| tuned9 | map50 | 0.854 | 0.845 | 0.866 |
| tuned9 | map50_95 | 0.467 | 0.458 | 0.477 |
| tuned9 | f1 | 0.832 | 0.824 | 0.841 |
| rtdetr20 | map50 | 0.842 | 0.831 | 0.853 |
| rtdetr20 | map50_95 | 0.449 | 0.439 | 0.458 |
| rtdetr20 | f1 | 0.821 | 0.812 | 0.830 |
| yolov8n | map50 | 0.818 | 0.806 | 0.831 |
| yolov8n | map50_95 | 0.432 | 0.423 | 0.441 |
| yolov8n | f1 | 0.798 | 0.788 | 0.808 |
| frcnn | map50 | 0.791 | 0.774 | 0.806 |
| frcnn | map50_95 | 0.404 | 0.393 | 0.415 |
| frcnn | f1 | 0.808 | 0.797 | 0.819 |

Diferencias pareadas (modelo - ctrl9):

| a | b | metric | mean_diff | lo | hi | excluye_cero |
| --- | --- | --- | --- | --- | --- | --- |
| tuned9 | ctrl9 | map50 | 0.015 | 0.012 | 0.018 | True |
| tuned9 | ctrl9 | map50_95 | 0.033 | 0.030 | 0.038 | True |
| tuned9 | ctrl9 | map75 | 0.070 | 0.062 | 0.078 | True |
| tuned9 | ctrl9 | precision | 0.026 | 0.021 | 0.030 | True |
| tuned9 | ctrl9 | recall | 0.003 | -0.000 | 0.007 | False |
| tuned9 | ctrl9 | f1 | 0.013 | 0.010 | 0.016 | True |
| rtdetr20 | ctrl9 | map50 | 0.002 | -0.000 | 0.004 | False |
| rtdetr20 | ctrl9 | map50_95 | 0.014 | 0.012 | 0.017 | True |
| rtdetr20 | ctrl9 | map75 | 0.034 | 0.029 | 0.039 | True |
| rtdetr20 | ctrl9 | precision | 0.009 | 0.005 | 0.012 | True |
| rtdetr20 | ctrl9 | recall | -0.003 | -0.006 | -0.001 | True |
| rtdetr20 | ctrl9 | f1 | 0.002 | -0.000 | 0.004 | False |
| yolov8n | ctrl9 | map50 | -0.022 | -0.026 | -0.018 | True |
| yolov8n | ctrl9 | map50_95 | -0.002 | -0.006 | 0.002 | False |
| yolov8n | ctrl9 | map75 | 0.018 | 0.010 | 0.025 | True |
| yolov8n | ctrl9 | precision | 0.004 | -0.002 | 0.010 | False |
| yolov8n | ctrl9 | recall | -0.040 | -0.046 | -0.035 | True |
| yolov8n | ctrl9 | f1 | -0.021 | -0.025 | -0.017 | True |
| frcnn | ctrl9 | map50 | -0.050 | -0.057 | -0.043 | True |
| frcnn | ctrl9 | map50_95 | -0.030 | -0.036 | -0.025 | True |
| frcnn | ctrl9 | map75 | -0.028 | -0.037 | -0.018 | True |
| frcnn | ctrl9 | precision | 0.013 | 0.008 | 0.018 | True |
| frcnn | ctrl9 | recall | -0.030 | -0.035 | -0.024 | True |
| frcnn | ctrl9 | f1 | -0.011 | -0.016 | -0.007 | True |

## Remuestreo por dominios completos (9) (1000 réplicas, IC 95 % percentil)

| model | metric | point | lo | hi |
| --- | --- | --- | --- | --- |
| ctrl9 | map50 | 0.840 | 0.739 | 0.906 |
| ctrl9 | map50_95 | 0.434 | 0.370 | 0.478 |
| ctrl9 | f1 | 0.819 | 0.736 | 0.875 |
| tuned9 | map50 | 0.854 | 0.761 | 0.920 |
| tuned9 | map50_95 | 0.467 | 0.398 | 0.522 |
| tuned9 | f1 | 0.832 | 0.751 | 0.887 |
| rtdetr20 | map50 | 0.842 | 0.740 | 0.910 |
| rtdetr20 | map50_95 | 0.449 | 0.379 | 0.500 |
| rtdetr20 | f1 | 0.821 | 0.736 | 0.880 |
| yolov8n | map50 | 0.818 | 0.696 | 0.900 |
| yolov8n | map50_95 | 0.432 | 0.359 | 0.478 |
| yolov8n | f1 | 0.798 | 0.699 | 0.864 |
| frcnn | map50 | 0.791 | 0.651 | 0.892 |
| frcnn | map50_95 | 0.404 | 0.321 | 0.474 |
| frcnn | f1 | 0.808 | 0.700 | 0.881 |

Diferencias pareadas (modelo - ctrl9):

| a | b | metric | mean_diff | lo | hi | excluye_cero |
| --- | --- | --- | --- | --- | --- | --- |
| tuned9 | ctrl9 | map50 | 0.015 | 0.010 | 0.022 | True |
| tuned9 | ctrl9 | map50_95 | 0.033 | 0.019 | 0.054 | True |
| tuned9 | ctrl9 | map75 | 0.069 | 0.039 | 0.111 | True |
| tuned9 | ctrl9 | precision | 0.026 | 0.017 | 0.037 | True |
| tuned9 | ctrl9 | recall | 0.003 | -0.006 | 0.016 | False |
| tuned9 | ctrl9 | f1 | 0.013 | 0.008 | 0.020 | True |
| rtdetr20 | ctrl9 | map50 | 0.002 | -0.004 | 0.007 | False |
| rtdetr20 | ctrl9 | map50_95 | 0.014 | 0.003 | 0.028 | True |
| rtdetr20 | ctrl9 | map75 | 0.034 | 0.013 | 0.059 | True |
| rtdetr20 | ctrl9 | precision | 0.008 | 0.001 | 0.014 | True |
| rtdetr20 | ctrl9 | recall | -0.003 | -0.010 | 0.004 | False |
| rtdetr20 | ctrl9 | f1 | 0.002 | -0.004 | 0.007 | False |
| yolov8n | ctrl9 | map50 | -0.022 | -0.046 | -0.007 | True |
| yolov8n | ctrl9 | map50_95 | -0.003 | -0.016 | 0.009 | False |
| yolov8n | ctrl9 | map75 | 0.016 | -0.009 | 0.043 | False |
| yolov8n | ctrl9 | precision | 0.004 | -0.016 | 0.025 | False |
| yolov8n | ctrl9 | recall | -0.040 | -0.066 | -0.018 | True |
| yolov8n | ctrl9 | f1 | -0.022 | -0.041 | -0.008 | True |
| frcnn | ctrl9 | map50 | -0.050 | -0.097 | -0.011 | True |
| frcnn | ctrl9 | map50_95 | -0.031 | -0.066 | -0.000 | True |
| frcnn | ctrl9 | map75 | -0.028 | -0.075 | 0.021 | False |
| frcnn | ctrl9 | precision | 0.013 | -0.008 | 0.032 | False |
| frcnn | ctrl9 | recall | -0.030 | -0.071 | 0.004 | False |
| frcnn | ctrl9 | f1 | -0.012 | -0.042 | 0.009 | False |

Notas: solo validación. El remuestreo por imagen subestima la incertidumbre porque las imágenes de un dominio están correlacionadas; el de dominios es más conservador pero con pocos dominios es muy ancho. Un solo entrenamiento por modelo: no incluye la variabilidad entre semillas (~0.01 en mAP50).