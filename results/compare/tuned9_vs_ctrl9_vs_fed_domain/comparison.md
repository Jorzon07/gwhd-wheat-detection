## Remuestreo por imágenes (1000 réplicas, IC 95 % percentil)

| model | metric | point | lo | hi |
| --- | --- | --- | --- | --- |
| tuned9 | map50 | 0.854 | 0.845 | 0.866 |
| tuned9 | map50_95 | 0.467 | 0.458 | 0.477 |
| tuned9 | f1 | 0.832 | 0.824 | 0.841 |
| ctrl9 | map50 | 0.840 | 0.829 | 0.851 |
| ctrl9 | map50_95 | 0.434 | 0.425 | 0.444 |
| ctrl9 | f1 | 0.819 | 0.810 | 0.828 |
| fed_domain | map50 | 0.780 | 0.765 | 0.795 |
| fed_domain | map50_95 | 0.400 | 0.389 | 0.411 |
| fed_domain | f1 | 0.770 | 0.759 | 0.782 |

Diferencias pareadas (modelo - tuned9):

| a | b | metric | mean_diff | lo | hi | excluye_cero |
| --- | --- | --- | --- | --- | --- | --- |
| ctrl9 | tuned9 | map50 | -0.015 | -0.018 | -0.012 | True |
| ctrl9 | tuned9 | map50_95 | -0.033 | -0.038 | -0.030 | True |
| ctrl9 | tuned9 | map75 | -0.070 | -0.078 | -0.062 | True |
| ctrl9 | tuned9 | precision | -0.026 | -0.030 | -0.021 | True |
| ctrl9 | tuned9 | recall | -0.003 | -0.007 | 0.000 | False |
| ctrl9 | tuned9 | f1 | -0.013 | -0.016 | -0.010 | True |
| fed_domain | tuned9 | map50 | -0.075 | -0.081 | -0.069 | True |
| fed_domain | tuned9 | map50_95 | -0.068 | -0.073 | -0.062 | True |
| fed_domain | tuned9 | map75 | -0.096 | -0.104 | -0.086 | True |
| fed_domain | tuned9 | precision | -0.047 | -0.055 | -0.039 | True |
| fed_domain | tuned9 | recall | -0.072 | -0.078 | -0.067 | True |
| fed_domain | tuned9 | f1 | -0.062 | -0.068 | -0.057 | True |

## Remuestreo por dominios completos (9) (1000 réplicas, IC 95 % percentil)

| model | metric | point | lo | hi |
| --- | --- | --- | --- | --- |
| tuned9 | map50 | 0.854 | 0.761 | 0.920 |
| tuned9 | map50_95 | 0.467 | 0.398 | 0.522 |
| tuned9 | f1 | 0.832 | 0.751 | 0.887 |
| ctrl9 | map50 | 0.840 | 0.739 | 0.906 |
| ctrl9 | map50_95 | 0.434 | 0.370 | 0.478 |
| ctrl9 | f1 | 0.819 | 0.736 | 0.875 |
| fed_domain | map50 | 0.780 | 0.640 | 0.876 |
| fed_domain | map50_95 | 0.400 | 0.315 | 0.464 |
| fed_domain | f1 | 0.770 | 0.659 | 0.846 |

Diferencias pareadas (modelo - tuned9):

| a | b | metric | mean_diff | lo | hi | excluye_cero |
| --- | --- | --- | --- | --- | --- | --- |
| ctrl9 | tuned9 | map50 | -0.015 | -0.022 | -0.010 | True |
| ctrl9 | tuned9 | map50_95 | -0.033 | -0.054 | -0.019 | True |
| ctrl9 | tuned9 | map75 | -0.069 | -0.111 | -0.039 | True |
| ctrl9 | tuned9 | precision | -0.026 | -0.037 | -0.017 | True |
| ctrl9 | tuned9 | recall | -0.003 | -0.016 | 0.006 | False |
| ctrl9 | tuned9 | f1 | -0.013 | -0.020 | -0.008 | True |
| fed_domain | tuned9 | map50 | -0.077 | -0.124 | -0.040 | True |
| fed_domain | tuned9 | map50_95 | -0.069 | -0.113 | -0.034 | True |
| fed_domain | tuned9 | map75 | -0.098 | -0.170 | -0.041 | True |
| fed_domain | tuned9 | precision | -0.050 | -0.094 | -0.017 | True |
| fed_domain | tuned9 | recall | -0.074 | -0.106 | -0.043 | True |
| fed_domain | tuned9 | f1 | -0.064 | -0.099 | -0.035 | True |

Notas: solo validación. El remuestreo por imagen subestima la incertidumbre porque las imágenes de un dominio están correlacionadas; el de dominios es más conservador pero con pocos dominios es muy ancho. Un solo entrenamiento por modelo: no incluye la variabilidad entre semillas (~0.01 en mAP50).