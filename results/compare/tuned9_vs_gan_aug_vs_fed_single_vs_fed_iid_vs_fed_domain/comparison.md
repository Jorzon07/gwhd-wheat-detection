## Remuestreo por imágenes (1000 réplicas, IC 95 % percentil)

| model | metric | point | lo | hi |
| --- | --- | --- | --- | --- |
| tuned9 | map50 | 0.854 | 0.845 | 0.866 |
| tuned9 | map50_95 | 0.467 | 0.458 | 0.477 |
| tuned9 | f1 | 0.832 | 0.824 | 0.841 |
| gan_aug | map50 | 0.850 | 0.838 | 0.860 |
| gan_aug | map50_95 | 0.460 | 0.449 | 0.470 |
| gan_aug | f1 | 0.833 | 0.824 | 0.842 |
| fed_single | map50 | 0.859 | 0.849 | 0.870 |
| fed_single | map50_95 | 0.464 | 0.454 | 0.474 |
| fed_single | f1 | 0.839 | 0.831 | 0.848 |
| fed_iid | map50 | 0.817 | 0.806 | 0.830 |
| fed_iid | map50_95 | 0.428 | 0.419 | 0.438 |
| fed_iid | f1 | 0.796 | 0.786 | 0.806 |
| fed_domain | map50 | 0.780 | 0.765 | 0.795 |
| fed_domain | map50_95 | 0.400 | 0.389 | 0.411 |
| fed_domain | f1 | 0.770 | 0.759 | 0.782 |

Diferencias pareadas (modelo - tuned9):

| a | b | metric | mean_diff | lo | hi | excluye_cero |
| --- | --- | --- | --- | --- | --- | --- |
| gan_aug | tuned9 | map50 | -0.006 | -0.009 | -0.003 | True |
| gan_aug | tuned9 | map50_95 | -0.008 | -0.011 | -0.006 | True |
| gan_aug | tuned9 | map75 | -0.019 | -0.024 | -0.014 | True |
| gan_aug | tuned9 | precision | 0.002 | -0.001 | 0.006 | False |
| gan_aug | tuned9 | recall | -0.001 | -0.004 | 0.002 | False |
| gan_aug | tuned9 | f1 | 0.001 | -0.002 | 0.003 | False |
| fed_single | tuned9 | map50 | 0.005 | 0.002 | 0.007 | True |
| fed_single | tuned9 | map50_95 | -0.004 | -0.007 | -0.001 | True |
| fed_single | tuned9 | map75 | -0.016 | -0.023 | -0.010 | True |
| fed_single | tuned9 | precision | -0.002 | -0.006 | 0.001 | False |
| fed_single | tuned9 | recall | 0.014 | 0.011 | 0.017 | True |
| fed_single | tuned9 | f1 | 0.007 | 0.005 | 0.009 | True |
| fed_iid | tuned9 | map50 | -0.037 | -0.041 | -0.034 | True |
| fed_iid | tuned9 | map50_95 | -0.039 | -0.042 | -0.036 | True |
| fed_iid | tuned9 | map75 | -0.060 | -0.067 | -0.053 | True |
| fed_iid | tuned9 | precision | -0.031 | -0.036 | -0.026 | True |
| fed_iid | tuned9 | recall | -0.039 | -0.043 | -0.036 | True |
| fed_iid | tuned9 | f1 | -0.036 | -0.039 | -0.033 | True |
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
| gan_aug | map50 | 0.850 | 0.741 | 0.921 |
| gan_aug | map50_95 | 0.460 | 0.380 | 0.520 |
| gan_aug | f1 | 0.833 | 0.746 | 0.891 |
| fed_single | map50 | 0.859 | 0.763 | 0.922 |
| fed_single | map50_95 | 0.464 | 0.391 | 0.516 |
| fed_single | f1 | 0.839 | 0.757 | 0.892 |
| fed_iid | map50 | 0.817 | 0.706 | 0.894 |
| fed_iid | map50_95 | 0.428 | 0.361 | 0.479 |
| fed_iid | f1 | 0.796 | 0.702 | 0.861 |
| fed_domain | map50 | 0.780 | 0.640 | 0.876 |
| fed_domain | map50_95 | 0.400 | 0.315 | 0.464 |
| fed_domain | f1 | 0.770 | 0.659 | 0.846 |

Diferencias pareadas (modelo - tuned9):

| a | b | metric | mean_diff | lo | hi | excluye_cero |
| --- | --- | --- | --- | --- | --- | --- |
| gan_aug | tuned9 | map50 | -0.006 | -0.020 | 0.002 | False |
| gan_aug | tuned9 | map50_95 | -0.009 | -0.020 | 0.000 | False |
| gan_aug | tuned9 | map75 | -0.019 | -0.038 | -0.005 | True |
| gan_aug | tuned9 | precision | 0.003 | -0.004 | 0.014 | False |
| gan_aug | tuned9 | recall | -0.002 | -0.017 | 0.008 | False |
| gan_aug | tuned9 | f1 | 0.000 | -0.008 | 0.005 | False |
| fed_single | tuned9 | map50 | 0.004 | -0.002 | 0.012 | False |
| fed_single | tuned9 | map50_95 | -0.004 | -0.022 | 0.011 | False |
| fed_single | tuned9 | map75 | -0.016 | -0.057 | 0.014 | False |
| fed_single | tuned9 | precision | -0.002 | -0.009 | 0.007 | False |
| fed_single | tuned9 | recall | 0.014 | 0.003 | 0.022 | True |
| fed_single | tuned9 | f1 | 0.007 | 0.002 | 0.012 | True |
| fed_iid | tuned9 | map50 | -0.038 | -0.056 | -0.023 | True |
| fed_iid | tuned9 | map50_95 | -0.040 | -0.061 | -0.022 | True |
| fed_iid | tuned9 | map75 | -0.060 | -0.099 | -0.029 | True |
| fed_iid | tuned9 | precision | -0.032 | -0.053 | -0.017 | True |
| fed_iid | tuned9 | recall | -0.040 | -0.056 | -0.022 | True |
| fed_iid | tuned9 | f1 | -0.037 | -0.052 | -0.023 | True |
| fed_domain | tuned9 | map50 | -0.077 | -0.124 | -0.040 | True |
| fed_domain | tuned9 | map50_95 | -0.069 | -0.113 | -0.034 | True |
| fed_domain | tuned9 | map75 | -0.098 | -0.170 | -0.041 | True |
| fed_domain | tuned9 | precision | -0.050 | -0.094 | -0.017 | True |
| fed_domain | tuned9 | recall | -0.074 | -0.106 | -0.043 | True |
| fed_domain | tuned9 | f1 | -0.064 | -0.099 | -0.035 | True |

Notas: solo validación. El remuestreo por imagen subestima la incertidumbre porque las imágenes de un dominio están correlacionadas; el de dominios es más conservador pero con pocos dominios es muy ancho. Un solo entrenamiento por modelo: no incluye la variabilidad entre semillas (~0.01 en mAP50).