# Evaluación final en prueba (991 imágenes, 17 dominios)

Umbral de confianza = mejor F1 en validación. Un solo entrenamiento por modelo.

| model | thr_val | map50 | map50_95 | map75 | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rtdetr_tuned | 0.529 | 0.792 | 0.382 | 0.313 | 0.854 | 0.733 | 0.789 |
| rtdetr_default | 0.536 | 0.774 | 0.368 | 0.298 | 0.843 | 0.716 | 0.774 |
| yolov8n | 0.326 | 0.765 | 0.364 | 0.293 | 0.838 | 0.685 | 0.754 |
| frcnn | 0.473 | 0.742 | 0.343 | 0.264 | 0.849 | 0.704 | 0.770 |
| retinanet | 0.425 | 0.686 | 0.320 | 0.250 | 0.839 | 0.656 | 0.736 |
| detr | 0.930 | 0.620 | 0.252 | 0.154 | 0.743 | 0.624 | 0.679 |
| fed_domain | 0.548 | 0.683 | 0.305 | 0.227 | 0.834 | 0.579 | 0.683 |
| fed_iid | 0.509 | 0.758 | 0.361 | 0.288 | 0.832 | 0.689 | 0.754 |
| fed_single | 0.490 | 0.792 | 0.377 | 0.307 | 0.846 | 0.743 | 0.791 |
| gan_aug | 0.504 | 0.799 | 0.386 | 0.318 | 0.855 | 0.740 | 0.793 |
| e3 | 0.322 | 0.806 | 0.385 | 0.308 | 0.848 | 0.743 | 0.792 |

## IC 95 % por remuestreo de imágenes

| model | metric | point | lo | hi |
| --- | --- | --- | --- | --- |
| rtdetr_tuned | map50 | 0.792 | 0.782 | 0.803 |
| rtdetr_tuned | map50_95 | 0.382 | 0.373 | 0.390 |
| rtdetr_tuned | f1 | 0.789 | 0.781 | 0.796 |
| rtdetr_default | map50 | 0.774 | 0.763 | 0.785 |
| rtdetr_default | map50_95 | 0.368 | 0.359 | 0.378 |
| rtdetr_default | f1 | 0.774 | 0.766 | 0.782 |
| yolov8n | map50 | 0.765 | 0.753 | 0.775 |
| yolov8n | map50_95 | 0.364 | 0.356 | 0.372 |
| yolov8n | f1 | 0.754 | 0.745 | 0.762 |
| frcnn | map50 | 0.742 | 0.731 | 0.753 |
| frcnn | map50_95 | 0.343 | 0.335 | 0.351 |
| frcnn | f1 | 0.770 | 0.762 | 0.778 |
| retinanet | map50 | 0.686 | 0.674 | 0.698 |
| retinanet | map50_95 | 0.320 | 0.312 | 0.328 |
| retinanet | f1 | 0.736 | 0.727 | 0.745 |
| detr | map50 | 0.620 | 0.604 | 0.636 |
| detr | map50_95 | 0.252 | 0.243 | 0.261 |
| detr | f1 | 0.679 | 0.667 | 0.691 |
| fed_domain | map50 | 0.683 | 0.668 | 0.698 |
| fed_domain | map50_95 | 0.305 | 0.295 | 0.315 |
| fed_domain | f1 | 0.683 | 0.671 | 0.695 |
| fed_iid | map50 | 0.758 | 0.746 | 0.769 |
| fed_iid | map50_95 | 0.361 | 0.352 | 0.370 |
| fed_iid | f1 | 0.754 | 0.745 | 0.763 |
| fed_single | map50 | 0.792 | 0.781 | 0.802 |
| fed_single | map50_95 | 0.377 | 0.368 | 0.387 |
| fed_single | f1 | 0.791 | 0.784 | 0.799 |
| gan_aug | map50 | 0.799 | 0.788 | 0.808 |
| gan_aug | map50_95 | 0.386 | 0.378 | 0.395 |
| gan_aug | f1 | 0.793 | 0.785 | 0.801 |
| e3 | map50 | 0.806 | 0.795 | 0.815 |
| e3 | map50_95 | 0.385 | 0.377 | 0.393 |
| e3 | f1 | 0.792 | 0.785 | 0.800 |

Diferencias pareadas (modelo - rtdetr_tuned):

| a | b | metric | mean_diff | lo | hi | excluye_cero |
| --- | --- | --- | --- | --- | --- | --- |
| rtdetr_default | rtdetr_tuned | map50 | -0.019 | -0.022 | -0.015 | True |
| rtdetr_default | rtdetr_tuned | map50_95 | -0.013 | -0.016 | -0.009 | True |
| rtdetr_default | rtdetr_tuned | map75 | -0.014 | -0.021 | -0.008 | True |
| rtdetr_default | rtdetr_tuned | precision | -0.011 | -0.016 | -0.007 | True |
| rtdetr_default | rtdetr_tuned | recall | -0.017 | -0.020 | -0.013 | True |
| rtdetr_default | rtdetr_tuned | f1 | -0.015 | -0.018 | -0.011 | True |
| yolov8n | rtdetr_tuned | map50 | -0.028 | -0.034 | -0.023 | True |
| yolov8n | rtdetr_tuned | map50_95 | -0.018 | -0.022 | -0.014 | True |
| yolov8n | rtdetr_tuned | map75 | -0.019 | -0.026 | -0.012 | True |
| yolov8n | rtdetr_tuned | precision | -0.016 | -0.022 | -0.011 | True |
| yolov8n | rtdetr_tuned | recall | -0.048 | -0.053 | -0.042 | True |
| yolov8n | rtdetr_tuned | f1 | -0.035 | -0.039 | -0.031 | True |
| frcnn | rtdetr_tuned | map50 | -0.051 | -0.058 | -0.045 | True |
| frcnn | rtdetr_tuned | map50_95 | -0.038 | -0.042 | -0.034 | True |
| frcnn | rtdetr_tuned | map75 | -0.048 | -0.056 | -0.040 | True |
| frcnn | rtdetr_tuned | precision | -0.005 | -0.010 | -0.000 | True |
| frcnn | rtdetr_tuned | recall | -0.028 | -0.033 | -0.023 | True |
| frcnn | rtdetr_tuned | f1 | -0.019 | -0.023 | -0.015 | True |
| retinanet | rtdetr_tuned | map50 | -0.106 | -0.114 | -0.098 | True |
| retinanet | rtdetr_tuned | map50_95 | -0.062 | -0.067 | -0.056 | True |
| retinanet | rtdetr_tuned | map75 | -0.061 | -0.070 | -0.053 | True |
| retinanet | rtdetr_tuned | precision | -0.016 | -0.021 | -0.010 | True |
| retinanet | rtdetr_tuned | recall | -0.076 | -0.083 | -0.070 | True |
| retinanet | rtdetr_tuned | f1 | -0.052 | -0.057 | -0.047 | True |
| detr | rtdetr_tuned | map50 | -0.172 | -0.183 | -0.161 | True |
| detr | rtdetr_tuned | map50_95 | -0.130 | -0.137 | -0.123 | True |
| detr | rtdetr_tuned | map75 | -0.159 | -0.170 | -0.147 | True |
| detr | rtdetr_tuned | precision | -0.111 | -0.119 | -0.101 | True |
| detr | rtdetr_tuned | recall | -0.108 | -0.117 | -0.099 | True |
| detr | rtdetr_tuned | f1 | -0.110 | -0.118 | -0.101 | True |
| fed_domain | rtdetr_tuned | map50 | -0.110 | -0.121 | -0.099 | True |
| fed_domain | rtdetr_tuned | map50_95 | -0.076 | -0.086 | -0.067 | True |
| fed_domain | rtdetr_tuned | map75 | -0.085 | -0.099 | -0.072 | True |
| fed_domain | rtdetr_tuned | precision | -0.020 | -0.029 | -0.011 | True |
| fed_domain | rtdetr_tuned | recall | -0.154 | -0.164 | -0.144 | True |
| fed_domain | rtdetr_tuned | f1 | -0.106 | -0.115 | -0.097 | True |
| fed_iid | rtdetr_tuned | map50 | -0.035 | -0.040 | -0.030 | True |
| fed_iid | rtdetr_tuned | map50_95 | -0.021 | -0.024 | -0.017 | True |
| fed_iid | rtdetr_tuned | map75 | -0.024 | -0.030 | -0.017 | True |
| fed_iid | rtdetr_tuned | precision | -0.022 | -0.028 | -0.016 | True |
| fed_iid | rtdetr_tuned | recall | -0.044 | -0.048 | -0.040 | True |
| fed_iid | rtdetr_tuned | f1 | -0.035 | -0.039 | -0.031 | True |
| fed_single | rtdetr_tuned | map50 | -0.001 | -0.004 | 0.002 | False |
| fed_single | rtdetr_tuned | map50_95 | -0.004 | -0.007 | -0.002 | True |
| fed_single | rtdetr_tuned | map75 | -0.006 | -0.010 | -0.001 | True |
| fed_single | rtdetr_tuned | precision | -0.008 | -0.012 | -0.004 | True |
| fed_single | rtdetr_tuned | recall | 0.010 | 0.007 | 0.013 | True |
| fed_single | rtdetr_tuned | f1 | 0.002 | -0.000 | 0.005 | False |
| gan_aug | rtdetr_tuned | map50 | 0.006 | 0.002 | 0.009 | True |
| gan_aug | rtdetr_tuned | map50_95 | 0.004 | 0.002 | 0.007 | True |
| gan_aug | rtdetr_tuned | map75 | 0.006 | 0.001 | 0.011 | True |
| gan_aug | rtdetr_tuned | precision | 0.001 | -0.003 | 0.005 | False |
| gan_aug | rtdetr_tuned | recall | 0.007 | 0.004 | 0.010 | True |
| gan_aug | rtdetr_tuned | f1 | 0.005 | 0.002 | 0.008 | True |
| e3 | rtdetr_tuned | map50 | 0.012 | 0.009 | 0.016 | True |
| e3 | rtdetr_tuned | map50_95 | 0.003 | 0.001 | 0.005 | True |
| e3 | rtdetr_tuned | map75 | -0.005 | -0.010 | 0.000 | False |
| e3 | rtdetr_tuned | precision | -0.006 | -0.010 | -0.002 | True |
| e3 | rtdetr_tuned | recall | 0.011 | 0.008 | 0.014 | True |
| e3 | rtdetr_tuned | f1 | 0.004 | 0.001 | 0.006 | True |

## IC 95 % por remuestreo de dominios completos (17)

| model | metric | point | lo | hi |
| --- | --- | --- | --- | --- |
| rtdetr_tuned | map50 | 0.792 | 0.740 | 0.846 |
| rtdetr_tuned | map50_95 | 0.382 | 0.338 | 0.426 |
| rtdetr_tuned | f1 | 0.789 | 0.744 | 0.828 |
| rtdetr_default | map50 | 0.774 | 0.713 | 0.835 |
| rtdetr_default | map50_95 | 0.368 | 0.324 | 0.414 |
| rtdetr_default | f1 | 0.774 | 0.724 | 0.818 |
| yolov8n | map50 | 0.765 | 0.701 | 0.817 |
| yolov8n | map50_95 | 0.364 | 0.322 | 0.399 |
| yolov8n | f1 | 0.754 | 0.699 | 0.796 |
| frcnn | map50 | 0.742 | 0.673 | 0.819 |
| frcnn | map50_95 | 0.343 | 0.301 | 0.390 |
| frcnn | f1 | 0.770 | 0.718 | 0.818 |
| retinanet | map50 | 0.686 | 0.612 | 0.781 |
| retinanet | map50_95 | 0.320 | 0.276 | 0.372 |
| retinanet | f1 | 0.736 | 0.679 | 0.791 |
| detr | map50 | 0.620 | 0.525 | 0.740 |
| detr | map50_95 | 0.252 | 0.199 | 0.320 |
| detr | f1 | 0.679 | 0.610 | 0.759 |
| fed_domain | map50 | 0.683 | 0.580 | 0.794 |
| fed_domain | map50_95 | 0.305 | 0.241 | 0.382 |
| fed_domain | f1 | 0.683 | 0.597 | 0.776 |
| fed_iid | map50 | 0.758 | 0.694 | 0.822 |
| fed_iid | map50_95 | 0.361 | 0.315 | 0.406 |
| fed_iid | f1 | 0.754 | 0.698 | 0.802 |
| fed_single | map50 | 0.792 | 0.739 | 0.844 |
| fed_single | map50_95 | 0.377 | 0.333 | 0.424 |
| fed_single | f1 | 0.791 | 0.749 | 0.829 |
| gan_aug | map50 | 0.799 | 0.743 | 0.854 |
| gan_aug | map50_95 | 0.386 | 0.341 | 0.432 |
| gan_aug | f1 | 0.793 | 0.747 | 0.835 |
| e3 | map50 | 0.806 | 0.755 | 0.849 |
| e3 | map50_95 | 0.385 | 0.347 | 0.422 |
| e3 | f1 | 0.792 | 0.746 | 0.831 |

Diferencias pareadas (modelo - rtdetr_tuned):

| a | b | metric | mean_diff | lo | hi | excluye_cero |
| --- | --- | --- | --- | --- | --- | --- |
| rtdetr_default | rtdetr_tuned | map50 | -0.018 | -0.030 | -0.004 | True |
| rtdetr_default | rtdetr_tuned | map50_95 | -0.013 | -0.027 | 0.005 | False |
| rtdetr_default | rtdetr_tuned | map75 | -0.014 | -0.040 | 0.017 | False |
| rtdetr_default | rtdetr_tuned | precision | -0.011 | -0.019 | -0.003 | True |
| rtdetr_default | rtdetr_tuned | recall | -0.016 | -0.027 | -0.002 | True |
| rtdetr_default | rtdetr_tuned | f1 | -0.014 | -0.024 | -0.003 | True |
| yolov8n | rtdetr_tuned | map50 | -0.030 | -0.054 | -0.013 | True |
| yolov8n | rtdetr_tuned | map50_95 | -0.018 | -0.037 | -0.003 | True |
| yolov8n | rtdetr_tuned | map75 | -0.020 | -0.050 | 0.007 | False |
| yolov8n | rtdetr_tuned | precision | -0.016 | -0.025 | -0.003 | True |
| yolov8n | rtdetr_tuned | recall | -0.050 | -0.080 | -0.027 | True |
| yolov8n | rtdetr_tuned | f1 | -0.036 | -0.054 | -0.025 | True |
| frcnn | rtdetr_tuned | map50 | -0.051 | -0.077 | -0.020 | True |
| frcnn | rtdetr_tuned | map50_95 | -0.038 | -0.054 | -0.016 | True |
| frcnn | rtdetr_tuned | map75 | -0.048 | -0.075 | -0.016 | True |
| frcnn | rtdetr_tuned | precision | -0.005 | -0.015 | 0.005 | False |
| frcnn | rtdetr_tuned | recall | -0.028 | -0.049 | -0.005 | True |
| frcnn | rtdetr_tuned | f1 | -0.019 | -0.032 | -0.006 | True |
| retinanet | rtdetr_tuned | map50 | -0.105 | -0.141 | -0.059 | True |
| retinanet | rtdetr_tuned | map50_95 | -0.061 | -0.082 | -0.034 | True |
| retinanet | rtdetr_tuned | map75 | -0.061 | -0.089 | -0.027 | True |
| retinanet | rtdetr_tuned | precision | -0.016 | -0.029 | -0.002 | True |
| retinanet | rtdetr_tuned | recall | -0.075 | -0.106 | -0.037 | True |
| retinanet | rtdetr_tuned | f1 | -0.052 | -0.071 | -0.032 | True |
| detr | rtdetr_tuned | map50 | -0.170 | -0.234 | -0.093 | True |
| detr | rtdetr_tuned | map50_95 | -0.129 | -0.173 | -0.079 | True |
| detr | rtdetr_tuned | map75 | -0.158 | -0.219 | -0.099 | True |
| detr | rtdetr_tuned | precision | -0.109 | -0.152 | -0.062 | True |
| detr | rtdetr_tuned | recall | -0.105 | -0.152 | -0.049 | True |
| detr | rtdetr_tuned | f1 | -0.108 | -0.153 | -0.054 | True |
| fed_domain | rtdetr_tuned | map50 | -0.108 | -0.170 | -0.036 | True |
| fed_domain | rtdetr_tuned | map50_95 | -0.075 | -0.129 | -0.014 | True |
| fed_domain | rtdetr_tuned | map75 | -0.084 | -0.160 | -0.001 | True |
| fed_domain | rtdetr_tuned | precision | -0.020 | -0.055 | 0.018 | False |
| fed_domain | rtdetr_tuned | recall | -0.152 | -0.207 | -0.081 | True |
| fed_domain | rtdetr_tuned | f1 | -0.105 | -0.157 | -0.043 | True |
| fed_iid | rtdetr_tuned | map50 | -0.035 | -0.049 | -0.019 | True |
| fed_iid | rtdetr_tuned | map50_95 | -0.021 | -0.035 | -0.006 | True |
| fed_iid | rtdetr_tuned | map75 | -0.024 | -0.049 | 0.002 | False |
| fed_iid | rtdetr_tuned | precision | -0.022 | -0.030 | -0.013 | True |
| fed_iid | rtdetr_tuned | recall | -0.043 | -0.063 | -0.025 | True |
| fed_iid | rtdetr_tuned | f1 | -0.035 | -0.048 | -0.023 | True |
| fed_single | rtdetr_tuned | map50 | -0.001 | -0.007 | 0.005 | False |
| fed_single | rtdetr_tuned | map50_95 | -0.004 | -0.012 | 0.005 | False |
| fed_single | rtdetr_tuned | map75 | -0.005 | -0.018 | 0.008 | False |
| fed_single | rtdetr_tuned | precision | -0.008 | -0.014 | -0.001 | True |
| fed_single | rtdetr_tuned | recall | 0.010 | 0.004 | 0.016 | True |
| fed_single | rtdetr_tuned | f1 | 0.003 | -0.002 | 0.007 | False |
| gan_aug | rtdetr_tuned | map50 | 0.006 | -0.001 | 0.015 | False |
| gan_aug | rtdetr_tuned | map50_95 | 0.005 | -0.003 | 0.013 | False |
| gan_aug | rtdetr_tuned | map75 | 0.006 | -0.008 | 0.020 | False |
| gan_aug | rtdetr_tuned | precision | 0.002 | -0.007 | 0.014 | False |
| gan_aug | rtdetr_tuned | recall | 0.007 | 0.000 | 0.014 | True |
| gan_aug | rtdetr_tuned | f1 | 0.005 | -0.001 | 0.012 | False |
| e3 | rtdetr_tuned | map50 | 0.012 | 0.001 | 0.021 | True |
| e3 | rtdetr_tuned | map50_95 | 0.003 | -0.007 | 0.013 | False |
| e3 | rtdetr_tuned | map75 | -0.005 | -0.020 | 0.011 | False |
| e3 | rtdetr_tuned | precision | -0.006 | -0.013 | 0.000 | False |
| e3 | rtdetr_tuned | recall | 0.011 | 0.004 | 0.017 | True |
| e3 | rtdetr_tuned | f1 | 0.003 | -0.000 | 0.008 | False |

## mAP50-95 por dominio de prueba

| domain | n_images | rtdetr_tuned | rtdetr_default | yolov8n | frcnn | retinanet | detr | fed_domain | fed_iid | fed_single | gan_aug | e3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARC_1 | 30 | 0.373 | 0.383 | 0.329 | 0.370 | 0.355 | 0.292 | 0.331 | 0.364 | 0.389 | 0.403 | 0.401 |
| Arvalis_10 | 58 | 0.628 | 0.622 | 0.569 | 0.641 | 0.623 | 0.563 | 0.630 | 0.595 | 0.647 | 0.648 | 0.616 |
| CIMMYT_1 | 68 | 0.400 | 0.356 | 0.352 | 0.303 | 0.310 | 0.257 | 0.263 | 0.356 | 0.412 | 0.375 | 0.381 |
| CIMMYT_2 | 77 | 0.514 | 0.490 | 0.442 | 0.440 | 0.418 | 0.275 | 0.343 | 0.465 | 0.527 | 0.507 | 0.494 |
| CIMMYT_3 | 59 | 0.401 | 0.377 | 0.376 | 0.366 | 0.353 | 0.274 | 0.323 | 0.362 | 0.410 | 0.388 | 0.396 |
| ULiège-GxABT_1 | 30 | 0.423 | 0.418 | 0.340 | 0.336 | 0.310 | 0.299 | 0.361 | 0.386 | 0.445 | 0.424 | 0.400 |
| UQ_1 | 22 | 0.241 | 0.233 | 0.225 | 0.223 | 0.201 | 0.144 | 0.229 | 0.231 | 0.247 | 0.264 | 0.264 |
| UQ_10 | 106 | 0.394 | 0.375 | 0.390 | 0.334 | 0.289 | 0.175 | 0.215 | 0.360 | 0.398 | 0.392 | 0.406 |
| UQ_11 | 84 | 0.268 | 0.250 | 0.256 | 0.223 | 0.195 | 0.122 | 0.108 | 0.233 | 0.271 | 0.266 | 0.278 |
| UQ_2 | 16 | 0.358 | 0.341 | 0.344 | 0.326 | 0.319 | 0.265 | 0.246 | 0.293 | 0.372 | 0.349 | 0.365 |
| UQ_3 | 14 | 0.361 | 0.325 | 0.343 | 0.299 | 0.262 | 0.216 | 0.277 | 0.323 | 0.378 | 0.338 | 0.380 |
| UQ_4 | 30 | 0.438 | 0.401 | 0.399 | 0.380 | 0.358 | 0.300 | 0.379 | 0.396 | 0.440 | 0.405 | 0.440 |
| UQ_5 | 30 | 0.317 | 0.268 | 0.312 | 0.269 | 0.233 | 0.151 | 0.283 | 0.282 | 0.290 | 0.332 | 0.344 |
| UQ_6 | 30 | 0.311 | 0.307 | 0.302 | 0.355 | 0.331 | 0.300 | 0.405 | 0.314 | 0.279 | 0.349 | 0.341 |
| UQ_7 | 17 | 0.409 | 0.394 | 0.406 | 0.372 | 0.329 | 0.229 | 0.235 | 0.385 | 0.397 | 0.430 | 0.425 |
| Usask_1 | 200 | 0.433 | 0.430 | 0.430 | 0.407 | 0.404 | 0.379 | 0.431 | 0.436 | 0.417 | 0.449 | 0.417 |
| Utokyo_3 | 120 | 0.312 | 0.361 | 0.342 | 0.323 | 0.304 | 0.303 | 0.397 | 0.336 | 0.295 | 0.328 | 0.325 |
