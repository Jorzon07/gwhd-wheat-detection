"""Muestra el contenido de un .npz de predicciones (para adaptar el cargador del bootstrap)."""
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
for k in z.files:
    a = z[k]; print(f"{k:25s} dtype={a.dtype} shape={a.shape}", "| primeros:", np.array2string(a.ravel()[:3], threshold=6) if a.size else "")
