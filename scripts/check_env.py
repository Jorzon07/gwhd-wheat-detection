"""Diagnóstico del entorno: no instala ni modifica nada."""
import os
import platform
import shutil
import subprocess
import sys

print("Python      :", sys.version.replace("\n", " "))
print("Plataforma  :", platform.platform())
print("CPUs lógicas:", os.cpu_count())

try:
    import psutil
    print(f"RAM total   : {psutil.virtual_memory().total / 1e9:.1f} GB")
except ImportError:
    print("RAM total   : (psutil no instalado, se omite)")

print(f"Disco libre : {shutil.disk_usage('.').free / 1e9:.1f} GB (unidad actual)")

try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
        capture_output=True, text=True, timeout=15,
    )
    print("nvidia-smi  :\n" + (out.stdout or out.stderr).strip())
except Exception as e:  # noqa: BLE001
    print("nvidia-smi  : no disponible ->", e)

try:
    import torch
    print("torch       :", torch.__version__, "| CUDA runtime:", torch.version.cuda,
          "| cuda disponible:", torch.cuda.is_available())
except ImportError:
    print("torch       : no instalado (normal en la Fase 1)")