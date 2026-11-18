"""Parche idempotente: run_cv.py compara rutas resueltas (absoluta vs relativa)."""
from pathlib import Path
p = Path(__file__).with_name("run_cv.py")
s = p.read_text(encoding="utf-8")
old = 'if Path(used).as_posix().rstrip("/") != Path(r["dir"]).as_posix().rstrip("/"):'
new = 'if Path(used).resolve() != Path(r["dir"]).resolve():'
if new in s:
    print("ya aplicado")
elif old in s:
    p.write_text(s.replace(old, new), encoding="utf-8"); print("aplicado")
else:
    raise SystemExit("no encontré la línea a parchear")
