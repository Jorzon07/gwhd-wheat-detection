"""Aplica (de forma idempotente) dos correcciones pequeñas. Ejecutar desde la raíz del proyecto."""
from pathlib import Path
import os

EDITS = [
    ("tests/test_hf_wrapper_torch.py",
     "return SimpleNamespace(loss=loss, logits=self.logits.expand(n, -1, -1), pred_boxes=self.boxes.expand(n, -1, -1))",
     "expand = lambda t: None if t is None else t.expand(n, -1, -1)   # noqa: E731\n"
     "        return SimpleNamespace(loss=loss, logits=expand(self.logits), pred_boxes=expand(self.boxes))"),
    ("src/evaluation/analysis.py",
     "ax[1, 0].legend(fontsize=7)",
     "if ax[1, 0].get_legend_handles_labels()[0]:\n            ax[1, 0].legend(fontsize=7)"),
]

for rel, old, new in EDITS:
    p = Path(rel)
    if not p.exists():
        print(f"[!] No existe {rel}"); continue
    s = p.read_text(encoding="utf-8")
    if new in s:
        print(f"[=] {rel}: ya aplicado"); continue
    if s.count(old) != 1:
        print(f"[!] {rel}: el texto a reemplazar aparece {s.count(old)} veces; no se toca"); continue
    # conserva la sangría original de la línea reemplazada
    line = next(l for l in s.splitlines() if old in l)
    indent = line[: len(line) - len(line.lstrip())]
    new_i = new.replace("\n        ", "\n" + indent).replace("\n            ", "\n" + indent + "    ")
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(s.replace(old, new_i), encoding="utf-8"); os.replace(tmp, p)
    print(f"[+] {rel}: corregido")
