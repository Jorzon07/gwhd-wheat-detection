"""Parche de la Fase 4B: añade DETR y RT-DETR (Hugging Face) a train.py, probe.py, el motor y la configuración.

Ejecutar UNA vez desde la raíz del proyecto: python scripts/apply_phase4b.py
Comprueba cada fragmento antes de escribir; si algún archivo no coincide con lo esperado, no modifica NADA.
Es idempotente. También protege a scripts/apply_phase4a.py para que no se vuelva a ejecutar por error.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH_CHOICES = '["frcnn", "retinanet", "detr", "rtdetr"]'

EDITS = {
    "src/training/engine.py": [
        ('                g["lr"] = lr\n',
         '                g["lr"] = lr * g.get("lr_scale", 1.0)          # DETR/RT-DETR: backbone con lr reducido\n'),
    ],
    "src/training/train.py": [
        ('               "seed", "train_images", "smoke"]',
         '               "seed", "train_images", "smoke", "grad_clip", "backbone_lr_mult"]'),
        ('    ap.add_argument("--arch", choices=["frcnn", "retinanet"], default=None, help="arquitectura (por defecto, la de la configuración)")\n',
         f'    ap.add_argument("--arch", choices={ARCH_CHOICES}, default=None, help="arquitectura (por defecto, la de la configuración)")\n'),
        ('    ap.add_argument("--lr", type=float, default=None)\n',
         '    ap.add_argument("--lr", type=float, default=None)\n'
         '    ap.add_argument("--optimizer", choices=["sgd", "adamw"], default=None)\n'
         '    ap.add_argument("--grad-clip", type=float, default=None, help="norma máxima del gradiente")\n'
         '    ap.add_argument("--backbone-lr-mult", type=float, default=None, help="solo DETR/RT-DETR: factor del lr del backbone")\n'),
        ('                     ("train_images", args.train_images)):\n',
         '                     ("train_images", args.train_images), ("optimizer", args.optimizer),\n'
         '                     ("grad_clip", args.grad_clip), ("backbone_lr_mult", args.backbone_lr_mult)):\n'),
        ('    params = [p for p in model.parameters() if p.requires_grad]\n    if tcfg["optimizer"] == "sgd":\n',
         '    params = [p for p in model.parameters() if p.requires_grad]\n'
         '    if hasattr(model, "param_groups"):                      # DETR/RT-DETR: backbone con learning rate reducido\n'
         '        params = model.param_groups(float(tcfg.get("backbone_lr_mult", 0.1)))\n'
         '    if tcfg["optimizer"] == "sgd":\n'),
    ],
    "src/training/probe.py": [
        ('    ap.add_argument("--arch", choices=["frcnn", "retinanet"], default=None, help="arquitectura a medir")\n',
         f'    ap.add_argument("--arch", choices={ARCH_CHOICES}, default=None, help="arquitectura a medir")\n'),
    ],
    "configs/data.yaml": [
        ("  grad_clip: 10.0\n",
         "  grad_clip: 10.0\n  backbone_lr_mult: 0.1            # solo DETR/RT-DETR: lr del backbone = lr * este factor\n"),
    ],
    # el parche 4A no debe volver a ejecutarse sobre archivos ya ampliados por este
    "scripts/apply_phase4a.py": [
        ("def main() -> int:\n    new_texts, already = {}, 0\n",
         "def main() -> int:\n"
         "    if '\"detr\", \"rtdetr\"' in (ROOT / \"src/training/train.py\").read_text(encoding=\"utf-8\"):\n"
         "        print(\"La Fase 4A ya está aplicada (y ampliada por la 4B): no se ejecuta de nuevo.\")\n"
         "        return 0\n"
         "    new_texts, already = {}, 0\n"),
    ],
}


def main() -> int:
    new_texts, already = {}, 0
    for rel, edits in EDITS.items():
        text = (ROOT / rel).read_text(encoding="utf-8")
        for old, new in edits:
            if new in text:          # el texto nuevo ya está (puede contener al viejo): aplicado, no se toca
                already += 1
                continue
            if text.count(old) != 1:
                print(f"ERROR: en {rel} no se encontró exactamente una vez este fragmento:\n---\n{old}\n---\n"
                      "No se ha modificado ningún archivo. Pégame este mensaje.")
                return 1
            text = text.replace(old, new)
        new_texts[rel] = text
    if already == sum(len(e) for e in EDITS.values()):
        print("El parche de la Fase 4B ya estaba aplicado.")
        return 0
    for rel, text in new_texts.items():
        (ROOT / rel).write_text(text, encoding="utf-8")
    print("Parche de la Fase 4B aplicado: engine.py, train.py, probe.py, configs/data.yaml y apply_phase4a.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())