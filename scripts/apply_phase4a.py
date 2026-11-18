"""Parche de la Fase 4A: integra RetinaNet (clave `arch`) en train.py, probe.py, la configuración y las curvas.

Ejecutar UNA vez desde la raíz del proyecto: python scripts/apply_phase4a.py
Comprueba cada fragmento antes de escribir; si algún archivo no coincide con lo esperado, no modifica NADA.
Es idempotente: si ya se aplicó, lo dice y termina.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EDITS = {
    "src/training/train.py": [
        ("from src.models.faster_rcnn import build_faster_rcnn, count_parameters",
         "from src.models.factory import build_detector, count_parameters"),
        ('RESUME_KEYS = ["variant",', 'RESUME_KEYS = ["arch", "variant",'),
        ('    ap.add_argument("--variant", choices=["v1", "v2"], default=None)\n',
         '    ap.add_argument("--variant", choices=["v1", "v2"], default=None)\n'
         '    ap.add_argument("--arch", choices=["frcnn", "retinanet"], default=None, help="arquitectura (por defecto, la de la configuración)")\n'),
        ('("image_size", args.image_size), ("variant", args.variant), ("lr", args.lr),',
         '("image_size", args.image_size), ("variant", args.variant), ("arch", args.arch), ("lr", args.lr),'),
        ('''    model = build_faster_rcnn(tcfg["variant"], bool(tcfg["pretrained"]), 2, size, int(tcfg["trainable_backbone_layers"]),
                              int(tcfg["detections_per_img"]), float(tcfg["score_thresh_eval"]),
                              float(tcfg["nms_thresh"])).to(device)''',
         '    model = build_detector(tcfg).to(device)'),
        ("Modelo {tcfg['variant']} |", "Modelo {tcfg.get('arch', 'frcnn')}/{tcfg['variant']} |"),
    ],
    "src/training/probe.py": [
        ("from src.models.faster_rcnn import build_faster_rcnn, count_parameters",
         "from src.models.factory import build_detector, count_parameters"),
        ('''        model = build_faster_rcnn(variant, True, 2, size, int(tcfg["trainable_backbone_layers"]),
                                  int(tcfg["detections_per_img"]), float(tcfg["score_thresh_eval"]),
                                  float(tcfg["nms_thresh"])).to(device).train()''',
         '        model = build_detector({**tcfg, "variant": variant, "image_size": size, "pretrained": True}).to(device).train()'),
        ('            m = build_faster_rcnn(variant, True, 2, 1024, int(tcfg["trainable_backbone_layers"]))',
         '            m = build_detector({**tcfg, "variant": variant, "image_size": 1024, "pretrained": True})'),
        ('    ap.add_argument("--variants", nargs="+", default=["v1", "v2"], choices=["v1", "v2"])\n',
         '    ap.add_argument("--variants", nargs="+", default=["v1", "v2"], choices=["v1", "v2"])\n'
         '    ap.add_argument("--arch", choices=["frcnn", "retinanet"], default=None, help="arquitectura a medir")\n'),
        ('    tcfg = cfg["train"]\n',
         '    tcfg = dict(cfg["train"])\n'
         '    if args.arch:\n        tcfg["arch"] = args.arch\n'
         '    variants = args.variants if tcfg.get("arch", "frcnn") == "frcnn" else ["v1"]   # RetinaNet: solo v1\n'),
        ("    for variant in args.variants:", "    for variant in variants:"),
    ],
    "configs/data.yaml": [
        ("\ntrain:\n  seed: 0\n",
         "\ntrain:\n  seed: 0\n  arch: frcnn                      # frcnn | retinanet (más arquitecturas en la Fase 4)\n"),
    ],
    "src/evaluation/analysis.py": [
        ('("train_loss_rpn_box_reg", "regresión RPN")]:',
         '("train_loss_rpn_box_reg", "regresión RPN"),\n'
         '                     ("train_classification", "clasificación (focal)"),\n'
         '                     ("train_bbox_regression", "regresión de cajas (RetinaNet)")]:'),
    ],
}

TEST_APPEND = '''

def test_curves_figure_handles_retinanet_loss_columns(tmp_path):
    from src.evaluation.analysis import curves_figure
    df = pd.DataFrame({"epoch": [0, 1], "train_loss": [1.0, 0.8], "train_classification": [0.6, 0.5],
                       "train_bbox_regression": [0.4, 0.3], "val_map50": [0.5, 0.6], "val_map50_95": [0.2, 0.3],
                       "lr_end": [0.002, 0.001], "train_time_s": [500, 505]})
    out = tmp_path / "c.png"
    curves_figure(df, out)
    assert out.exists() and out.stat().st_size > 1000
'''


def main() -> int:
    if '"detr", "rtdetr"' in (ROOT / "src/training/train.py").read_text(encoding="utf-8"):
        print("La Fase 4A ya está aplicada (y ampliada por la 4B): no se ejecuta de nuevo.")
        return 0
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
    tpath = ROOT / "tests/test_analysis.py"
    ttext = tpath.read_text(encoding="utf-8")
    add_test = "test_curves_figure_handles_retinanet_loss_columns" not in ttext
    if already == sum(len(e) for e in EDITS.values()) and not add_test:
        print("El parche de la Fase 4A ya estaba aplicado.")
        return 0
    for rel, text in new_texts.items():
        (ROOT / rel).write_text(text, encoding="utf-8")
    if add_test:
        tpath.write_text(ttext + TEST_APPEND, encoding="utf-8")
    print("Parche de la Fase 4A aplicado: train.py, probe.py, configs/data.yaml, analysis.py y tests/test_analysis.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())