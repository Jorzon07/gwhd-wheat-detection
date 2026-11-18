"""Construcción de detectores a partir de la configuración de entrenamiento (clave `arch`). Necesita PyTorch."""
from __future__ import annotations

from src.models.faster_rcnn import count_parameters

ARCHS = ("frcnn", "retinanet", "detr", "rtdetr")
__all__ = ["ARCHS", "build_detector", "count_parameters"]


def build_detector(tcfg: dict):
    """tcfg es el bloque `train` de la configuración. Sin `arch` se asume Faster R-CNN (ejecuciones anteriores)."""
    arch = tcfg.get("arch", "frcnn")
    size = int(tcfg["image_size"])
    common = dict(pretrained=bool(tcfg["pretrained"]), image_size=size,
                  trainable_backbone_layers=int(tcfg["trainable_backbone_layers"]),
                  detections_per_img=int(tcfg["detections_per_img"]), score_thresh=float(tcfg["score_thresh_eval"]),
                  nms_thresh=float(tcfg["nms_thresh"]))
    if arch == "frcnn":
        from src.models.faster_rcnn import build_faster_rcnn
        return build_faster_rcnn(tcfg["variant"], common["pretrained"], 2, size, common["trainable_backbone_layers"],
                                 common["detections_per_img"], common["score_thresh"], common["nms_thresh"])
    if arch == "retinanet":
        from src.models.retinanet import build_retinanet
        return build_retinanet(**common)
    if arch in ("detr", "rtdetr"):
        from src.models.hf_detectors import build_hf_detector
        return build_hf_detector(arch, common["pretrained"], detections_per_img=common["detections_per_img"],
                                 score_thresh=common["score_thresh"])
    raise ValueError(f"arch debe ser una de {ARCHS}, no {arch!r}")