"""Faster R-CNN ResNet-50 FPN de torchvision adaptado a una sola clase (espiga de trigo). Necesita PyTorch.

Dos variantes con pesos preentrenados en COCO (COCO_V1): 'v1' (fasterrcnn_resnet50_fpn) y 'v2'
(fasterrcnn_resnet50_fpn_v2). La documentación de torchvision avisa de que el módulo de detección está en Beta: se
fijan las versiones instaladas en el freeze del entorno.
"""
from __future__ import annotations

VARIANTS = ("v1", "v2")


def build_faster_rcnn(variant: str = "v1", pretrained: bool = True, num_classes: int = 2, image_size: int = 1024,
                      trainable_backbone_layers: int = 3, detections_per_img: int = 300,
                      score_thresh: float = 0.001, nms_thresh: float = 0.5):
    """`num_classes` incluye el fondo (espiga = 2). `image_size` fija min_size = max_size para que torchvision no
    reescale de nuevo imágenes que el Dataset ya entrega a ese tamaño. Se usan hasta 300 detecciones por imagen
    (hay imágenes con ~190 espigas) y un umbral de puntuación bajo para poder calcular mAP; el umbral operativo se
    elige después con validación."""
    from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights,
                                              fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if variant not in VARIANTS:
        raise ValueError(f"variant debe ser una de {VARIANTS}, no {variant!r}")
    builder, weights = ((fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights.COCO_V1) if variant == "v1"
                        else (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1))
    kw = dict(min_size=image_size, max_size=image_size, box_detections_per_img=detections_per_img,
              box_score_thresh=score_thresh, box_nms_thresh=nms_thresh)
    if pretrained:
        model = builder(weights=weights, trainable_backbone_layers=trainable_backbone_layers, **kw)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)     # cabeza nueva: fondo + espiga
    else:
        model = builder(weights=None, weights_backbone=None, num_classes=num_classes, **kw)
    return model


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable