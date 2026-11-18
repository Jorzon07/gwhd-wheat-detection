"""RetinaNet ResNet-50 FPN de torchvision adaptado a una sola clase (espiga de trigo). Necesita PyTorch.

Pesos preentrenados en COCO (COCO_V1). La documentación de torchvision avisa de que el módulo de detección está en
Beta: se fijan las versiones instaladas en el freeze del entorno.
"""
from __future__ import annotations

import math


def replace_cls_logits(model, num_classes: int) -> None:
    """Sustituye solo la última capa de clasificación (cls_logits) y conserva las 4 convoluciones preentrenadas de la
    cabeza. La inicialización sigue la de RetinaNet: pesos N(0, 0.01) y sesgo con el prior de la focal loss (p = 0.01),
    que evita que al principio la pérdida la dominen los negativos."""
    import torch
    head = model.head.classification_head
    old = head.cls_logits
    num_anchors = getattr(head, "num_anchors", old.out_channels // head.num_classes)
    cls = torch.nn.Conv2d(old.in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(cls.weight, std=0.01)
    torch.nn.init.constant_(cls.bias, -math.log((1 - 0.01) / 0.01))
    head.cls_logits = cls
    head.num_classes = num_classes


def build_retinanet(pretrained: bool = True, num_classes: int = 2, image_size: int = 800,
                    trainable_backbone_layers: int = 3, detections_per_img: int = 300,
                    score_thresh: float = 0.001, nms_thresh: float = 0.5):
    """`num_classes` incluye el fondo (espiga = 2), igual que en Faster R-CNN, de modo que las etiquetas del Dataset
    (siempre 1) sirven para ambos. `image_size` fija min_size = max_size para que torchvision no reescale."""
    from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn
    kw = dict(min_size=image_size, max_size=image_size, detections_per_img=detections_per_img,
              score_thresh=score_thresh, nms_thresh=nms_thresh)
    if pretrained:
        model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1,
                                       trainable_backbone_layers=trainable_backbone_layers, **kw)
        replace_cls_logits(model, num_classes)
    else:
        model = retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes, **kw)
    return model