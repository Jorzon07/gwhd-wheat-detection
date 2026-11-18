"""DETR y RT-DETR de Hugging Face Transformers con la misma interfaz que los detectores de torchvision. Necesita PyTorch.

Interfaz: model(images, targets) en entrenamiento -> dict de pérdidas; model(images) en evaluación -> lista de dicts con
boxes (xyxy, en píxeles de la entrada), scores y labels (siempre 1 = espiga, como en torchvision).

Detalles verificados en el código fuente de Transformers 5.22.2:
* Las etiquetas son dicts con `class_labels` (índices desde 0, sin clase de fondo) y `boxes` en (cx, cy, w, h) normalizado.
* DETR tiene 100 consultas por defecto (insuficientes con hasta ~190 espigas por imagen): aquí se suben a 300.
  RT-DETR ya trae 300. DETR normaliza con media/desviación de ImageNet; RT-DETR espera la imagen en 0-1 sin normalizar.
* DETR usa softmax con una clase extra "sin objeto" (la última); RT-DETR usa sigmoide con tantos logits como clases.
"""
from __future__ import annotations

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CHECKPOINTS = {"detr": "facebook/detr-resnet-50", "rtdetr": "PekingU/rtdetr_r18vd"}      # ambos preentrenados en COCO


def xyxy_to_cxcywh_norm(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    b = boxes.float()
    cx = (b[:, 0] + b[:, 2]) / 2 / width
    cy = (b[:, 1] + b[:, 3]) / 2 / height
    w = (b[:, 2] - b[:, 0]) / width
    h = (b[:, 3] - b[:, 1]) / height
    return torch.stack([cx, cy, w, h], dim=1)


def cxcywh_norm_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height],
                       dim=-1)


class HFDetectionWrapper(torch.nn.Module):
    def __init__(self, hf_model, kind: str, normalize: bool, max_dets: int = 300, score_thresh: float = 0.001):
        super().__init__()
        if kind not in ("detr", "rtdetr"):
            raise ValueError(f"kind debe ser 'detr' o 'rtdetr', no {kind!r}")
        self.hf, self.kind, self.normalize = hf_model, kind, bool(normalize)
        self.max_dets, self.score_thresh = int(max_dets), float(score_thresh)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images, targets=None):
        x = torch.stack(list(images))                                  # todas las imágenes tienen el mismo tamaño
        _, _, height, width = x.shape
        if self.normalize:
            x = (x - self.mean) / self.std
        if self.training:
            if targets is None:
                raise ValueError("En entrenamiento hacen falta los objetivos (targets)")
            labels = [{"class_labels": (t["labels"] - 1).long(),          # torchvision usa 1 = espiga; HF empieza en 0
                       "boxes": xyxy_to_cxcywh_norm(t["boxes"], width, height)} for t in targets]
            out = self.hf(pixel_values=x, labels=labels)
            return {"hf_loss": out.loss}                                  # pérdida total ponderada (emparejamiento húngaro)
        out = self.hf(pixel_values=x)
        logits, boxes = out.logits.float(), out.pred_boxes.float()
        scores = logits.softmax(-1)[..., 0] if self.kind == "detr" else logits.sigmoid()[..., 0]
        xyxy = cxcywh_norm_to_xyxy(boxes, width, height)
        xyxy = torch.stack([xyxy[..., 0].clamp(0, width), xyxy[..., 1].clamp(0, height),
                            xyxy[..., 2].clamp(0, width), xyxy[..., 3].clamp(0, height)], dim=-1)
        results = []
        for b in range(x.shape[0]):
            s, bx = scores[b], xyxy[b]
            keep = s >= self.score_thresh
            s, bx = s[keep], bx[keep]
            if len(s) > self.max_dets:
                top = s.topk(self.max_dets).indices
                s, bx = s[top], bx[top]
            results.append({"boxes": bx, "scores": s, "labels": torch.ones(len(s), dtype=torch.int64, device=s.device)})
        return results

    def param_groups(self, backbone_lr_mult: float = 0.1):
        """Backbone con learning rate reducido (receta habitual de DETR). `lr_scale` lo aplica el motor de entrenamiento."""
        backbone, rest = [], []
        for name, p in self.hf.named_parameters():
            if p.requires_grad:
                (backbone if "backbone" in name else rest).append(p)
        groups = [{"params": rest, "lr_scale": 1.0}]
        if backbone:
            groups.append({"params": backbone, "lr_scale": float(backbone_lr_mult)})
        return groups


def _expand_queries(hf, n: int, noise: float = 0.1) -> None:
    """Amplía las consultas de objeto de DETR a `n` repitiendo las preentrenadas (+ ruido para romper la simetría).
    Crear consultas nuevas al azar descarta lo que el decodificador preentrenado aprendió a hacer con ellas."""
    old = hf.model.query_position_embeddings.weight.data
    if old.shape[0] >= n:
        return
    reps = -(-n // old.shape[0])
    new = old.repeat(reps, 1)[:n].clone()
    new += noise * old.std() * torch.randn_like(new)
    emb = torch.nn.Embedding(n, old.shape[1]).to(device=old.device, dtype=old.dtype)
    emb.weight.data.copy_(new)
    hf.model.query_position_embeddings = emb
    hf.config.num_queries = n


def build_hf_detector(kind: str, pretrained: bool = True, num_queries: int = 300, detections_per_img: int = 300,
                      score_thresh: float = 0.001):
    """Una sola clase (espiga). Las cabezas de clasificación y, en DETR, las consultas se reinicializan
    (`ignore_mismatched_sizes`) porque cambian de tamaño; el resto de pesos viene de COCO."""
    if not pretrained:
        raise ValueError("Los modelos de Hugging Face solo se construyen con pesos preentrenados")
    if kind == "detr":
        from transformers import DetrForObjectDetection                    # requiere también timm (backbone por defecto)
        hf = DetrForObjectDetection.from_pretrained(
            CHECKPOINTS["detr"], num_labels=1, ignore_mismatched_sizes=True, auxiliary_loss=True)
        _expand_queries(hf, num_queries)                                   # parte de las 100 consultas preentrenadas
        return HFDetectionWrapper(hf, "detr", normalize=True, max_dets=detections_per_img, score_thresh=score_thresh)
    if kind == "rtdetr":
        from transformers import RTDetrForObjectDetection
        hf = RTDetrForObjectDetection.from_pretrained(CHECKPOINTS["rtdetr"], num_labels=1, ignore_mismatched_sizes=True)
        return HFDetectionWrapper(hf, "rtdetr", normalize=False, max_dets=detections_per_img, score_thresh=score_thresh)
    raise ValueError(f"kind debe ser 'detr' o 'rtdetr', no {kind!r}")