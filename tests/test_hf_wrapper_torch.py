import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.models.factory import build_detector  # noqa: E402
from src.models.hf_detectors import (HFDetectionWrapper, IMAGENET_MEAN, IMAGENET_STD,  # noqa: E402
                                     cxcywh_norm_to_xyxy, xyxy_to_cxcywh_norm)


class FakeHF(torch.nn.Module):
    """Imita la interfaz de DetrForObjectDetection / RTDetrForObjectDetection con salidas controladas."""

    def __init__(self, logits=None, boxes=None):
        super().__init__()
        self.backbone_conv = torch.nn.Linear(1, 1)          # el nombre contiene 'backbone'
        self.decoder_head = torch.nn.Linear(1, 1)
        self.logits, self.boxes, self.seen = logits, boxes, {}

    def forward(self, pixel_values, labels=None):
        self.seen = {"pixel_values": pixel_values, "labels": labels}
        n = pixel_values.shape[0]
        loss = self.decoder_head.weight.sum() + 1.0 if labels is not None else None
        expand = lambda t: None if t is None else t.expand(n, -1, -1)   # noqa: E731
        return SimpleNamespace(loss=loss, logits=expand(self.logits), pred_boxes=expand(self.boxes))


def images(n=2, size=100, value=0.5):
    return [torch.full((3, size, size), value) for _ in range(n)]


def test_box_conversions_known_values_roundtrip_and_empty():
    b = torch.tensor([[10., 20., 50., 60.]])
    c = xyxy_to_cxcywh_norm(b, width=100, height=200)
    assert torch.allclose(c, torch.tensor([[0.3, 0.2, 0.4, 0.2]]))
    assert torch.allclose(cxcywh_norm_to_xyxy(c, 100, 200), b)
    assert xyxy_to_cxcywh_norm(torch.zeros((0, 4)), 100, 100).shape == (0, 4)


def test_training_converts_labels_and_boxes_and_returns_a_single_loss():
    fake = FakeHF()
    w = HFDetectionWrapper(fake, "rtdetr", normalize=False).train()
    tg = [{"boxes": torch.tensor([[10., 20., 50., 60.]]), "labels": torch.tensor([1])},
          {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}]      # imagen sin espigas
    out = w(images(), tg)
    assert set(out) == {"hf_loss"} and out["hf_loss"].requires_grad
    lab = fake.seen["labels"]
    assert lab[0]["class_labels"].tolist() == [0]                                              # 1 (torchvision) -> 0 (HF)
    assert torch.allclose(lab[0]["boxes"], torch.tensor([[0.3, 0.4, 0.4, 0.4]]))
    assert lab[1]["class_labels"].shape == (0,) and lab[1]["boxes"].shape == (0, 4)
    assert torch.allclose(fake.seen["pixel_values"], torch.stack(images()))                      # RT-DETR: sin normalizar


def test_training_without_targets_is_rejected():
    w = HFDetectionWrapper(FakeHF(), "detr", normalize=True).train()
    with pytest.raises(ValueError, match="objetivos"):
        w(images())


def test_detr_normalizes_with_imagenet_statistics_and_rtdetr_does_not():
    logits, boxes = torch.zeros(1, 1, 2), torch.full((1, 1, 4), 0.5)
    fake = FakeHF(logits, boxes)
    HFDetectionWrapper(fake, "detr", normalize=True).eval()(images(1))
    got = fake.seen["pixel_values"][0, :, 0, 0]
    expected = torch.tensor([(0.5 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)])
    assert torch.allclose(got, expected, atol=1e-6)
    fake2 = FakeHF(torch.zeros(1, 1, 1), boxes)
    HFDetectionWrapper(fake2, "rtdetr", normalize=False).eval()(images(1))
    assert torch.allclose(fake2.seen["pixel_values"], torch.full((1, 3, 100, 100), 0.5))


def test_rtdetr_postprocessing_uses_sigmoid_threshold_and_clamps_to_the_image():
    logits = torch.tensor([[[2.0], [-3.0], [0.0]]])                                 # sigmoide: 0.881, 0.047, 0.5
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.4, 0.4]]])
    w = HFDetectionWrapper(FakeHF(logits, boxes), "rtdetr", normalize=False, score_thresh=0.1).eval()
    res = w(images(1))
    assert len(res) == 1 and len(res[0]["scores"]) == 2                              # la de 0.047 queda fuera
    assert torch.allclose(res[0]["scores"], torch.tensor([1 / (1 + math.exp(-2.0)), 0.5]), atol=1e-6)
    assert torch.allclose(res[0]["boxes"][0], torch.tensor([40., 40., 60., 60.]))
    assert torch.allclose(res[0]["boxes"][1], torch.tensor([70., 70., 100., 100.]))   # recortada al borde (110 -> 100)
    assert res[0]["labels"].dtype == torch.int64 and (res[0]["labels"] == 1).all()


def test_detr_postprocessing_uses_softmax_with_a_last_no_object_class():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])                                # clase espiga, sin objeto
    boxes = torch.full((1, 2, 4), 0.5)
    w = HFDetectionWrapper(FakeHF(logits, boxes), "detr", normalize=True, score_thresh=0.5).eval()
    res = w(images(1))
    assert len(res[0]["scores"]) == 1 and abs(float(res[0]["scores"][0]) - math.exp(2) / (math.exp(2) + 1)) < 1e-6


def test_max_dets_keeps_the_highest_scores_per_image():
    logits = torch.arange(10, dtype=torch.float32).view(1, 10, 1)                    # puntuaciones crecientes
    boxes = torch.full((1, 10, 4), 0.5)
    res = HFDetectionWrapper(FakeHF(logits, boxes), "rtdetr", normalize=False, max_dets=3, score_thresh=0.0).eval()(images(2))
    assert len(res) == 2
    for r in res:
        assert len(r["scores"]) == 3 and torch.allclose(r["scores"], torch.sigmoid(torch.tensor([9., 8., 7.])))


def test_param_groups_split_backbone_and_cover_every_trainable_parameter_once():
    w = HFDetectionWrapper(FakeHF(), "rtdetr", normalize=False)
    g = w.param_groups(backbone_lr_mult=0.1)
    assert [x["lr_scale"] for x in g] == [1.0, 0.1]
    ids = [id(p) for grp in g for p in grp["params"]]
    assert sorted(ids) == sorted(id(p) for p in w.hf.parameters()) and len(ids) == len(set(ids))
    assert all(id(p) in {id(q) for q in w.hf.backbone_conv.parameters()} for p in g[1]["params"])


def test_hf_builders_require_pretrained_weights_and_unknown_kind_is_rejected():
    cfg = dict(arch="detr", variant="v1", pretrained=False, image_size=128, trainable_backbone_layers=3,
               detections_per_img=50, score_thresh_eval=0.001, nms_thresh=0.5)
    with pytest.raises(ValueError, match="preentrenados"):
        build_detector(cfg)
    with pytest.raises(ValueError):
        HFDetectionWrapper(FakeHF(), "yolo", normalize=False)