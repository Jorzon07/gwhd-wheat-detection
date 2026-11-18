import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.models.factory import build_detector  # noqa: E402
from src.models.retinanet import replace_cls_logits  # noqa: E402


def cfg(arch):
    return dict(arch=arch, variant="v1", pretrained=False, image_size=128, trainable_backbone_layers=3,
                detections_per_img=50, score_thresh_eval=0.001, nms_thresh=0.5)


def test_retinanet_head_replacement_keeps_pretrained_convs_and_uses_focal_prior():
    from torchvision.models.detection import retinanet_resnet50_fpn
    model = retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=91)
    conv_before = model.head.classification_head.conv[0][0].weight.clone()
    replace_cls_logits(model, 2)
    head = model.head.classification_head
    assert head.num_classes == 2 and head.cls_logits.out_channels == head.num_anchors * 2
    assert torch.allclose(head.conv[0][0].weight, conv_before)                       # convoluciones intactas
    assert torch.allclose(head.cls_logits.bias, torch.full_like(head.cls_logits.bias, -math.log(99.0)))


@pytest.mark.parametrize("arch", ["frcnn", "retinanet"])
def test_detectors_run_train_and_eval_on_cpu(arch):
    torch.manual_seed(0)
    m = build_detector(cfg(arch))
    imgs = [torch.rand(3, 128, 128)]
    tg = [{"boxes": torch.tensor([[10., 10., 60., 60.]]), "labels": torch.tensor([1])}]
    m.train()
    losses = m(imgs, tg)
    assert losses and all(torch.isfinite(v) for v in losses.values())
    m.eval()
    with torch.no_grad():
        out = m(imgs)
    assert {"boxes", "labels", "scores"} <= set(out[0]) and len(out[0]["boxes"]) <= 50


@pytest.mark.parametrize("arch", ["frcnn", "retinanet"])
def test_images_without_ground_truth_do_not_break_training(arch):
    torch.manual_seed(0)
    m = build_detector(cfg(arch)).train()
    tg = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}]
    losses = m([torch.rand(3, 128, 128)], tg)
    assert all(torch.isfinite(v) for v in losses.values())


def test_unknown_arch_is_rejected():
    with pytest.raises(ValueError, match="arch"):
        build_detector(cfg("yolo"))