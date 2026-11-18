"""Motor de entrenamiento y evaluación para detectores de torchvision. Necesita PyTorch."""
from __future__ import annotations

import time

import numpy as np
import torch

from src.evaluation.metrics import evaluate
from src.training.utils import rescale_boxes


class NonFiniteLossError(RuntimeError):
    pass


def _to_device(images, targets, device):
    images = [im.to(device, non_blocking=True) for im in images]
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return images, targets


def train_one_epoch(model, loader, optimizer, scaler, device, lr_fn, start_step: int, grad_clip: float | None,
                    accum_steps: int, amp: bool, log_fn=None, log_every: int = 20, max_consecutive_bad: int = 10) -> dict:
    """Una época. `lr_fn(paso_del_optimizador) -> lr`. Con acumulación de gradiente, un paso del optimizador agrupa
    `accum_steps` lotes. Un lote con pérdida no finita se descarta (y con él los gradientes acumulados del grupo);
    más de `max_consecutive_bad` seguidos aborta el entrenamiento."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    n_batches, step = len(loader), start_step
    sums: dict[str, torch.Tensor] = {}
    n_good = bad_in_row = skipped = n_images = 0
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for i, (images, targets) in enumerate(loader):
        if i % accum_steps == 0:
            lr = lr_fn(step)
            for g in optimizer.param_groups:
                g["lr"] = lr * g.get("lr_scale", 1.0)          # DETR/RT-DETR: backbone con lr reducido
        images, targets = _to_device(images, targets, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        if not bool(torch.isfinite(loss)):
            skipped += 1
            bad_in_row += 1
            optimizer.zero_grad(set_to_none=True)
            if bad_in_row > max_consecutive_bad:
                raise NonFiniteLossError(f"{bad_in_row} lotes seguidos con pérdida no finita (época en curso, "
                                         f"paso {step}); revisa el learning rate o los datos")
            continue
        bad_in_row = 0
        scaler.scale(loss / accum_steps).backward()
        for k, v in loss_dict.items():
            sums[k] = sums.get(k, 0.0) + v.detach().float()
        n_good += 1
        n_images += len(images)
        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            grad_norm = None
            if grad_clip:
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, grad_clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if log_fn is not None and step % log_every == 0:
                log_fn({"step": step, "lr": optimizer.param_groups[0]["lr"], "loss": float(loss.detach()),
                        "grad_norm": grad_norm, "elapsed_s": round(time.perf_counter() - t0, 1)})
    dt = time.perf_counter() - t0
    comp = {k: float(v) / max(n_good, 1) for k, v in sums.items()}
    return {"loss": float(sum(comp.values())), **comp, "step": step, "skipped_batches": skipped,
            "time_s": dt, "images_per_s": n_images / dt if dt > 0 else float("nan")}


@torch.no_grad()
def predict(model, loader, names: list[str], device, scale: float, amp: bool) -> dict:
    """Predicciones por imagen, devueltas en el marco de la imagen ORIGINAL (cajas * scale). `names[image_id]`."""
    model.eval()
    preds = {}
    for images, targets in loader:
        images = [im.to(device, non_blocking=True) for im in images]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            outputs = model(images)
        for out, tgt in zip(outputs, targets):
            name = names[int(tgt["image_id"])]
            preds[name] = (rescale_boxes(out["boxes"].float().cpu().numpy(), scale),
                           out["scores"].float().cpu().numpy().astype(np.float64))
    return preds


def evaluate_model(model, loader, samples, device, original_size: int, amp: bool, score_threshold: float = 0.5):
    """Evalúa con el evaluador común. `samples` es el WheatSamples cuyo loader se usa (para nombres y verdad terreno,
    que se toma en el marco original). Devuelve (métricas, predicciones)."""
    scale = original_size / samples.image_size
    preds = predict(model, loader, samples.names, device, scale, amp)
    gts = {n: samples.boxes[n] for n in preds}
    res = evaluate(preds, gts, image_size=(original_size, original_size), score_threshold=score_threshold)
    return res, preds


@torch.no_grad()
def measure_inference(model, samples, device, amp: bool, n: int = 100, warmup: int = 10) -> dict:
    """Tiempo de inferencia por imagen con lote 1 (sincronizando la GPU), sin incluir la lectura de la imagen."""
    model.eval()
    times = []
    for k in range(min(n + warmup, len(samples))):
        img, _, _ = samples.get(k)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0).to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            model([x])
        torch.cuda.synchronize()
        if k >= warmup:
            times.append(time.perf_counter() - t0)
    if not times:
        return {"inference_ms_mean": float("nan"), "inference_ms_median": float("nan"), "inference_n_images": 0}
    return {"inference_ms_mean": 1000 * float(np.mean(times)), "inference_ms_median": 1000 * float(np.median(times)),
            "inference_n_images": len(times)}