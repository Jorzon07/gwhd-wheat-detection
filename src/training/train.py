"""Entrenamiento de un detector de espigas con validación en cada época. NO usa nunca la partición de prueba.

Uso:
  python -m src.training.train --smoke [--image-size 640 --batch-size 2]        # sobreajuste de 16 imágenes
  python -m src.training.train --name baseline_frcnn [--epochs 12 --image-size 1024 --batch-size 2 --variant v1]
  python -m src.training.train --resume outputs/runs/<carpeta>                   # continúa con su configuración guardada
Cada ejecución crea una carpeta nueva en outputs/runs/ (nunca se sobrescribe una existente).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime

import torch

from src.config import load_config, resolve
from src.data.dataset import WheatDataset, make_loader, set_epoch
from src.data.io_utils import sha256_of
from src.data.samples import SPLIT_FILES, samples_from_config
from src.models.factory import build_detector, count_parameters
from src.training.engine import NonFiniteLossError, evaluate_model, measure_inference, train_one_epoch
from src.training.utils import (BestTracker, CheckpointManager, CsvLogger, environment_info, format_eta, lr_at,
                                make_run_dir, save_predictions, set_seed, to_builtin, validate_resume)

# claves que no pueden cambiar al reanudar una ejecución
RESUME_KEYS = ["arch", "variant", "pretrained", "trainable_backbone_layers", "image_size", "batch_size", "accum_steps",
               "epochs", "optimizer", "lr", "momentum", "weight_decay", "warmup_steps", "min_lr_ratio", "amp",
               "seed", "train_images", "smoke", "grad_clip", "backbone_lr_mult"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--resume", default=None, help="carpeta de una ejecución anterior")
    ap.add_argument("--smoke", action="store_true", help="prueba de humo: sobreajusta unas pocas imágenes sin aumentos")
    ap.add_argument("--smoke-images", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--accum-steps", type=int, default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--variant", choices=["v1", "v2"], default=None)
    ap.add_argument("--arch", choices=["frcnn", "retinanet", "detr", "rtdetr"], default=None, help="arquitectura (por defecto, la de la configuración)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--optimizer", choices=["sgd", "adamw"], default=None)
    ap.add_argument("--grad-clip", type=float, default=None, help="norma máxima del gradiente")
    ap.add_argument("--backbone-lr-mult", type=float, default=None, help="solo DETR/RT-DETR: factor del lr del backbone")
    ap.add_argument("--train-images", type=int, default=None, help="usa solo las N primeras imágenes de train")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    return ap.parse_args()


def build_train_config(cfg: dict, args) -> dict:
    t = dict(cfg["train"])
    for key, val in (("epochs", args.epochs), ("batch_size", args.batch_size), ("accum_steps", args.accum_steps),
                     ("image_size", args.image_size), ("variant", args.variant), ("arch", args.arch), ("lr", args.lr),
                     ("train_images", args.train_images), ("optimizer", args.optimizer),
                     ("grad_clip", args.grad_clip), ("backbone_lr_mult", args.backbone_lr_mult)):
        if val is not None:
            t[key] = val
    if args.no_amp:
        t["amp"] = False
    if args.no_pretrained:
        t["pretrained"] = False
    t["smoke"] = bool(args.smoke)
    if args.smoke:                                   # valores por defecto de la prueba de humo (si no se piden otros)
        t["train_images"] = args.train_images or args.smoke_images
        t["epochs"] = args.epochs or 30
        t["accum_steps"] = args.accum_steps or 1
        t["lr"] = args.lr or 0.01
        t["warmup_steps"], t["eval_every"], t["patience"] = 10, 5, 0
    return t


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA no está disponible: ejecuta scripts/check_gpu.py")
    device = torch.device("cuda")
    original_size = int(cfg["dataset"]["expected_image_size"][0])

    if args.resume:
        run_dir = resolve(args.resume)
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        tcfg = saved["train"]
        print(f"Reanudando {run_dir.name} con su configuración guardada (los demás argumentos se ignoran).")
    else:
        tcfg = build_train_config(cfg, args)
        run_dir = make_run_dir(resolve(tcfg["runs_dir"]), "smoke" if args.smoke else args.name)
        saved = None
    smoke, seed = bool(tcfg["smoke"]), int(tcfg["seed"])
    set_seed(seed)
    size, amp = int(tcfg["image_size"]), bool(tcfg["amp"])
    lc = cfg["phase3"]["loader"]

    # ---- datos (la prueba de humo evalúa sobre las mismas imágenes de train, sin aumentos) ----
    n_sub = tcfg.get("train_images")
    idx = list(range(int(n_sub))) if n_sub else None
    train_samples = samples_from_config(cfg, "train", train=not smoke, image_size=size)
    if smoke:
        val_samples = samples_from_config(cfg, "train", train=False, image_size=size)
        val_ds = WheatDataset(val_samples, idx)
    else:
        val_samples = samples_from_config(cfg, "validation", train=False, image_size=size)
        val_ds = WheatDataset(val_samples)
    train_loader = make_loader(WheatDataset(train_samples, idx), int(tcfg["batch_size"]), shuffle=True,
                               num_workers=int(lc["num_workers"]), pin_memory=bool(lc["pin_memory"]), seed=seed)
    val_loader = make_loader(val_ds, int(tcfg["eval_batch_size"]), shuffle=False,
                             num_workers=int(lc["num_workers"]), pin_memory=bool(lc["pin_memory"]), seed=seed)

    # ---- modelo y optimizador ----
    model = build_detector(tcfg).to(device)
    total_p, trainable_p = count_parameters(model)
    params = [p for p in model.parameters() if p.requires_grad]
    if hasattr(model, "param_groups"):                      # DETR/RT-DETR: backbone con learning rate reducido
        params = model.param_groups(float(tcfg.get("backbone_lr_mult", 0.1)))
    if tcfg["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params, lr=float(tcfg["lr"]), momentum=float(tcfg["momentum"]),
                                    weight_decay=float(tcfg["weight_decay"]))
    elif tcfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    else:
        raise SystemExit(f"optimizer desconocido: {tcfg['optimizer']}")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    epochs, accum = int(tcfg["epochs"]), int(tcfg["accum_steps"])
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = epochs * steps_per_epoch
    lr_fn = lambda s: lr_at(s, total_steps, float(tcfg["lr"]), int(tcfg["warmup_steps"]),            # noqa: E731
                            float(tcfg["min_lr_ratio"]))

    # ---- estado de la ejecución ----
    mgr = CheckpointManager(run_dir, torch.save, lambda p: torch.load(p, map_location="cpu", weights_only=True))
    logger, step_logger = CsvLogger(run_dir / "metrics.csv"), CsvLogger(run_dir / "train_steps.csv")
    tracker = BestTracker(int(tcfg["patience"]))
    split_dir = resolve(cfg["phase3"]["split_dir"])
    hashes = {n: sha256_of(split_dir / SPLIT_FILES[n]) for n in ("train", "validation")}
    start_epoch, global_step = 0, 0
    if args.resume:
        if saved["split_sha256"] != hashes:
            raise SystemExit("Los CSV de la partición han cambiado desde que empezó esta ejecución: no se puede reanudar.")
        ckpt = mgr.load("last.pt")
        if ckpt is None:
            raise SystemExit(f"{run_dir} no tiene last.pt")
        diff = validate_resume(ckpt["config"], tcfg, RESUME_KEYS)
        if diff:
            raise SystemExit("La configuración no coincide con el checkpoint:\n  " + "\n  ".join(diff))
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        tracker.load_state_dict(ckpt["tracker"])
        start_epoch, global_step = ckpt["epoch"] + 1, ckpt["global_step"]
        print(f"Reanudando en la época {start_epoch + 1}/{epochs} (paso {global_step}).")
    else:
        (run_dir / "config.json").write_text(json.dumps({
            "train": tcfg, "phase3": cfg["phase3"], "split_dir": str(split_dir), "split_sha256": hashes,
            "environment": environment_info(), "parameters": {"total": total_p, "trainable": trainable_p},
            "n_train_images": len(train_loader.dataset), "n_val_images": len(val_ds),
            "started": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    print(f"Carpeta: {run_dir}\nModelo {tcfg.get('arch', 'frcnn')}/{tcfg['variant']} | {total_p / 1e6:.1f} M parámetros ({trainable_p / 1e6:.1f} M entrenables) | "
          f"{len(train_loader.dataset)} imágenes de train, {len(val_ds)} de validación | tamaño {size} | "
          f"lote {tcfg['batch_size']}x{accum} | {epochs} épocas ({steps_per_epoch} pasos/época)")

    t_start, epoch_times, peak_gib = time.perf_counter(), [], 0.0
    stopped_early, last_epoch = False, start_epoch - 1
    try:
        for epoch in range(start_epoch, epochs):
            set_epoch(train_loader, epoch, seed)
            torch.cuda.reset_peak_memory_stats()
            tr = train_one_epoch(model, train_loader, optimizer, scaler, device, lr_fn, global_step,
                                 float(tcfg["grad_clip"]) or None, accum, amp,
                                 log_fn=lambda r, e=epoch: step_logger.log({"epoch": e, **to_builtin(r)}),
                                 log_every=int(tcfg["log_every"]))
            global_step = tr["step"]
            last_epoch = epoch
            gib = torch.cuda.max_memory_allocated() / 2 ** 30
            peak_gib = max(peak_gib, gib)
            row = {"epoch": epoch, "lr_end": optimizer.param_groups[0]["lr"], "train_peak_mem_gib": round(gib, 2),
                   **{f"train_{k}": v for k, v in tr.items() if k != "step"}}
            improved, msg = False, ""
            if (epoch + 1) % int(tcfg["eval_every"]) == 0 or epoch == epochs - 1:
                t_ev = time.perf_counter()
                res, preds = evaluate_model(model, val_loader, val_samples, device, original_size, amp)
                row.update({f"val_{k}": v for k, v in res.items() if isinstance(v, (int, float))})
                row["val_eval_time_s"] = round(time.perf_counter() - t_ev, 1)
                improved = tracker.update(epoch, res[tcfg["selection_metric"]])
                msg = (f" | val mAP50 {res['map50']:.3f} mAP50-95 {res['map50_95']:.3f} P {res['precision']:.3f} "
                       f"R {res['recall']:.3f} F1 {res['f1']:.3f}" + (" *mejor*" if improved else ""))
                if improved:
                    mgr.save_best(to_builtin({"model": model.state_dict(), "epoch": epoch, "metrics": res,
                                              "config": tcfg}))
                    save_predictions(run_dir / "val_predictions_best.npz", preds)
            mgr.save_last({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scaler": scaler.state_dict(), "epoch": epoch, "global_step": global_step,
                           "tracker": tracker.state_dict(), "config": to_builtin(tcfg)})
            logger.log(to_builtin(row))
            epoch_times.append(time.perf_counter() - t_start)
            done = epoch - start_epoch + 1
            eta = (epoch_times[-1] / done) * (epochs - epoch - 1)
            print(f"[ép {epoch + 1}/{epochs}] pérdida {tr['loss']:.3f} | {tr['images_per_s']:.1f} img/s | "
                  f"pico {gib:.1f} GiB | lotes omitidos {tr['skipped_batches']}{msg} | ETA {format_eta(eta)}", flush=True)
            if tracker.should_stop():
                stopped_early = True
                print(f"Parada anticipada: {tracker.patience} épocas sin mejora de {tcfg['selection_metric']}.")
                break
    except torch.cuda.OutOfMemoryError:
        print("\nSin memoria en la GPU. Reduce --batch-size (y sube --accum-steps para conservar el lote efectivo) "
              "o reduce --image-size. El último checkpoint guardado sigue siendo válido (--resume).")
        return 3
    except NonFiniteLossError as e:
        print(f"\nEntrenamiento abortado: {e}")
        return 4

    # ---- resumen final con los pesos de la mejor época ----
    best = mgr.load("best.pt")
    summary = {"run_dir": str(run_dir), "smoke": smoke, "epochs_completed": last_epoch + 1,
               "stopped_early": stopped_early, "selection_metric": tcfg["selection_metric"],
               "best_epoch": tracker.best_epoch, "best_value": tracker.best,
               "best_val_metrics": best["metrics"] if best else None,
               "wall_time_s": round(time.perf_counter() - t_start, 1), "peak_train_mem_gib": round(peak_gib, 2),
               "parameters": {"total": total_p, "trainable": trainable_p}, "environment": environment_info()}
    if best:
        model.load_state_dict(best["model"])
        summary.update(measure_inference(model, val_samples, device, amp))
    (run_dir / "summary.json").write_text(json.dumps(to_builtin(summary), indent=2), encoding="utf-8")
    print("\nResumen guardado en", run_dir / "summary.json")
    if best:
        m = best["metrics"]
        print(f"Mejor época {best['epoch'] + 1}: mAP50 {m['map50']:.3f} | mAP50-95 {m['map50_95']:.3f} | "
              f"mejor F1 {m['best_f1']:.3f} con umbral {m['best_f1_threshold']:.3f} | "
              f"inferencia {summary.get('inference_ms_median', float('nan')):.0f} ms/imagen")
    return 0


if __name__ == "__main__":
    sys.exit(main())