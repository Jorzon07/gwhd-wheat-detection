"""Aprendizaje federado (FedAvg) sobre GWHD: cada dominio de train es un cliente. NO usa nunca la partición de prueba.

Un servidor mantiene el modelo global. En cada ronda: se eligen clientes, cada uno parte de los pesos globales, entrena
`local_epochs` épocas sobre SUS imágenes (optimizador AdamW nuevo, como es habitual en FedAvg) y devuelve sus pesos; el
servidor los promedia ponderando por el número de imágenes. El modelo global se evalúa en validación con el evaluador
común tras cada ronda. Los datos de un cliente nunca se mezclan con los de otro (solo viajan pesos).

Uso:
  python -m src.training.federated --smoke
  python -m src.training.federated --name fedavg_domain --rounds 9 --local-epochs 1
  python -m src.training.federated --name fedavg_iid --partition iid         # mismos tamaños, reparto aleatorio
  python -m src.training.federated --name fedavg_single --partition single   # 1 cliente = control de la implementación
  python -m src.training.federated --resume outputs/runs/<carpeta>
Mismo formato de salida que train.py (config.json, summary.json, val_predictions_best.npz), así que sirve con compare_runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np
import torch

from src.training.utils import (BestTracker, CheckpointManager, CsvLogger, environment_info, format_eta, lr_at,
                                make_run_dir, save_predictions, set_seed, to_builtin)

# Receta de RT-DETR elegida por Hyperband (ver outputs/hyperband/rtdetr/best.json)
HB = {"arch": "rtdetr", "image_size": 800, "batch_size": 4, "optimizer": "adamw", "lr": 0.000246604,
      "grad_clip": 10.0, "backbone_lr_mult": 0.3}


# ---------------------------------------------------------------- piezas puras (probadas sin GPU) ----------
class Accumulator:
    """Media ponderada de pesos en flujo (no guarda todos los clientes a la vez). Los tensores no flotantes
    (contadores de BatchNorm) conservan el valor global."""

    def __init__(self, reference: dict):
        self.ref = {k: v.detach().cpu().clone() for k, v in reference.items()}
        self.sum = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in self.ref.items() if v.is_floating_point()}
        self.total = 0.0

    def add(self, state: dict, weight: float) -> None:
        if weight <= 0:
            raise ValueError("el peso de un cliente debe ser positivo")
        for k in self.sum:
            self.sum[k] += float(weight) * state[k].detach().float().cpu()
        self.total += float(weight)

    def result(self) -> dict:
        if self.total <= 0:
            raise ValueError("ningún cliente aportó pesos")
        out = {}
        for k, v in self.ref.items():
            out[k] = (self.sum[k] / self.total).to(v.dtype) if k in self.sum else v.clone()
        return out


def fedavg(global_state: dict, client_states: list, weights: list) -> dict:
    acc = Accumulator(global_state)
    for s, w in zip(client_states, weights):
        acc.add(s, w)
    return acc.result()


def partition_clients(domains: list, mode: str, seed: int = 0) -> dict:
    """{cliente: [índices]}. 'domain': un cliente por dominio. 'single': un único cliente con todo. 'iid': mismos
    tamaños que 'domain' pero con imágenes repartidas al azar."""
    n = len(domains)
    by: dict = {}
    for i, d in enumerate(domains):
        by.setdefault(str(d), []).append(i)
    if mode == "domain":
        return {k: by[k] for k in sorted(by)}
    if mode == "single":
        return {"all": list(range(n))}
    if mode == "iid":
        perm = np.random.default_rng(seed).permutation(n)
        out, pos = {}, 0
        for k in sorted(by):
            m = len(by[k])
            out[f"iid_{k}"] = sorted(int(i) for i in perm[pos:pos + m])
            pos += m
        return out
    raise ValueError(f"partición desconocida: {mode!r}")


def select_clients(names: list, k: int, rnd: int, seed: int) -> list:
    """Todos los clientes si k <= 0 o k >= len; si no, k al azar (reproducible por ronda)."""
    names = sorted(names)
    if k <= 0 or k >= len(names):
        return names
    rng = np.random.default_rng([int(seed), int(rnd)])
    return sorted(rng.choice(names, size=k, replace=False).tolist())


def round_lr(rnd: int, rounds: int, base_lr: float, min_lr_ratio: float) -> float:
    """Coseno por rondas (sin calentamiento global; el calentamiento es local, dentro de cada cliente)."""
    return lr_at(rnd, rounds, base_lr, 0, min_lr_ratio)


def run_round(model, global_state: dict, selected: list, train_fn, skip_exc=(RuntimeError,)):
    """Una ronda. `selected` = [(nombre, n_imágenes)]; `train_fn(nombre) -> dict` entrena `model` (ya cargado con el
    estado global) y devuelve estadísticas. Un cliente que falla con `skip_exc` se omite y se anota. Devuelve
    (nuevo estado global, lista de estadísticas por cliente)."""
    acc, stats = Accumulator(global_state), []
    for name, n in selected:
        model.load_state_dict(global_state)
        t0 = time.perf_counter()
        try:
            info = dict(train_fn(name))
        except skip_exc as e:
            stats.append({"client": name, "n_images": n, "ok": False, "error": str(e)[:200]})
            continue
        acc.add(model.state_dict(), n)
        stats.append({"client": name, "n_images": n, "ok": True, "time_s": round(time.perf_counter() - t0, 1), **info})
    if acc.total <= 0:
        raise RuntimeError("Ningún cliente completó la ronda")
    return acc.result(), stats


def count_params(state: dict) -> int:
    return int(sum(v.numel() for v in state.values() if v.is_floating_point()))


# ---------------------------------------------------------------- programa principal ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--name", default="fedavg")
    ap.add_argument("--resume", default=None, help="carpeta de una ejecución anterior")
    ap.add_argument("--smoke", action="store_true", help="2 rondas, 3 clientes, 8 imágenes por cliente, 16 de validación")
    ap.add_argument("--partition", choices=["domain", "iid", "single"], default="domain")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--clients-per-round", type=int, default=0, help="0 = todos los clientes en cada ronda")
    ap.add_argument("--local-warmup", type=int, default=10, help="pasos de calentamiento lineal al inicio de cada cliente")
    ap.add_argument("--arch", default=HB["arch"], choices=["frcnn", "retinanet", "detr", "rtdetr"])
    ap.add_argument("--image-size", type=int, default=HB["image_size"])
    ap.add_argument("--batch-size", type=int, default=HB["batch_size"])
    ap.add_argument("--lr", type=float, default=HB["lr"])
    ap.add_argument("--grad-clip", type=float, default=HB["grad_clip"])
    ap.add_argument("--backbone-lr-mult", type=float, default=HB["backbone_lr_mult"])
    ap.add_argument("--no-amp", action="store_true")
    return ap.parse_args()


def main() -> int:
    from src.config import load_config, resolve
    from src.data.dataset import WheatDataset, make_loader, set_epoch
    from src.data.io_utils import sha256_of
    from src.data.samples import SPLIT_FILES, samples_from_config
    from src.models.factory import build_detector, count_parameters
    from src.training.engine import NonFiniteLossError, evaluate_model, measure_inference, train_one_epoch

    args = parse_args()
    cfg = load_config(args.config)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA no está disponible: ejecuta scripts/check_gpu.py")
    device = torch.device("cuda")
    original_size = int(cfg["dataset"]["expected_image_size"][0])
    lc = cfg["phase3"]["loader"]

    if args.resume:
        run_dir = resolve(args.resume)
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        tcfg, fed = saved["train"], saved["federated"]
        print(f"Reanudando {run_dir.name} con su configuración guardada (los demás argumentos se ignoran).")
    else:
        tcfg = dict(cfg["train"])
        tcfg.update({"arch": args.arch, "image_size": args.image_size, "batch_size": args.batch_size,
                     "optimizer": "adamw", "lr": args.lr, "grad_clip": args.grad_clip,
                     "backbone_lr_mult": args.backbone_lr_mult, "amp": not args.no_amp, "accum_steps": 1,
                     "smoke": bool(args.smoke)})
        fed = {"partition": args.partition, "rounds": 2 if args.smoke else args.rounds,
               "local_epochs": args.local_epochs, "clients_per_round": args.clients_per_round,
               "local_warmup": args.local_warmup, "aggregation": "FedAvg ponderado por n_imágenes",
               "client_optimizer": "AdamW nuevo en cada cliente y ronda"}
        run_dir = make_run_dir(resolve(tcfg["runs_dir"]), "smoke_fed" if args.smoke else args.name)
        saved = None
    smoke, seed = bool(tcfg["smoke"]), 42
    set_seed(seed)
    size, amp = int(tcfg["image_size"]), bool(tcfg["amp"])
    rounds, E = int(fed["rounds"]), int(fed["local_epochs"])

    # ---- datos: solo train (clientes) y validación (evaluación del modelo global) ----
    train_samples = samples_from_config(cfg, "train", train=True, image_size=size)
    val_samples = samples_from_config(cfg, "validation", train=False, image_size=size)
    clients = partition_clients(train_samples.meta["domain"].astype(str).tolist(), fed["partition"], seed)
    if smoke:
        keep = sorted(clients, key=lambda k: len(clients[k]))[:3]
        clients = {k: clients[k][:8] for k in keep}
    val_ds = WheatDataset(val_samples, list(range(16)) if smoke else None)
    val_loader = make_loader(val_ds, int(tcfg["eval_batch_size"]), shuffle=False, num_workers=int(lc["num_workers"]),
                             pin_memory=bool(lc["pin_memory"]), seed=seed)
    sizes = {k: len(v) for k, v in clients.items()}

    model = build_detector(tcfg).to(device)
    total_p, trainable_p = count_parameters(model)
    global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    n_float = count_params(global_state)

    mgr = CheckpointManager(run_dir, torch.save, lambda p: torch.load(p, map_location="cpu", weights_only=True))
    logger, client_logger = CsvLogger(run_dir / "metrics.csv"), CsvLogger(run_dir / "clients.csv")
    tracker = BestTracker(0)                                   # sin parada anticipada: se hacen todas las rondas
    split_dir = resolve(cfg["phase3"]["split_dir"])
    hashes = {n: sha256_of(split_dir / SPLIT_FILES[n]) for n in ("train", "validation")}
    start_round = 0
    if args.resume:
        if saved["split_sha256"] != hashes:
            raise SystemExit("Los CSV de la partición han cambiado desde que empezó esta ejecución: no se puede reanudar.")
        ckpt = mgr.load("last.pt")
        if ckpt is None:
            raise SystemExit(f"{run_dir} no tiene last.pt")
        global_state = {k: v.clone() for k, v in ckpt["model"].items()}
        tracker.load_state_dict(ckpt["tracker"])
        start_round = ckpt["round"] + 1
        print(f"Reanudando en la ronda {start_round + 1}/{rounds}.")
    else:
        (run_dir / "config.json").write_text(json.dumps({
            "train": tcfg, "phase3": cfg["phase3"], "split_dir": str(split_dir), "split_sha256": hashes,
            "federated": fed, "clients": sizes, "environment": environment_info(),
            "parameters": {"total": total_p, "trainable": trainable_p},
            "n_train_images": int(sum(sizes.values())), "n_val_images": len(val_ds),
            "started": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    print(f"Carpeta: {run_dir}\nFedAvg {tcfg['arch']} | partición {fed['partition']} | {len(clients)} clientes, "
          f"{sum(sizes.values())} imágenes (de {min(sizes.values())} a {max(sizes.values())} por cliente) | "
          f"{rounds} rondas x {E} época(s) local(es) | {total_p / 1e6:.1f} M parámetros | tamaño {size}", flush=True)

    t_start, round_times, comm_bytes, last_round = time.perf_counter(), [], 0.0, start_round - 1
    try:
        for rnd in range(start_round, rounds):
            lr = round_lr(rnd, rounds, float(tcfg["lr"]), float(tcfg["min_lr_ratio"]))
            chosen = select_clients(list(clients), int(fed["clients_per_round"]), rnd, seed)
            warm = int(fed["local_warmup"])

            def train_fn(name, rnd=rnd, lr=lr):
                ds = WheatDataset(train_samples, clients[name])
                loader = make_loader(ds, int(tcfg["batch_size"]), shuffle=True, num_workers=int(lc["num_workers"]),
                                     pin_memory=bool(lc["pin_memory"]), seed=seed)
                params = model.param_groups(float(tcfg["backbone_lr_mult"])) if hasattr(model, "param_groups") \
                    else [p for p in model.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(tcfg["weight_decay"]))
                scaler = torch.amp.GradScaler("cuda", enabled=amp)
                lr_fn = lambda s: lr * min(1.0, 0.1 + 0.9 * (s + 1) / max(1, warm))      # noqa: E731
                step, last = 0, {}
                for e in range(E):
                    set_epoch(loader, rnd * E + e, seed)
                    last = train_one_epoch(model, loader, opt, scaler, device, lr_fn, step,
                                           float(tcfg["grad_clip"]) or None, 1, amp)
                    step = last["step"]
                return {"loss": last["loss"], "skipped_batches": last["skipped_batches"], "steps": step}

            global_state, stats = run_round(model, global_state, [(c, sizes[c]) for c in chosen], train_fn,
                                            skip_exc=(NonFiniteLossError,))
            n_ok = sum(1 for s in stats if s["ok"])
            comm_bytes += n_ok * 2 * n_float * 4                  # bajada + subida, float32, por cliente
            for s in stats:
                client_logger.log(to_builtin({"round": rnd, "lr": lr, **s}))
            last_round = rnd

            model.load_state_dict(global_state)                   # evaluación del modelo GLOBAL en validación
            t_ev = time.perf_counter()
            res, preds = evaluate_model(model, val_loader, val_samples, device, original_size, amp)
            improved = tracker.update(rnd, res[tcfg["selection_metric"]])
            row = {"round": rnd, "lr": lr, "clients_ok": n_ok, "clients_failed": len(stats) - n_ok,
                   "mean_client_loss": float(np.mean([s["loss"] for s in stats if s["ok"]])) if n_ok else float("nan"),
                   "comm_mb_cumulative": round(comm_bytes / 2 ** 20, 1), "eval_time_s": round(time.perf_counter() - t_ev, 1),
                   **{f"val_{k}": v for k, v in res.items() if isinstance(v, (int, float))}}
            if improved:
                mgr.save_best(to_builtin({"model": global_state, "epoch": rnd, "metrics": res, "config": tcfg}))
                save_predictions(run_dir / "val_predictions_best.npz", preds)
            mgr.save_last(to_builtin({"model": global_state, "round": rnd, "tracker": tracker.state_dict(),
                                      "config": tcfg}))
            logger.log(to_builtin(row))
            round_times.append(time.perf_counter() - t_start)
            eta = (round_times[-1] / (rnd - start_round + 1)) * (rounds - rnd - 1)
            print(f"[ronda {rnd + 1}/{rounds}] lr {lr:.2e} | {n_ok}/{len(stats)} clientes | pérdida media "
                  f"{row['mean_client_loss']:.3f} | val mAP50 {res['map50']:.3f} mAP50-95 {res['map50_95']:.3f} "
                  f"P {res['precision']:.3f} R {res['recall']:.3f}" + (" *mejor*" if improved else "")
                  + f" | ETA {format_eta(eta)}", flush=True)
    except torch.cuda.OutOfMemoryError:
        print("\nSin memoria en la GPU. Reduce --batch-size. El último checkpoint sigue siendo válido (--resume).")
        return 3

    best = mgr.load("best.pt")
    summary = {"run_dir": str(run_dir), "smoke": smoke, "epochs_completed": last_round + 1, "stopped_early": False,
               "selection_metric": tcfg["selection_metric"], "best_epoch": tracker.best_epoch,
               "best_value": tracker.best, "best_val_metrics": best["metrics"] if best else None,
               "wall_time_s": round(time.perf_counter() - t_start, 1), "federated": fed,
               "comm_total_mb": round(comm_bytes / 2 ** 20, 1), "params_floats": n_float,
               "parameters": {"total": total_p, "trainable": trainable_p}, "environment": environment_info()}
    if best:
        model.load_state_dict(best["model"])
        summary.update(measure_inference(model, val_samples, device, amp))
    (run_dir / "summary.json").write_text(json.dumps(to_builtin(summary), indent=2), encoding="utf-8")
    print("\nResumen guardado en", run_dir / "summary.json")
    if best:
        m = best["metrics"]
        print(f"Mejor ronda {best['epoch'] + 1}: mAP50 {m['map50']:.3f} | mAP50-95 {m['map50_95']:.3f} | "
              f"mejor F1 {m['best_f1']:.3f} con umbral {m['best_f1_threshold']:.3f} | "
              f"comunicación total {summary['comm_total_mb']:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
