"""Entrena un pix2pix layout -> imagen de trigo sobre recortes de las imágenes de TRAIN. No lee validación ni prueba.

Cada muestra: recorte aleatorio de `crop` px (volteo/giro aleatorios) de una imagen de train, y su layout de cajas
(cajas con >= 50 % de área visible; el resto del recorte queda sin etiqueta, igual que en los aumentos del proyecto).
Pérdida: GAN (BCE, PatchGAN) + 100 * L1. Entrenamiento por TIEMPO (--minutes): lr constante la primera mitad y descenso
lineal a 0 en la segunda. Reanudable (--resume).

Uso:
  python -m src.gan.train_gan --minutes 75
  python -m src.gan.train_gan --smoke            # CPU/GPU, 20 iteraciones con un modelo diminuto
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import load_config, resolve
from src.data.samples import samples_from_config
from src.gan.layout import apply_array, crop_boxes, render_layout
from src.gan.models import PatchDiscriminator, UNetGenerator, init_weights
from src.training.utils import CsvLogger, format_eta, set_seed


class CropDataset(Dataset):
    def __init__(self, samples, crop: int = 256, per_image: int = 4, min_vis: float = 0.5):
        self.samples, self.crop, self.per_image, self.min_vis = samples, int(crop), int(per_image), float(min_vis)

    def __len__(self):
        return len(self.samples) * self.per_image

    def __getitem__(self, i):
        img, boxes, _ = self.samples.get(i % len(self.samples))          # train=False: imagen sin aumentos
        h, w = img.shape[:2]
        c = self.crop
        x0, y0 = random.randint(0, w - c), random.randint(0, h - c)
        patch = np.ascontiguousarray(img[y0:y0 + c, x0:x0 + c]).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0
        cond = render_layout(crop_boxes(boxes, x0, y0, c, c, self.min_vis), c)
        flip, k = random.random() < 0.5, random.randrange(4)
        return torch.from_numpy(apply_array(cond, flip, k)), torch.from_numpy(apply_array(patch, flip, k))


def save_grid(path: Path, cond, fake, real) -> None:
    """Filas: layout | generada | real."""
    def to_img(t):
        return ((t.detach().float().clamp(-1, 1).cpu().numpy() + 1) * 127.5).astype(np.uint8).transpose(1, 2, 0)
    rows = []
    for c, f, r in zip(cond, fake, real):
        lay = np.repeat(((c[0:1].cpu().numpy() + 1) * 127.5).astype(np.uint8).transpose(1, 2, 0), 3, axis=2)
        rows.append(np.concatenate([lay, to_img(f), to_img(r)], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--out", default="outputs/gan/pix2pix")
    ap.add_argument("--minutes", type=float, default=75.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--num-downs", type=int, default=7)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--l1", type=float, default=100.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-amp", action="store_true", help="fp32 en lugar de bf16")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.minutes, a.ngf, a.num_downs, a.crop, a.batch_size, a.workers = 0.05, 8, 5, 64, 4, 0
    if a.crop % (2 ** a.num_downs):
        raise SystemExit(f"--crop debe ser múltiplo de 2**num_downs = {2 ** a.num_downs}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda" and not a.no_amp and torch.cuda.is_bf16_supported()
    set_seed(a.seed)
    out = resolve(a.out) if not a.smoke else resolve(a.out + "_smoke")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config)
    samples = samples_from_config(cfg, "train", train=False, image_size=1024)      # SOLO train
    ds = CropDataset(samples, a.crop)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, drop_last=True,
                    pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)

    G = UNetGenerator(2, 3, a.ngf, a.num_downs).to(dev)
    D = PatchDiscriminator(5, 64 if not a.smoke else 8).to(dev)
    G.apply(init_weights)
    D.apply(init_weights)
    oG = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr=a.lr, betas=(0.5, 0.999))
    bce, l1 = torch.nn.BCEWithLogitsLoss(), torch.nn.L1Loss()
    it, elapsed0 = 0, 0.0
    last = out / "last.pt"
    if a.resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=True)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]); oG.load_state_dict(ck["oG"]); oD.load_state_dict(ck["oD"])
        it, elapsed0 = int(ck["it"]), float(ck["elapsed"])
        print(f"Reanudando en la iteración {it} ({elapsed0 / 60:.1f} min acumulados).")
    (out / "config.json").write_text(json.dumps({**vars(a), "amp_bf16": amp, "n_train_images": len(samples),
                                                 "params_G": sum(p.numel() for p in G.parameters()),
                                                 "params_D": sum(p.numel() for p in D.parameters())}, indent=2), encoding="utf-8")
    print(f"pix2pix | {len(samples)} imágenes de train | G {sum(p.numel() for p in G.parameters()) / 1e6:.1f} M, "
          f"D {sum(p.numel() for p in D.parameters()) / 1e6:.1f} M | lote {a.batch_size}, recorte {a.crop} | "
          f"{a.minutes:g} min | bf16 {amp} | {dev}", flush=True)

    random.seed(123)
    fixed = [ds[j * 37] for j in range(6)]                                    # muestras fijas para ver la evolución
    fc, fr = torch.stack([f[0] for f in fixed]).to(dev), torch.stack([f[1] for f in fixed]).to(dev)
    random.seed(a.seed + it)
    logger = CsvLogger(out / "log.csv")
    budget, t0, t_save, t_samp = a.minutes * 60.0, time.perf_counter(), 0.0, 0.0
    t_log = t0
    done, acc = False, {"d": 0.0, "g_gan": 0.0, "g_l1": 0.0, "n": 0}
    G.train(); D.train()

    def save_all(final=False):
        tmp = out / "last.pt.tmp"
        torch.save({"G": G.state_dict(), "D": D.state_dict(), "oG": oG.state_dict(), "oD": oD.state_dict(), "it": it,
                    "elapsed": elapsed0 + time.perf_counter() - t0}, tmp)
        os.replace(tmp, last)
        torch.save({"G": G.state_dict(), "ngf": a.ngf, "num_downs": a.num_downs, "crop": a.crop, "it": it}, out / "generator.pt")

    while not done:
        for cond, real in dl:
            el = elapsed0 + time.perf_counter() - t0
            frac = el / budget
            if frac >= 1.0:
                done = True
                break
            lr = a.lr if frac < 0.5 else a.lr * 2.0 * (1.0 - frac)
            for o in (oG, oD):
                for g in o.param_groups:
                    g["lr"] = lr
            cond, real = cond.to(dev, non_blocking=True), real.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                fake = G(cond)
                d_real = D(cond, real).float()
                d_fake = D(cond, fake.detach()).float()
                loss_d = 0.5 * (bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake)))
            oD.zero_grad(set_to_none=True)
            loss_d.backward()
            oD.step()
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                d_fake = D(cond, fake).float()
                loss_gan = bce(d_fake, torch.ones_like(d_fake))
                loss_l1 = l1(fake.float(), real) * a.l1
                loss_g = loss_gan + loss_l1
            oG.zero_grad(set_to_none=True)
            loss_g.backward()
            oG.step()
            it += 1
            if not all(torch.isfinite(x) for x in (loss_d, loss_g)):
                raise SystemExit(f"Pérdida no finita en la iteración {it}: se detiene (el último last.pt es válido).")
            acc["d"] += float(loss_d.detach()); acc["g_gan"] += float(loss_gan.detach()); acc["g_l1"] += float(loss_l1.detach()) / a.l1; acc["n"] += 1
            if it % 100 == 0:
                n = acc["n"]
                row = {"it": it, "elapsed_min": round(el / 60, 2), "lr": lr, "loss_d": acc["d"] / n,
                       "loss_g_gan": acc["g_gan"] / n, "l1": acc["g_l1"] / n, "it_per_s": round(n / max(1e-9, time.perf_counter() - t_log), 2)}
                logger.log(row)
                print(f"[{it:6d}] {el / 60:5.1f}/{a.minutes:g} min | D {row['loss_d']:.3f} | G_gan {row['loss_g_gan']:.3f} | "
                      f"L1 {row['l1']:.4f} | lr {lr:.1e} | ETA {format_eta(budget - el)}", flush=True)
                acc, t_log = {"d": 0.0, "g_gan": 0.0, "g_l1": 0.0, "n": 0}, time.perf_counter()
            now = time.perf_counter() - t0
            if now - t_save > 300:
                save_all(); t_save = now
            if now - t_samp > 600:
                with torch.no_grad():
                    G.eval(); save_grid(out / f"samples_it{it:06d}.png", fc, G(fc), fr); G.train()
                t_samp = now
    save_all(final=True)
    with torch.no_grad():
        G.eval(); save_grid(out / "samples_final.png", fc, G(fc), fr)
    print(f"\nListo: {it} iteraciones. Generador en {out / 'generator.pt'}; muestras en {out / 'samples_final.png'} "
          "(columnas: layout | generada | real).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
