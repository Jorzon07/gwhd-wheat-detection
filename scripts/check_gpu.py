"""Verifica PyTorch + CUDA + torchvision en esta máquina. No descarga nada. Uso: python scripts/check_gpu.py"""
import sys
import time


def stage(name, fn):
    t0 = time.perf_counter()
    try:
        out = fn()
        print(f"[OK]   {name} ({time.perf_counter() - t0:.2f} s){' -> ' + str(out) if out else ''}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[FALLO] {name}: {type(e).__name__}: {e}")
        return False


def main() -> int:
    import torch
    import torchvision
    print("Python      :", sys.version.split()[0])
    print("torch       :", torch.__version__, "| torchvision:", torchvision.__version__)
    print("CUDA (torch):", torch.version.cuda, "| cuDNN:", torch.backends.cudnn.version())
    if not torch.cuda.is_available():
        print("CUDA NO disponible: torch no ve la GPU (¿se instaló la versión CPU o falta el driver?).")
        return 1
    p = torch.cuda.get_device_properties(0)
    print(f"GPU         : {p.name} | capacidad {p.major}.{p.minor} | {p.total_memory / 2**30:.1f} GiB | "
          f"{p.multi_processor_count} SM")
    print("Arquitecturas compiladas en esta build:", torch.cuda.get_arch_list())
    ok = True
    dev = "cuda"

    def matmul():
        x = torch.randn(2048, 2048, device=dev)
        y = x @ x
        torch.cuda.synchronize()
        return f"suma={y.sum().item():.1f}"

    def amp():
        x = torch.randn(1024, 1024, device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            y = x @ x
        torch.cuda.synchronize()
        return f"dtype={y.dtype}"

    def conv_backward():
        c = torch.nn.Conv2d(3, 8, 3, padding=1).to(dev)
        c(torch.randn(2, 3, 64, 64, device=dev)).sum().backward()
        torch.cuda.synchronize()

    def tv_ops():
        from torchvision.ops import nms, roi_align
        b = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [50., 50., 60., 60.]], device=dev)
        keep = nms(b, torch.tensor([0.9, 0.8, 0.7], device=dev), 0.5)
        r = roi_align(torch.randn(1, 3, 32, 32, device=dev), [b[:1]], output_size=4)
        torch.cuda.synchronize()
        return f"nms conserva {keep.tolist()}, roi_align {tuple(r.shape)}"

    def frcnn_step():
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        torch.cuda.reset_peak_memory_stats()
        m = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2,
                                    min_size=512, max_size=512).to(dev).train()
        imgs = [torch.rand(3, 512, 512, device=dev) for _ in range(2)]
        tg = [{"boxes": torch.tensor([[10., 10., 100., 100.]], device=dev), "labels": torch.tensor([1], device=dev)}
              for _ in range(2)]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = sum(m(imgs, tg).values())
        loss.backward()
        torch.cuda.synchronize()
        return f"pérdida={loss.item():.3f}, pico de memoria={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"

    for name, fn in [("matmul en la GPU", matmul), ("autocast fp16", amp), ("convolución + backward", conv_backward),
                     ("operadores CUDA de torchvision (nms, roi_align)", tv_ops),
                     ("Faster R-CNN R50-FPN: un paso con AMP (512 px, lote 2, pesos aleatorios)", frcnn_step)]:
        ok &= stage(name, fn)
    print("\nRESULTADO:", "todo correcto" if ok else "hay fallos: pégame la salida completa")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())