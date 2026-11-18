"""DETR: en vez de crear 300 consultas al azar, parte de las 100 consultas preentrenadas (repetidas + ruido pequeño).
Idempotente. Ejecutar desde la raíz del proyecto."""
from pathlib import Path
import os

P = Path("src/models/hf_detectors.py")
s = P.read_text(encoding="utf-8")
if "_expand_queries" in s:
    print("[=] ya aplicado"); raise SystemExit

HELPER = '''def _expand_queries(hf, n: int, noise: float = 0.1) -> None:
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


'''
OLD = '''        hf = DetrForObjectDetection.from_pretrained(
            CHECKPOINTS["detr"], num_labels=1, num_queries=num_queries, ignore_mismatched_sizes=True,
            auxiliary_loss=True)'''
NEW = '''        hf = DetrForObjectDetection.from_pretrained(
            CHECKPOINTS["detr"], num_labels=1, ignore_mismatched_sizes=True, auxiliary_loss=True)
        _expand_queries(hf, num_queries)                                   # parte de las 100 consultas preentrenadas'''
if s.count(OLD) != 1 or s.count("def build_hf_detector(") != 1:
    raise SystemExit("[!] El texto esperado no coincide; no se toca el archivo")
s = s.replace(OLD, NEW).replace("def build_hf_detector(", HELPER + "def build_hf_detector(")
tmp = P.with_suffix(".tmp"); tmp.write_text(s, encoding="utf-8"); os.replace(tmp, P)
print("[+] src/models/hf_detectors.py actualizado")
