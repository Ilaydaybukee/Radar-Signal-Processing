"""SAR iş akışının ortak, güvenli yardımcıları."""
from __future__ import annotations
import hashlib, json, random, re
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CLASSES = ["Fishing", "Sailing", "Tanker", "Passenger", "Pleasure"]

def seed_everything(seed: int = SEED) -> None:
    """Python, NumPy ve varsa PyTorch üreteçlerini tekrarlanabilir yapar."""
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    except ImportError: pass

def config() -> dict:
    with open(ROOT / "configs/config.yaml", encoding="utf-8") as f: return yaml.safe_load(f)

def tiffs() -> list[Path]:
    """Proje altında, sanal ortam ve üretilen sonuçlar hariç ham TIFF'leri bulur."""
    ignored = {".venv", "venv", ".git", "processed", "splits"}
    return sorted(p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in {".tif", ".tiff"} and not ignored.intersection(p.parts))

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def label_from_name(name: str) -> str | None:
    low=name.lower()
    for c in CLASSES:
        if c.lower() in low: return c
    if re.search(r"(^|[_\- ])sea([_\- .]|$)", low): return "Sea"
    if re.search(r"(^|[_\- ])ship([_\- .]|$)", low): return "Ship"
    return None

def decide_task(labels) -> tuple[str | None,list[str],str]:
    labels={str(x) for x in labels if pd.notna(x) and str(x)}
    if {"Ship","Sea"}.issubset(labels): return "binary_ship_sea", ["Sea","Ship"], "Ship ve Sea etiketleri bulundu."
    ships=[c for c in CLASSES if c in labels]
    if len(ships)>=2: return "multiclass_ship_type", ships, "En az iki gerçek gemi türü bulundu."
    return None, sorted(labels), "Eğitim için en az iki doğrulanabilir sınıf yok."

def read_gray(path) -> np.ndarray:
    a=np.asarray(tifffile.imread(path))
    a=np.squeeze(a)
    if a.ndim==3: a=np.mean(a.astype(np.float32), axis=-1 if a.shape[-1] <= 4 else 0)
    if a.ndim != 2: raise ValueError(f"Desteklenmeyen TIFF boyutu: {a.shape}")
    return a

def normalize(a: np.ndarray, low=1, high=99) -> np.ndarray:
    """Aykırı uçları yüzdeliklerle kırpar; float32 [0,1] üretir ve NaN üretmez."""
    x=np.nan_to_num(a.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo,hi=np.percentile(x,[low,high])
    if hi <= lo: return np.zeros_like(x, dtype=np.float32)
    return np.clip((x-lo)/(hi-lo),0,1).astype(np.float32)

def load_manifest() -> pd.DataFrame:
    p=ROOT/"results/tables/split_manifest.csv"
    if not p.exists(): raise FileNotFoundError("Önce 04_prepare_splits.py çalıştırılmalıdır.")
    return pd.read_csv(p)

def write_problem(reason: str) -> None:
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"results/dataset_problem_report.md").write_text("# Veri Seti Problem Raporu\n\n"+reason+"\n\nOlmayan sınıflar üretilmedi; eğitim durduruldu.\n",encoding="utf-8")
