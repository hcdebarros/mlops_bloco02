"""Novas características de imagem para monitoramento (extensão de `image_extract.py`).

Baseline da aula 011 (lgallindo/bentoML-processamento-imagens):
    luminance_mean  -> média do canal L do Pillow, calculada pelo histograma
    média de R, G, B -> idem, a partir dos histogramas R/G/B

Características novas:
    1. contrast_rms        desvio-padrão do canal L (pelo histograma)          numérica
    2. colorfulness        índice de Hasler & Süsstrunk (2003)                 numérica
       colorfulness_class  faixa do artigo ("nada colorida" ... "extremamente") categórica
    3. sharpness_lapvar    variância do Laplaciano sobre o canal L             numérica
       is_blurry           sharpness_lapvar < BLUR_THRESHOLD                    categórica

Dependências: Pillow + NumPy (o NumPy já vem como dependência do BentoML).

Uso:
    python extract_features.py images/*.webp
    python extract_features.py images/*.webp --corpus corpus_beaches   # compara com o corpus
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

FEATURES_DIR = Path("features")
CHANNELS = ("L", "R", "G", "B")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

BLUR_THRESHOLD = 100.0  # limiar empírico usual; calibrar com o corpus real

# Faixas de Hasler & Süsstrunk (2003) — limite superior (exclusivo) de cada classe
COLORFULNESS_LEVELS = (
    (15.0, "nada colorida"),
    (33.0, "pouco colorida"),
    (45.0, "moderadamente colorida"),
    (59.0, "medianamente colorida"),
    (82.0, "bastante colorida"),
    (109.0, "muito colorida"),
    (float("inf"), "extremamente colorida"),
)

# Laplaciano 4-vizinhos: [[0,1,0],[1,-4,1],[0,1,0]]


# --------------------------------------------------------------------------
# Auxiliares: mesmo critério da aula (canal L = image.convert("L") do Pillow,
# que usa L = 0.299 R + 0.587 G + 0.114 B — ITU-R BT.601)
# --------------------------------------------------------------------------
def channel_histogram(image: PILImage, channel: str) -> list[int]:
    if channel == "L":
        return image.convert("L").histogram()
    return image.convert("RGB").getchannel(channel).histogram()


def histogram_mean(counts: Sequence[int]) -> float:
    total = sum(counts)
    return sum(i * c for i, c in enumerate(counts)) / total


def histogram_std(counts: Sequence[int]) -> float:
    total = sum(counts)
    mu = histogram_mean(counts)
    var = sum(c * (i - mu) ** 2 for i, c in enumerate(counts)) / total
    return var ** 0.5


# --------------------------------------------------------------------------
# Baseline (aula 011)
# --------------------------------------------------------------------------
def baseline_features(image: PILImage) -> dict:
    return {
        f"{ch.lower()}_mean" if ch != "L" else "luminance_mean":
            histogram_mean(channel_histogram(image, ch))
        for ch in CHANNELS
    }


# --------------------------------------------------------------------------
# Característica 1 — contraste RMS
# --------------------------------------------------------------------------
def contrast_rms(image: PILImage) -> float:
    """Desvio-padrão do canal L (0 = imagem uniforme; ~128 = só preto e branco)."""
    return histogram_std(channel_histogram(image, "L"))


# --------------------------------------------------------------------------
# Característica 2 — colorfulness (Hasler & Süsstrunk, 2003)
# --------------------------------------------------------------------------
def colorfulness(image: PILImage) -> float:
    """
    rg = R - G ;  yb = (R + G)/2 - B
    C  = sqrt(σ_rg² + σ_yb²) + 0.3 · sqrt(μ_rg² + μ_yb²)
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def colorfulness_class(value: float) -> str:
    return next(label for limit, label in COLORFULNESS_LEVELS if value < limit)


# --------------------------------------------------------------------------
# Característica 3 — nitidez (variância do Laplaciano)
# --------------------------------------------------------------------------
def sharpness_lapvar(image: PILImage) -> float:
    """Variância da resposta do Laplaciano sobre L. Baixa => poucas bordas => borrada."""
    y = np.asarray(image.convert("L"), dtype=np.float64)
    p = np.pad(y, 1, mode="edge")
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * y
    return float(lap.var())


# --------------------------------------------------------------------------
# Extração completa
# --------------------------------------------------------------------------
def extract_extra_features(image: PILImage) -> dict:
    c = colorfulness(image)
    s = sharpness_lapvar(image)
    return {
        "contrast_rms": contrast_rms(image),
        "colorfulness": c,
        "colorfulness_class": colorfulness_class(c),
        "sharpness_lapvar": s,
        "is_blurry": s < BLUR_THRESHOLD,
    }


def extract_all(image_path: Path) -> dict:
    with Image.open(image_path) as image:
        image.load()
        row = {"image": image_path.name, "width": image.width, "height": image.height}
        row |= baseline_features(image)
        row |= extract_extra_features(image)
    return row


def save_features(row: dict, features_dir: Path = FEATURES_DIR) -> Path:
    """Grava features/<stem>.json, no mesmo espírito da Camada 1 da aula."""
    features_dir.mkdir(parents=True, exist_ok=True)
    out = features_dir / f"{Path(row['image']).stem}.json"
    out.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# Referência do corpus: percentis de cada característica numérica
# --------------------------------------------------------------------------
NUMERIC = ("luminance_mean", "contrast_rms", "colorfulness", "sharpness_lapvar")


def corpus_reference(rows: list[dict]) -> dict:
    ref = {"n_images": len(rows)}
    for key in NUMERIC:
        v = np.array([r[key] for r in rows])
        ref[key] = {"mean": v.mean(), "p05": np.percentile(v, 5),
                    "p50": np.percentile(v, 50), "p95": np.percentile(v, 95)}
    ref["is_blurry_rate"] = float(np.mean([r["is_blurry"] for r in rows]))
    classes: dict[str, int] = {}
    for r in rows:
        classes[r["colorfulness_class"]] = classes.get(r["colorfulness_class"], 0) + 1
    ref["colorfulness_class_counts"] = classes
    return ref


def flag_vs_corpus(row: dict, ref: dict) -> dict:
    """Marca 'baixo'/'alto' quando a característica sai do intervalo [p05, p95] do corpus."""
    flags = {}
    for key in NUMERIC:
        if row[key] < ref[key]["p05"]:
            flags[key] = "baixo"
        elif row[key] > ref[key]["p95"]:
            flags[key] = "alto"
        else:
            flags[key] = "ok"
    return flags


def _round(obj):
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 2)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--corpus", type=Path, help="pasta com o corpus de referência")
    ap.add_argument("--csv", type=Path, default=Path("results.csv"))
    args = ap.parse_args(argv)

    rows = [_round(extract_all(p)) for p in args.images]
    for row in rows:
        save_features(row)

    if args.corpus:
        corpus_rows = [extract_all(p) for p in _list_images(args.corpus)]
        ref = _round(corpus_reference(corpus_rows))
        Path("corpus_reference.json").write_text(
            json.dumps(ref, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("corpus:", json.dumps(ref, ensure_ascii=False, indent=2))
        for row in rows:
            row["vs_corpus"] = flag_vs_corpus(row, ref)

    with args.csv.open("w", newline="", encoding="utf-8") as f:
        flat = [{**{k: v for k, v in r.items() if k != "vs_corpus"},
                 **{f"vs_corpus_{k}": v for k, v in r.get("vs_corpus", {}).items()}}
                for r in rows]
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
