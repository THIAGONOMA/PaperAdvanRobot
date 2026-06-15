"""Loader do OMG-Empathy: vídeos diádicos + anotação contínua de valência.

Estrutura esperada da base (download "full"):

    OMG_Empathy2019_full_fY4m3eyn/
    ├── OMG_Empathy2019/
    │   ├── Training/{Videos,Annotations}      # stories 2,4,5,8
    │   └── Validation/{Videos,Annotations}    # story 1
    ├── OMG_Empathy2019_testSet/Videos         # stories 3,6,7
    └── Annotations/                           # anotações do test set

Cada anotação é um CSV com header `valence` e um valor por frame (faixa -1..1).
As amostras são geradas por janela temporal, com a valência média da janela
como rótulo. A métrica oficial é o CCC.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from ..config import CFG
from .types import GroundTruth, Sample

DEFAULT_FPS = 25.0

# Diretórios (relativos à raiz da base) por split: (videos, annotations).
_SPLIT_DIRS: dict[str, tuple[str, str]] = {
    "train": ("OMG_Empathy2019/Training/Videos", "OMG_Empathy2019/Training/Annotations"),
    "validation": ("OMG_Empathy2019/Validation/Videos", "OMG_Empathy2019/Validation/Annotations"),
    "test": ("OMG_Empathy2019_testSet/Videos", "Annotations"),
}


def _root() -> Path:
    return Path(CFG.data.omg_path)


def _video_fps(video_path: Path, default: float = DEFAULT_FPS) -> float:
    """Lê o FPS do vídeo via OpenCV; usa o default se indisponível."""
    try:
        import cv2  # importação preguiçosa (opcional)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps and fps > 0 else default
    except Exception:
        return default


def load_valence_annotation(csv_path: Path) -> list[float]:
    """Lê a anotação contínua de valência (um valor por frame)."""
    values: list[float] = []
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            cell = row[0].strip()
            if not cell or cell.lower() == "valence":
                continue
            try:
                values.append(float(cell))
            except ValueError:
                continue
    return values


def iter_video_pairs(split: str) -> Iterator[tuple[Path, Path]]:
    """Itera pares (vídeo, anotação) do split, casando pelo nome do arquivo."""
    if split not in _SPLIT_DIRS:
        raise ValueError(f"Split inválido: {split!r}. Use {list(_SPLIT_DIRS)}.")
    root = _root()
    videos_dir, ann_dir = (root / d for d in _SPLIT_DIRS[split])
    for video in sorted(videos_dir.glob("*.mp4")):
        ann = ann_dir / f"{video.stem}.csv"
        if ann.exists():
            yield video, ann


def iter_samples(split: str, window_s: float | None = None) -> Iterator[Sample]:
    """Itera amostras do OMG por janela temporal.

    Args:
        split: 'train' | 'validation' | 'test'.
        window_s: tamanho da janela em segundos (default = config).

    Yields:
        Sample com caminho de vídeo, instante representativo e valência média.
    """
    window_s = window_s or CFG.data.window_s
    for video, ann in iter_video_pairs(split):
        valence = load_valence_annotation(ann)
        if not valence:
            continue
        fps = _video_fps(video)
        win = max(1, int(round(window_s * fps)))
        n = len(valence)
        for w_idx, start in enumerate(range(0, n, win)):
            chunk = valence[start:start + win]
            if not chunk:
                continue
            start_t = start / fps
            end_t = min(start + win, n) / fps
            yield Sample(
                sample_id=f"{video.stem}_w{w_idx:04d}",
                dataset="omg",
                split=split,
                ground_truth=GroundTruth(valence=sum(chunk) / len(chunk)),
                start_time=start_t,
                end_time=end_t,
                video_path=str(video),
                frame_time=(start_t + end_t) / 2.0,
            )
