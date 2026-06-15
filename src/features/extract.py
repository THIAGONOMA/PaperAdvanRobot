"""Integração da extração de features com as amostras (com cache do extrator)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..config import CFG
from ..data.types import Sample
from .blendshapes import BlendshapeExtractor


@lru_cache(maxsize=1)
def get_extractor() -> BlendshapeExtractor:
    return BlendshapeExtractor()


def blendshapes_for(sample: Sample) -> Optional[dict[str, float]]:
    """Devolve os blendshapes da amostra (extrai do vídeo se necessário)."""
    if sample.blendshapes:
        return sample.blendshapes
    if sample.video_path and sample.frame_time is not None:
        large = sample.dataset == "omg"
        return get_extractor().from_video_frame(
            sample.video_path, sample.frame_time,
            large_frame=large, listener_side=CFG.data.omg_listener_side,
        )
    return None


def blendshapes_sequence_for(
    sample: Sample, n_frames: Optional[int] = None
) -> list[dict[str, float]]:
    """Extrai blendshapes de N frames ao longo da janela temporal da amostra.

    Captura a dinâmica da expressão (não só o instante central). Retorna apenas
    os frames em que um rosto foi detectado.
    """
    import numpy as np

    n = n_frames or CFG.data.n_frames_per_window
    if not (sample.video_path and sample.start_time is not None
            and sample.end_time is not None and sample.end_time > sample.start_time):
        coefs = blendshapes_for(sample)
        return [coefs] if coefs else []

    extractor = get_extractor()
    large = sample.dataset == "omg"
    side = CFG.data.omg_listener_side
    times = np.linspace(sample.start_time, sample.end_time, num=n, endpoint=False)
    seq: list[dict[str, float]] = []
    for t in times:
        coefs = extractor.from_video_frame(
            sample.video_path, float(t), large_frame=large, listener_side=side
        )
        if coefs:
            seq.append(coefs)
    return seq


def keyframe_for(sample: Sample, cache_dir: str = "data/keyframes") -> str:
    """Garante um keyframe JPEG para a amostra (condição C3) e retorna o caminho."""
    if sample.image_path and Path(sample.image_path).exists():
        return sample.image_path
    if not (sample.video_path and sample.frame_time is not None):
        raise ValueError(f"Amostra {sample.sample_id} sem vídeo/instante para keyframe.")
    out = Path(cache_dir) / f"{sample.sample_id}.jpg"
    side = CFG.data.omg_listener_side if sample.dataset == "omg" else None
    return get_extractor().keyframe(
        sample.video_path, sample.frame_time, out, listener_side=side
    )


def keyframes_for(
    sample: Sample, n_frames: Optional[int] = None, cache_dir: str = "data/keyframes"
) -> list[str]:
    """Extrai N keyframes ao longo da janela (condição C3 multi-imagem)."""
    import numpy as np

    n = n_frames or CFG.data.c3_n_frames
    if not (sample.video_path and sample.start_time is not None
            and sample.end_time is not None and sample.end_time > sample.start_time):
        return [keyframe_for(sample, cache_dir)]

    side = CFG.data.omg_listener_side if sample.dataset == "omg" else None
    extractor = get_extractor()
    # instantes nos centros de N sub-janelas iguais
    edges = np.linspace(sample.start_time, sample.end_time, num=n + 1)
    times = [(edges[i] + edges[i + 1]) / 2 for i in range(n)]
    paths: list[str] = []
    for i, t in enumerate(times):
        out = Path(cache_dir) / f"{sample.sample_id}_f{i}.jpg"
        try:
            paths.append(extractor.keyframe(sample.video_path, float(t), out, listener_side=side))
        except Exception:
            continue
    return paths or [keyframe_for(sample, cache_dir)]
