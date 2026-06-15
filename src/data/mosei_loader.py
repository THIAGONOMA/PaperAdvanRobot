"""Loader do CMU-MOSEI (versão Kaggle em arquivos .csd / HDF5).

ATENÇÃO: o MOSEI NÃO disponibiliza vídeo bruto (apenas features pré-extraídas).
Portanto a condição C3 (multimodal/imagem) não se aplica. Mapeamento:

    C1 (texto)  -> transcrição reconstruída de CMU_MOSEI_TimestampedWords.csd
    C2 (facial) -> features FACET (CMU_MOSEI_VisualFacet42.csd, 35-dim)
    C3 (vídeo)  -> indisponível

Estrutura interna dos .csd (HDF5):

    arquivo
    └── <root da computational sequence>   (ex.: 'All Labels', 'words', 'FACET 4.2')
        ├── data
        │   └── {video_id}
        │       ├── features   (n_segmentos|n_frames|n_palavras, dims)
        │       └── intervals  (mesma 1a dim, 2)  -> [start, end] em segundos
        └── metadata

Um segmento de rótulo é (video_id, índice). A transcrição (C1) e o vetor FACET
(C2) são alinhados por sobreposição de intervalos de tempo com o segmento.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np

from ..config import CFG
from .types import GroundTruth, Sample

logger = logging.getLogger(__name__)

LABEL_NAMES = ["sentiment", "happiness", "sadness", "anger", "surprise", "disgust", "fear"]
EKMAN = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]

CSD_FILES = {
    "labels": "labels/CMU_MOSEI_Labels.csd",
    "words": "languages/CMU_MOSEI_TimestampedWords.csd",
    "glove": "languages/CMU_MOSEI_TimestampedWordVectors.csd",
    "covarep": "acoustics/CMU_MOSEI_COVAREP.csd",
    "facet": "visuals/CMU_MOSEI_VisualFacet42.csd",
}

# Tokens de pausa/silêncio do alinhamento de palavras a serem ignorados.
_SKIP_WORDS = {"sp", "sil", ""}


def _mosei_root() -> Path:
    root = Path(CFG.data.mosei_path)
    candidates = [root / "CMU-MOSEI", root]
    for c in candidates:
        if (c / "labels").is_dir():
            return c
    raise FileNotFoundError(
        f"Diretório do MOSEI não encontrado. Verifique data.mosei_path em config.yaml. "
        f"Tentei: {candidates}"
    )


def _csd_path(key: str) -> Path:
    return _mosei_root() / CSD_FILES[key]


def _open_data(path: Path) -> tuple[h5py.File, h5py.Group]:
    """Abre um .csd e retorna (handle, grupo 'data' = {video_id: {features, intervals}})."""
    f = h5py.File(str(path), "r")
    root_name = list(f.keys())[0]
    return f, f[root_name]["data"]


def _decode_word(val) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace").strip()
    return str(val).strip()


def _overlap_mask(intervals: np.ndarray, start: float, end: float) -> np.ndarray:
    """Linhas cujos intervalos [s,e] sobrepõem [start, end]."""
    return (intervals[:, 1] > start) & (intervals[:, 0] < end)


# ---------------------------------------------------------------------------
# Splits oficiais — partição por video_id (segmentos do mesmo vídeo no mesmo
# split). Usa folds do SDK se instalado; senão hash determinístico (~70/8/22).
# ---------------------------------------------------------------------------

def _split_for_video(video_id: str) -> str:
    h = int(hashlib.md5(video_id.encode()).hexdigest(), 16) % 100
    if h < 71:
        return "train"
    if h < 79:
        return "validation"
    return "test"


def _try_sdk_splits() -> Optional[dict[str, set[str]]]:
    try:
        from mmsdk import mmdatasdk
        folds = mmdatasdk.cmu_mosei.standard_folds
        return {
            "train": set(folds.standard_train_fold),
            "validation": set(folds.standard_valid_fold),
            "test": set(folds.standard_test_fold),
        }
    except Exception:
        return None


def split_for_video(video_id: str, sdk_folds: Optional[dict[str, set[str]]]) -> str:
    if sdk_folds:
        for split_name, vid_set in sdk_folds.items():
            if video_id in vid_set:
                return split_name
        return "train"
    return _split_for_video(video_id)


# ---------------------------------------------------------------------------
# Conversão de rótulos
# ---------------------------------------------------------------------------

def label_vector_to_ground_truth(vec: np.ndarray) -> GroundTruth:
    """Converte o vetor de 7 dimensões do Labels.csd em GroundTruth."""
    scores = {name: float(vec[i]) for i, name in enumerate(LABEL_NAMES) if i < len(vec)}
    emotion_scores = {emo: scores.get(emo, 0.0) for emo in EKMAN}
    emotions = [emo for emo in EKMAN if emotion_scores[emo] > 0.0] or ["neutral"]
    return GroundTruth(
        valence=None,
        emotions=emotions,
        sentiment=scores.get("sentiment"),
        emotion_scores=emotion_scores,
    )


# ---------------------------------------------------------------------------
# Iteração de amostras (alinhamento por intervalo de tempo)
# ---------------------------------------------------------------------------

def iter_samples(split: str) -> Iterator[Sample]:
    """Itera segmentos do MOSEI para o split dado.

    Cada segmento de rótulo (video_id, índice) define um intervalo de tempo.
    A transcrição (C1) e o vetor FACET (C2) são alinhados por sobreposição de
    intervalos com esse segmento.
    """
    sdk_folds = _try_sdk_splits()

    lf, labels_data = _open_data(_csd_path("labels"))
    wf = wd = ff = fd = None
    try:
        if _csd_path("words").exists():
            wf, wd = _open_data(_csd_path("words"))
        if _csd_path("facet").exists():
            ff, fd = _open_data(_csd_path("facet"))

        for video_id in labels_data.keys():
            if split_for_video(video_id, sdk_folds) != split:
                continue

            label_feats = np.asarray(labels_data[video_id]["features"])
            label_ivals = np.asarray(labels_data[video_id]["intervals"])

            w_feats = w_ivals = None
            if wd is not None and video_id in wd:
                w_feats = np.asarray(wd[video_id]["features"]).ravel()
                w_ivals = np.asarray(wd[video_id]["intervals"])
            f_feats = f_ivals = None
            if fd is not None and video_id in fd:
                f_feats = np.asarray(fd[video_id]["features"])
                f_ivals = np.asarray(fd[video_id]["intervals"])

            for seg_idx in range(label_feats.shape[0]):
                start = float(label_ivals[seg_idx][0])
                end = float(label_ivals[seg_idx][1])
                gt = label_vector_to_ground_truth(label_feats[seg_idx])

                transcript = None
                if w_feats is not None:
                    mask = _overlap_mask(w_ivals, start, end)
                    words = [_decode_word(w) for w in w_feats[mask]]
                    words = [w for w in words if w.lower() not in _SKIP_WORDS]
                    transcript = " ".join(words) or None

                facet = None
                if f_feats is not None:
                    rows = f_feats[_overlap_mask(f_ivals, start, end)]
                    if len(rows):
                        facet = np.mean(rows, axis=0).tolist()

                yield Sample(
                    sample_id=f"{video_id}[{seg_idx}]",
                    dataset="mosei",
                    split=split,
                    ground_truth=gt,
                    start_time=start,
                    end_time=end,
                    transcript=transcript,
                    facet=facet,
                )
    finally:
        for h in (lf, wf, ff):
            if h is not None:
                h.close()
