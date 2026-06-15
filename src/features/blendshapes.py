"""Extração dos 52 blendshapes faciais via MediaPipe Face Landmarker.

Aplicável apenas a datasets com vídeo bruto (OMG-Empathy). Os imports do
MediaPipe/OpenCV são preguiçosos para não impor essas dependências pesadas ao
restante do pipeline.

Requer Python <= 3.12 (limite atual do mediapipe).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from . import BLENDSHAPE_NAMES

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_DEFAULT_MODEL = "models/face_landmarker.task"


class BlendshapeExtractor:
    """Wrapper sobre o MediaPipe Face Landmarker (modo IMAGE)."""

    def __init__(self, model_path: str = _DEFAULT_MODEL, auto_download: bool = True) -> None:
        self.model_path = Path(model_path)
        self.auto_download = auto_download
        self._landmarker = None  # inicialização preguiçosa
        self._last_box: Optional[tuple[int, int, int]] = None  # (x, y, win) p/ cache

    def _ensure_model(self) -> None:
        if self.model_path.exists():
            return
        if not self.auto_download:
            raise FileNotFoundError(
                f"Modelo não encontrado em {self.model_path}. "
                "Baixe o face_landmarker.task ou use auto_download=True."
            )
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, self.model_path)

    def _ensure_loaded(self) -> None:
        if self._landmarker is not None:
            return
        self._ensure_model()
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
            output_face_blendshapes=True,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    # --- API pública -------------------------------------------------------

    def from_image(self, image_path: str | Path) -> Optional[dict[str, float]]:
        """Retorna {nome_blendshape: score} para a primeira face do arquivo."""
        self._ensure_loaded()
        import mediapipe as mp

        image = mp.Image.create_from_file(str(image_path))
        return self._detect(image)

    def from_rgb_array(self, rgb) -> Optional[dict[str, float]]:
        """Extrai blendshapes de um frame RGB (numpy array HxWx3)."""
        self._ensure_loaded()
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._detect(image)

    def from_video_frame(
        self,
        video_path: str | Path,
        t_seconds: float,
        large_frame: bool = True,
        listener_side: str = "right",
    ) -> Optional[dict[str, float]]:
        """Lê o frame do vídeo no instante t e extrai os blendshapes.

        Para o OMG (frames split-screen com rostos pequenos e distantes), usa
        `from_large_frame`, que recorta a metade do ouvinte e faz busca em
        janelas para localizar o rosto.
        """
        rgb = _read_rgb_frame(video_path, t_seconds)
        if rgb is None:
            return None
        if large_frame:
            return self.from_large_frame(rgb, listener_side)
        return self.from_rgb_array(rgb)

    def from_large_frame(
        self, rgb, listener_side: str = "right"
    ) -> Optional[dict[str, float]]:
        """Detecta rosto em frame grande/split-screen via recorte + busca em janelas.

        O detector do MediaPipe reduz a entrada para 192px; em frames largos o
        rosto fica pequeno demais. Recortamos a metade do ouvinte e varremos em
        janelas onde o rosto ocupa fração suficiente para ser detectado.
        """
        self._ensure_loaded()
        import mediapipe as mp

        half = crop_listener_half(rgb, listener_side)
        h, w = half.shape[:2]
        win = int(min(h, w) / 1.5)

        def detect_at(x: int, y: int) -> Optional[dict[str, float]]:
            tile = np.ascontiguousarray(half[y:y + win, x:x + win])
            return self._detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=tile))

        # 1) tenta o recorte que funcionou da última vez (rosto fica ~parado)
        if self._last_box is not None:
            lx, ly, lwin = self._last_box
            if lwin == win and 0 <= lx <= w - win and 0 <= ly <= h - win:
                coefs = detect_at(lx, ly)
                if coefs:
                    return coefs

        # 2) varredura em janelas
        step = max(1, win // 2)
        ys = list(range(0, max(1, h - win + 1), step))
        xs = list(range(0, max(1, w - win + 1), step))
        if not ys or ys[-1] != h - win:
            ys.append(max(0, h - win))
        if not xs or xs[-1] != w - win:
            xs.append(max(0, w - win))

        for y in ys:
            for x in xs:
                coefs = detect_at(x, y)
                if coefs:
                    self._last_box = (x, y, win)
                    return coefs

        # 3) fallback: metade inteira
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(half))
        return self._detect(image)

    def keyframe(
        self,
        video_path: str | Path,
        t_seconds: float,
        out_path: str | Path,
        listener_side: Optional[str] = None,
    ) -> str:
        """Extrai um keyframe do vídeo no instante t e salva como JPEG (condição C3).

        Se `listener_side` for dado e o frame for split-screen, recorta a metade
        do ouvinte (para o Gemma ver apenas o sujeito anotado).
        """
        import cv2

        rgb = _read_rgb_frame(video_path, t_seconds)
        if rgb is None:
            raise RuntimeError(f"Falha ao ler frame de {video_path} em t={t_seconds}s.")
        if listener_side:
            rgb = crop_listener_half(rgb, listener_side)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return str(out_path)

    # --- interno -----------------------------------------------------------

    def _detect(self, image) -> Optional[dict[str, float]]:
        result = self._landmarker.detect(image)
        if not result.face_blendshapes:
            return None
        coefs = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        # garante a presença das 52 chaves canônicas (0.0 quando ausente)
        return {name: coefs.get(name, 0.0) for name in BLENDSHAPE_NAMES}


def crop_listener_half(rgb, listener_side: str = "right"):
    """Recorta a metade do ouvinte em frames split-screen (largura >> altura).

    Frames quadrados/normais são retornados sem alteração.
    """
    h, w = rgb.shape[:2]
    if w <= 1.5 * h:
        return rgb
    mid = w // 2
    return rgb[:, mid:] if listener_side == "right" else rgb[:, :mid]


def _read_rgb_frame(video_path: str | Path, t_seconds: float):
    """Lê um frame do vídeo no instante dado e retorna em RGB (ou None)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_seconds) * 1000.0)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
