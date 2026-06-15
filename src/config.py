"""Carregamento de configuração a partir de config.yaml + variáveis de ambiente."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "config/config.yaml"))


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:8000/v1"
    model: str = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
    api_key: str = "not-needed"
    temperature: float = 0.0
    timeout: int = 120


class RunConfig(BaseModel):
    seeds: list[int] = Field(default_factory=lambda: [13, 42, 1234])
    max_attempts: int = 3
    max_concurrency: int = 8
    conditions: list[str] = Field(default_factory=lambda: ["C1", "C2", "C3"])
    results_dir: str = "results"


class DataConfig(BaseModel):
    omg_path: str = "data/omg/"
    mosei_path: str = "data/mosei/"
    window_s: float = 4.0
    top_n_blendshapes: int = 15
    omg_listener_side: str = "right"
    n_frames_per_window: int = 8
    c3_n_frames: int = 3


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    @property
    def llm_base_url(self) -> str:
        return os.getenv("LLM_BASE_URL", self.llm.base_url)

    @property
    def llm_model(self) -> str:
        return os.getenv("LLM_MODEL", self.llm.model)


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    path = Path(path)
    if not path.exists():
        return Config()
    raw = yaml.safe_load(path.read_text()) or {}
    return Config(**raw)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()


CFG = get_config()
