# PaperAdvanRobot — Comparativo de Reconhecimento de Emoção (Baseline por Regras vs. LLM)

Pipeline reprodutível para comparar um método baseado em **blendshapes faciais → valência-arousal por regras** (`face_blendshape`) com um **LLM multimodal (Gemma 4 26B-A4B)** servido por vLLM, em duas bases: **OMG-Empathy** e **CMU-MOSEI**.

- Arquitetura conceitual (C4/UML): [`arch.md`](arch.md)
- Especificação técnica: [`docs/techspec.md`](docs/techspec.md)

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
```

## Modelo (vLLM)

O LLM é acessado via API compatível com OpenAI. Ajuste o endpoint em `config/config.yaml`
ou via variáveis de ambiente `LLM_BASE_URL` / `LLM_MODEL`.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    model="cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
    api_key="not-needed",
    temperature=0,
)
```

## Uso

```bash
# MOSEI, condição texto, few-shot k=3
python -m src.cli run --dataset mosei --condition C1 --k-shots 3

# OMG, condição blendshapes-como-texto
python -m src.cli run --dataset omg --condition C2
```

## Estrutura

```
src/
  config.py        # configuração (config.yaml + env)
  data/            # loaders OMG/MOSEI + splits
  features/        # blendshapes (MediaPipe) + serialização
  baseline/        # face_blendshape: regras -> V-A
  llm/             # schema, prompts, grafo LangGraph, runner
  eval/            # CCC, F1, McNemar, bootstrap
  report/          # tabelas/figuras do paper
```

## Testes

```bash
pytest -q
```

## Condições de entrada do LLM

| Condição | Entrada | Datasets |
|---|---|---|
| C1 | transcrição/texto | MOSEI |
| C2 | blendshapes (ou FACET) serializados em texto | OMG / MOSEI |
| C3 | frames de vídeo (multimodal) | OMG |

> Observação: o MOSEI não disponibiliza vídeo bruto; por isso C3 fica restrito ao OMG e C2 no MOSEI usa as features FACET como proxy.
