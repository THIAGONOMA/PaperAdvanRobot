"""Montagem das mensagens do LLM por condição de entrada (C1/C2/C3) e tarefa."""
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .state import GraphState

# --- System prompts por tarefa ---------------------------------------------

# Âncoras de calibração da valência, compartilhadas pelas duas tarefas.
_VALENCE_CALIB = (
    "Estimate VALENCE on an INTEGER scale from -100 to +100 (how negative vs. "
    "positive the person feels) and AROUSAL from -100 to +100 (calm vs. excited). "
    "Valence calibration anchors: -100 extremely negative/distressed; -60 clearly "
    "negative; -25 mildly negative; 0 strictly neutral; +25 mildly positive; "
    "+60 clearly positive; +100 extremely positive/joyful. "
    "Be DISCRIMINATIVE and use the FULL range: map even subtle facial cues to a "
    "specific value (e.g. -37, +18, +52) instead of defaulting to 0 or round "
    "multiples of ten. Reserve values within +/-12 only for genuinely "
    "neutral/ambiguous expressions. "
    "Respond ONLY with the requested JSON, no extra text."
)

# Valência (OMG): foco direto na dimensão afetiva, sem decomposição de Ekman.
_COMMON_VALENCE = "You are an expert affective-computing annotator. " + _VALENCE_CALIB

# Emoção (MOSEI): a decomposição de Ekman É a tarefa.
_COMMON_EMOTION = (
    "You are an expert affective-computing annotator. "
    "First rate the intensity (0-100) of EACH of the six basic Ekman emotions "
    "(happiness, sadness, anger, fear, disgust, surprise) for the target person. "
    "Then " + _VALENCE_CALIB
)

_JSON_VALENCE = (
    ' Output ONLY a JSON object (no markdown, no extra text) with EXACTLY these keys, '
    'all values INTEGERS: '
    '{"valence": <-100..100>, "arousal": <-100..100>, "confidence": <0-100>}.'
)

_JSON_EMOTION = (
    ' Output ONLY a JSON object (no markdown, no extra text) with EXACTLY these keys, '
    'all values INTEGERS: '
    '{"ekman": {"happiness": <0-100>, "sadness": <0-100>, "anger": <0-100>, '
    '"fear": <0-100>, "disgust": <0-100>, "surprise": <0-100>}, '
    '"emotion_label": "<happiness|sadness|anger|fear|disgust|surprise|neutral>", '
    '"valence": <-100..100>, "arousal": <-100..100>, "confidence": <0-100>}.'
)

SYSTEM_VALENCE = (
    "TASK: Estimate the affective state of the LISTENER in a dyadic conversation. "
    "The listener is reacting, over a short time window, to a storyteller. "
    + _COMMON_VALENCE
    + _JSON_VALENCE
)

SYSTEM_EMOTION = (
    "TASK: Recognize the expressed emotion of the speaker in a short video segment. "
    + _COMMON_EMOTION
    + " Set 'emotion_label' to the strongest Ekman emotion, or 'neutral' if all "
    "intensities are low."
    + _JSON_EMOTION
)

# Legenda dos blendshapes do MediaPipe -> emoções. Enfatiza que os PICOS (max)
# carregam mais sinal afetivo que a média.
_BLENDSHAPE_LEGEND = (
    "Blendshape meaning guide (weigh PEAK activations 'max' more than 'mean', "
    "since brief peaks carry the strongest affective signal): "
    "mouthSmileLeft/Right, cheekSquintLeft/Right -> happiness (positive valence); "
    "mouthFrownLeft/Right, browDownLeft/Right -> sadness/anger (negative valence); "
    "browInnerUp -> sadness/fear; "
    "eyeWideLeft/Right, jawOpen, browOuterUp -> surprise/fear (high arousal); "
    "noseSneerLeft/Right, mouthUpperUpLeft/Right -> disgust (negative valence); "
    "eyeSquintLeft/Right, mouthClose, mouthPress -> tension/low arousal."
)


def _system_for(state: GraphState) -> str:
    return SYSTEM_VALENCE if state["task"] == "valence" else SYSTEM_EMOTION


def few_shot(state: GraphState) -> list[BaseMessage]:
    """Exemplos in-context (k-shot) a partir do split de treino (se fornecidos).

    Cada exemplo é um par (entrada do usuário, saída JSON esperada), apresentado
    como turno Human seguido de turno AI.
    """
    examples: list[BaseMessage] = []
    for ex in state.get("features", {}).get("few_shot", []):
        examples.append(HumanMessage(content=ex["input"]))
        examples.append(AIMessage(content=ex["output"]))
    return examples


def _user_content(state: GraphState) -> list[dict]:
    f = state["features"]
    cond = state["condition"]
    if cond == "C1":
        return [{
            "type": "text",
            "text": (
                "Transcript of what the person said:\n"
                f'"{f["transcript"]}"\n\n'
                "Infer the emotional state from the text."
            ),
        }]
    if cond == "C2":
        return [{
            "type": "text",
            "text": (
                "Facial blendshape coefficients for the target person. These INPUT "
                "values are on a 0-1 scale (higher = stronger activation); your "
                "OUTPUT scores must follow the scales specified above (0-100 and "
                "-100..100):\n"
                f"{f['blendshapes_text']}\n\n"
                f"{_BLENDSHAPE_LEGEND}\n\n"
                "Infer the emotional state from these facial signals."
            ),
        }]
    # C3 — multimodal (uma ou várias imagens da janela temporal)
    images = f.get("images_b64") or ([f["image_b64"]] if "image_b64" in f else [])
    n = len(images)
    intro = (
        f"{n} sequential frames of the target person's face, sampled across a short "
        "time window. Consider the expression and its changes over the frames, then "
        "infer the overall emotional state."
        if n > 1 else
        "Image of the target person's face. Infer their emotional state from the "
        "visible facial expression."
    )
    content: list[dict] = [{"type": "text", "text": intro}]
    for b64 in images:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


def build_messages(state: GraphState) -> list[BaseMessage]:
    return [SystemMessage(content=_system_for(state)), *few_shot(state),
            HumanMessage(content=_user_content(state))]
