# Arquitetura — Estudo Comparativo de Reconhecimento de Emoção (Baseline por Regras vs. LLM)

> Paper para a *Advanced Robotics*. Comparação entre o método baseado em **blendshapes faciais → valência-arousal por regras** (`face_blendshape`) e um **LLM multimodal (Gemma 4 26B-A4B)** servido via vLLM.
> Bases de dados: **OMG-Empathy** (com vídeo) e **CMU-MOSEI** (apenas features pré-processadas).

## 1. Visão geral

O objetivo do sistema é executar, de forma reprodutível, o mesmo protocolo de avaliação sobre dois sistemas de predição de emoção e produzir as tabelas/figuras do paper.

Restrições de design que moldam a arquitetura:

- **OMG-Empathy** disponibiliza **vídeo bruto** → permite extrair os 52 blendshapes do MediaPipe e rodar o baseline original. Tarefa = **regressão de valência contínua** (métrica **CCC**).
- **CMU-MOSEI** **não** disponibiliza vídeo bruto (só features FACET/COVAREP/GloVe via SDK) → o pipeline de blendshapes não roda; usa-se modalidade textual/facial pré-extraída. Tarefa = **classificação de emoção** (métrica **F1/acurácia**).
- O **LLM** é acessado por uma **API compatível com OpenAI** exposta pelo vLLM na **porta 80** (modelo `gemma-4-26B-A4B-it-AWQ-4bit`, L4 24GB).
- Comparação justa: em pelo menos uma condição, baseline e LLM recebem **a mesma entrada** (blendshapes serializados em texto — condição C2).

---

## 2. C4 — Nível 1: Contexto

```mermaid
C4Context
    title Contexto do Sistema — Avaliacao Comparativa de Emocao

    Person(pesquisador, "Pesquisador (voce)", "Conduz o experimento do LLM e a avaliacao")
    Person(lea, "Lea", "Responsavel pelo baseline de blendshapes")

    System(plataforma, "Plataforma de Avaliacao Comparativa", "Orquestra extracao de features, inferencia (baseline + LLM) e avaliacao reprodutivel")

    System_Ext(omg, "OMG-Empathy Dataset", "Videos diadicos + anotacao continua de valencia")
    System_Ext(mosei, "CMU-MOSEI (via CMU-MultimodalSDK)", "Features GloVe/COVAREP/FACET + rotulos de emocao/sentimento")
    System_Ext(mediapipe, "MediaPipe Face Landmarker", "478 landmarks + 52 blendshapes")
    System_Ext(vllm, "vLLM Server (porta 80)", "Serve o Gemma 4 26B-A4B (API tipo OpenAI)")

    Rel(pesquisador, plataforma, "Configura e executa experimentos")
    Rel(lea, plataforma, "Fornece o baseline e o split de referencia")
    Rel(plataforma, omg, "Baixa videos + anotacoes")
    Rel(plataforma, mosei, "Carrega features alinhadas")
    Rel(plataforma, mediapipe, "Extrai blendshapes (apenas OMG)")
    Rel(plataforma, vllm, "Envia prompts / imagens, recebe predicoes", "HTTP/JSON")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

---

## 3. C4 — Nível 2: Containers

```mermaid
C4Container
    title Containers — Plataforma de Avaliacao Comparativa

    Person(pesquisador, "Pesquisador", "")

    System_Boundary(plataforma, "Plataforma de Avaliacao Comparativa") {
        Container(ingest, "Ingestao de Dados", "Python", "Download/organizacao do OMG e carga do MOSEI via SDK; gera os splits oficiais")
        Container(features, "Extracao de Features", "Python + MediaPipe", "OMG: 52 blendshapes por frame. MOSEI: usa features FACET/COVAREP/GloVe ja prontas")
        Container(baseline, "Motor Baseline", "Python (regras)", "face_blendshape: blendshapes -> valencia-arousal + emocao categorica")
        Container(llm, "Harness de Inferencia LLM", "Python", "Monta prompts (C1/C2/C3), chama o vLLM, valida e parseia a saida")
        Container(eval, "Avaliacao e Estatistica", "Python", "CCC (OMG), F1/Acuracia (MOSEI), McNemar, bootstrap, latencia")
        ContainerDb(store, "Armazenamento de Resultados", "Parquet/JSON", "Predicoes, metricas, logs de prompt e medicoes de custo")
        Container(report, "Geracao de Relatorio", "Python/Notebook", "Tabelas e figuras do paper")
    }

    System_Ext(vllm, "vLLM Server (porta 80)", "Gemma 4 26B-A4B")
    System_Ext(omg, "OMG-Empathy", "")
    System_Ext(mosei, "CMU-MOSEI / SDK", "")

    Rel(pesquisador, llm, "Define condicoes e prompts")
    Rel(ingest, omg, "Baixa videos + valencia")
    Rel(ingest, mosei, "Carrega features alinhadas")
    Rel(ingest, features, "Fornece amostras e splits")
    Rel(features, baseline, "Blendshapes (OMG) / features (MOSEI)")
    Rel(features, llm, "Representacao de entrada por condicao")
    Rel(baseline, store, "Predicoes V-A / emocao")
    Rel(llm, vllm, "POST /v1/chat/completions", "HTTP/JSON")
    Rel(llm, store, "Predicoes + logs")
    Rel(store, eval, "Predicoes vs. ground truth")
    Rel(eval, report, "Metricas + intervalos de confianca")
```

---

## 4. C4 — Nível 3: Componentes do Harness de Inferência LLM

```mermaid
C4Component
    title Componentes — Harness de Inferencia LLM

    Container_Boundary(llm, "Harness de Inferencia LLM") {
        Component(builder, "Prompt Builder", "Python", "Gera prompt por condicao: C1 (transcricao), C2 (blendshapes->texto), C3 (frames/imagem)")
        Component(shots, "Gerenciador de Exemplos", "Python", "Zero-shot / few-shot (k exemplos do split de treino)")
        Component(client, "Cliente vLLM", "openai SDK", "Chamada HTTP a porta 80; temperatura=0, seed fixa, guided JSON")
        Component(parser, "Parser e Validador", "Python + JSON Schema", "Valida o JSON de saida e mapeia para a taxonomia comum")
        Component(fallback, "Retry / Fallback", "Python", "Reenvio em falha de formato; registra taxa de falha")
        Component(meter, "Medidor de Custo", "Python", "Latencia p50/p95, tokens, throughput")
    }

    System_Ext(vllm, "vLLM Server (porta 80)", "Gemma 4 26B-A4B")
    ContainerDb(store, "Armazenamento de Resultados", "", "")

    Rel(builder, shots, "Injeta exemplos no contexto")
    Rel(shots, client, "Mensagens montadas")
    Rel(client, vllm, "POST /v1/chat/completions")
    Rel(client, meter, "Tempos e contagem de tokens")
    Rel(client, parser, "Resposta bruta")
    Rel(parser, fallback, "Saida invalida -> reprocessa")
    Rel(parser, store, "Predicao normalizada")
    Rel(meter, store, "Metricas de custo")
```

---

## 5. UML — Diagrama de Classes (modelo de dados e núcleo)

```mermaid
classDiagram
    class Sample {
        +str sample_id
        +str dataset
        +str split
        +float start_time
        +float end_time
    }

    class BlendshapeFrame {
        +float timestamp
        +Dict coefficients
        +to_text() str
    }

    class ModalityFeatures {
        +ndarray text_glove
        +ndarray audio_covarep
        +ndarray visual_facet
        +str transcript
    }

    class GroundTruth {
        +float valence
        +List emotions
        +float sentiment
    }

    class Predictor {
        <<interface>>
        +predict(input) Prediction
        +name() str
    }

    class BaselineRuleEngine {
        +predict(BlendshapeFrame) Prediction
        +blendshape_to_va(coefs) ValenceArousal
        +detect_complex_state(va, dva_dt) str
    }

    class LLMPredictor {
        +InputCondition condition
        +int k_shots
        +predict(input) Prediction
    }

    class Prediction {
        +str predictor_name
        +str emotion_label
        +float valence
        +float arousal
        +float confidence
        +float latency_ms
    }

    class ValenceArousal {
        +float valence
        +float arousal
    }

    class Evaluator {
        +ccc(y_true, y_pred) float
        +f1(y_true, y_pred) float
        +mcnemar(pred_a, pred_b) float
        +bootstrap_ci(metric) Tuple
    }

    Predictor <|.. BaselineRuleEngine
    Predictor <|.. LLMPredictor
    Sample "1" --> "*" BlendshapeFrame
    Sample "1" --> "1" ModalityFeatures
    Sample "1" --> "1" GroundTruth
    Predictor ..> Prediction : produz
    Prediction --> ValenceArousal
    Evaluator ..> Prediction : consome
```

---

## 6. UML — Diagrama de Sequência (avaliação de uma amostra)

```mermaid
sequenceDiagram
    autonumber
    participant R as Runner
    participant F as Extracao de Features
    participant B as Baseline regras
    participant H as Harness LLM
    participant V as vLLM Gemma porta 80
    participant E as Avaliacao
    participant S as Resultados

    R->>F: get_sample(sample_id)
    alt Dataset = OMG (tem video)
        F->>F: MediaPipe -> 52 blendshapes
    else Dataset = MOSEI (sem video)
        F->>F: carrega features FACET/COVAREP/GloVe
    end
    F-->>R: amostra + features + ground truth

    par Baseline
        R->>B: predict(features)
        B->>B: blendshapes -> valencia-arousal (regras)
        B-->>S: Prediction(baseline)
    and LLM
        R->>H: predict(features, condicao C1/C2/C3)
        H->>H: monta prompt + exemplos (k-shot)
        H->>V: POST /v1/chat/completions JSON temp=0
        V-->>H: resposta (JSON)
        H->>H: valida + parseia + mapeia taxonomia
        alt formato invalido
            H->>V: retry (fallback)
        end
        H-->>S: Prediction(LLM) + latencia/tokens
    end

    R->>E: avaliar predicoes vs ground truth
    E->>E: CCC para OMG, F1/Acuracia para MOSEI, McNemar, bootstrap
    E-->>S: metricas + intervalos de confianca
    S-->>R: tabelas/figuras do paper
```

---

## 7. Condições experimentais (entrada do LLM)

| Condição | Entrada do LLM | Comparável ao baseline? | Datasets |
|---|---|---|---|
| **C1** | Transcrição/texto da fala | Indireta (texto vs. facial) | MOSEI |
| **C2** | 52 blendshapes serializados em texto/JSON | **Direta (mesma feature)** | OMG (e MOSEI se usar FACET) |
| **C3** | Frames de vídeo (visão nativa do Gemma) | Não (sem equivalente clássico) | OMG (tem vídeo) |

## 8. Métricas por dataset

| Dataset | Tarefa | Métrica primária | Métricas secundárias |
|---|---|---|---|
| **OMG-Empathy** | Regressão de valência contínua | **CCC** | RMSE, Pearson |
| **CMU-MOSEI** | Classificação de emoção (Ekman) | **F1 ponderado** | Acurácia, matriz de confusão |
| **Ambos** | Viabilidade em robótica | **Latência p50/p95** | Throughput, memória, tokens |

## 9. Decisões e premissas

- Saída do LLM forçada via **guided JSON decoding** do vLLM; `temperature=0` e `seed` fixa, com **3 execuções** para reportar média ± desvio.
- **Split oficial** de cada base, idêntico ao usado pela Léa (a confirmar por e-mail).
- Taxonomia comum mapeada a partir do conjunto Ekman; `Contempt`/estados complexos do baseline tratados em apêndice.
- Limitação declarada: possível contaminação do MOSEI no pré-treino do LLM.
