# Data Sources & Loading Pipeline

> Documentação das bases de dados utilizadas no estudo comparativo, incluindo
> origem, obtenção, estrutura, pré-processamento e pipeline de carregamento.
> Redigido para reprodutibilidade total e inclusão na seção de metodologia do
> paper (*Advanced Robotics*).

---

## 1. OMG-Empathy Dataset

### 1.1 Descrição

O **OMG-Empathy Prediction** (One-Minute Empathy) é um corpus de interações
diádicas (ator + ouvinte) projetado para avaliação de modelos de predição de
empatia. Cada sessão mostra um ator contando uma história sobre um dos 8
tópicos emocionais e o ouvinte reagindo naturalmente.

| Atributo | Valor |
|---|---|
| Publicação | Barros et al., OMG-Empathy 2019 Challenge |
| Tipo de dado | Vídeo (ator + ouvinte) + anotação contínua de valência |
| Sujeitos | 10 ouvintes × 8 histórias × 4 atores |
| Total de vídeos | 80 |
| Duração total | ~415 min (~7 h) |
| Anotação | Valência contínua (joystick, escala −1 a +1), 1 valor por frame |
| Métrica oficial | Concordance Correlation Coefficient (CCC) |
| Tracks | Personalized Empathy / Generalized Empathy |
| Licença | Disponível mediante download com senha |

### 1.2 Obtenção

- **URL de download**: Universidade de Hamburgo (WTM), arquivo `OMG_Empathy2019_full_fY4m3eyn.zip`
- **Senha**: `M2ComZJChbPc`
- **Descompactação**: `unzip -P M2ComZJChbPc OMG_Empathy2019_full_fY4m3eyn.zip -d data/`

### 1.3 Estrutura no disco

```
data/OMG_Empathy2019_full_fY4m3eyn/
├── OMG_Empathy2019/
│   ├── readme.txt
│   ├── Training/
│   │   ├── Videos/        # 40 vídeos (.mp4): Subject_{1..10}_Story_{2,4,5,8}
│   │   └── Annotations/   # 40 CSVs: header "valence", 1 float por frame
│   └── Validation/
│       ├── Videos/        # 10 vídeos: Subject_{1..10}_Story_1
│       └── Annotations/   # 10 CSVs
├── OMG_Empathy2019_testSet/
│   └── Videos/            # 30 vídeos: Subject_{1..10}_Story_{3,6,7}
└── Annotations/           # 30 CSVs (anotações do test set, na raiz)
```

### 1.4 Splits oficiais (por história)

| Split | Histórias | Vídeos | Janelas (4 s) |
|---|---|---|---|
| **Train** | 2, 4, 5, 8 | 40 | 3.116 |
| **Validation** | 1 | 10 | 960 |
| **Test** | 3, 6, 7 | 30 | 2.296 |
| **Total** | — | 80 | 6.372 |

> A separação garante que nenhum sujeito/história aparece simultaneamente em
> mais de um split.

### 1.5 Formato das anotações

Cada arquivo CSV contém:

```csv
valence
0.0
0.10000000000000001
0.20000000000000001
-0.10000000000000001
...
```

- Header: `valence`
- Uma linha por frame do vídeo (taxa assumida = FPS do vídeo; ~25 fps)
- Faixa: \[-1.0, +1.0\] (negativo = desprazer, positivo = prazer)

### 1.6 Pipeline de carregamento (`src/data/omg_loader.py`)

1. **Pareamento**: para cada `.mp4` no diretório de vídeos do split, localiza o `.csv` correspondente no diretório de anotações (mesmo `stem`).
2. **FPS**: lê do vídeo via OpenCV (`CAP_PROP_FPS`); fallback = 25 fps se indisponível.
3. **Janelamento temporal**: divide as anotações de valência em janelas de `window_s` segundos (default = 4 s, configurável em `config.yaml`). A **valência da janela** é a média aritmética dos valores de frame contidos nela.
4. **Geração de `Sample`**: cada janela vira um `Sample` com:
   - `sample_id`: `{video_stem}_w{idx:04d}` (ex.: `Subject_3_Story_6_w0012`)
   - `ground_truth.valence`: média da janela
   - `video_path`: caminho do `.mp4` (para extração posterior de blendshapes/keyframes)
   - `frame_time`: instante representativo (centro da janela, em segundos)

### 1.7 Extração de features multi-frame

Os vídeos OMG são split-screen (ator à esquerda, ouvinte à direita). A face do
**ouvinte** é a de interesse; a extração implementa:

1. **Recorte**: `crop_listener_half` isola a metade direita do frame (configurável
   via `data.omg_listener_side` em `config.yaml`).
2. **Detecção por tiling**: o MediaPipe Face Landmarker opera melhor em faces que
   ocupam boa parte da imagem. Para frames largos, `from_large_frame` aplica
   janela deslizante (sliding window) na metade recortada, testando sub-regiões
   até obter detecção. Um cache (`_last_box`) reutiliza a última região bem-
   sucedida para frames consecutivos.
3. **Multi-frame por janela**: em vez de um único keyframe central, extraímos
   **8 frames** uniformemente espaçados dentro da janela de 4 s
   (`data.n_frames_per_window = 8`). Os 52 coeficientes de blendshape são
   agregados em estatísticas resumo:
   - **Média** (`mean`): ativação tônica durante a janela
   - **Máximo** (`max`): pico de ativação (sinal afetivo mais forte)
   - **Desvio-padrão** (`std`): variabilidade temporal

   Essas 3 estatísticas × top-N blendshapes mais ativas são serializadas em
   texto para a condição **C2**. Para a condição **C3**, são enviados
   **3 frames** (`data.c3_n_frames = 3`) como imagens JPEG codificadas em base64.

4. **Baseline multi-frame**: o módulo `rule_engine.predict_sequence` aplica as
   mesmas regras blendshapes→valência-arousal a cada frame e agrega (média +
   pico), garantindo comparação justa com o LLM.

---

## 2. CMU-MOSEI Dataset

### 2.1 Descrição

O **CMU Multimodal Opinion Sentiment and Emotion Intensity** (CMU-MOSEI) é o
principal benchmark para análise multimodal de sentimento e emoção em vídeos
espontâneos do YouTube.

| Atributo | Valor |
|---|---|
| Publicação | Zadeh et al., ACL 2018 |
| Tipo de dado | Features pré-extraídas (texto, áudio, facial) — **sem vídeo bruto** |
| Vídeos (IDs únicos) | ~3.293 |
| Segmentos anotados | ~23.000 |
| Rótulos por segmento | Sentimento \[-3, +3\] + 6 emoções Ekman \[0, 3\] |
| Emoções | happiness, sadness, anger, surprise, disgust, fear |
| Métrica recomendada | F1 ponderado / acurácia (emoção); MAE / Corr (sentimento) |
| Licença | MIT (SDK); Apache 2.0 (Kaggle mirror) |

### 2.2 Obtenção

- **Fonte**: Kaggle — `samarwarsi/cmu-mosei` ([link](https://www.kaggle.com/datasets/samarwarsi/cmu-mosei/data))
- **Download**: via Kaggle CLI (apenas os arquivos `.csd` necessários, ~1.7 GB no total)

```bash
# Apenas os 3 CSDs usados no pipeline (labels, transcrição, FACET):
kaggle datasets download -d samarwarsi/cmu-mosei \
  -f CMU-MOSEI/labels/CMU_MOSEI_Labels.csd \
  -f CMU-MOSEI/languages/CMU_MOSEI_TimestampedWords.csd \
  -f CMU-MOSEI/visuals/CMU_MOSEI_VisualFacet42.csd \
  -p data/cmu-mosei --unzip
```

- **Tamanho total do dataset**: ~29.1 GB (compactado). Porém **apenas ~1.7 GB**
  são necessários (Labels + TimestampedWords + VisualFacet42). Os demais CSDs
  (GloVe, COVAREP, OpenFace2, Phones) não são usados neste estudo.

### 2.3 Estrutura no disco

Os dados são armazenados em arquivos `.csd` (Computational Sequential Data),
que são **HDF5** internamente.

```
data/cmu-mosei/CMU-MOSEI/
├── labels/
│   └── CMU_MOSEI_Labels.csd           # 7-dim: [sentimento, hap, sad, ang, sur, dis, fea]
├── languages/
│   ├── CMU_MOSEI_TimestampedWords.csd  # palavras (byte strings) por segmento
│   ├── CMU_MOSEI_TimestampedWordVectors.csd  # GloVe 300-dim
│   └── CMU_MOSEI_TimestampedPhones.csd
├── acoustics/
│   └── CMU_MOSEI_COVAREP.csd          # 74-dim (features acústicas)
└── visuals/
    ├── CMU_MOSEI_VisualOpenFace2.csd   # 713-dim (OpenFace 2.0)
    └── CMU_MOSEI_VisualFacet42.csd     # 35-dim (FACET action units)
```

### 2.4 Formato dos arquivos `.csd` (HDF5)

Cada `.csd` contém um grupo raiz (nome da computational sequence, ex.:
`All Labels`, `words`, `FACET 4.2`). Dentro dele, um grupo `data` contém
subgrupos por **video_id**. Cada video_id armazena **todos os seus segmentos
como linhas** de dois arrays:

```
CMU_MOSEI_Labels.csd
└── All Labels/
    ├── data/
    │   ├── --qXJuDtHPw/
    │   │   ├── features   # numpy array (n_segments, 7)
    │   │   └── intervals  # numpy array (n_segments, 2) — [start_s, end_s]
    │   ├── _dI--eQ6qVU/
    │   │   ├── features
    │   │   └── intervals
    │   └── ...
    └── metadata/
```

A indexação é por **vídeo** (cada vídeo contém N segmentos como linhas da
matriz `features`). O alinhamento entre modalidades (labels, palavras, FACET)
é feito por **sobreposição de intervalos de tempo** — o intervalo `[start, end]`
de um segmento de label define a janela temporal, e as linhas correspondentes
de palavras/FACET cujos intervalos sobrepõem essa janela são selecionadas.

### 2.5 Layout do vetor de rótulos (7 dimensões)

| Índice | Nome | Faixa | Descrição |
|---|---|---|---|
| 0 | sentiment | \[-3, +3\] | Intensidade de sentimento (negativo ↔ positivo) |
| 1 | happiness | \[0, 3\] | Intensidade de alegria |
| 2 | sadness | \[0, 3\] | Intensidade de tristeza |
| 3 | anger | \[0, 3\] | Intensidade de raiva |
| 4 | surprise | \[0, 3\] | Intensidade de surpresa |
| 5 | disgust | \[0, 3\] | Intensidade de nojo |
| 6 | fear | \[0, 3\] | Intensidade de medo |

> A conversão para rótulo categórico usa **threshold > 0**: toda emoção com
> intensidade > 0 entra como rótulo ativo (multi-rótulo). Se nenhuma emoção
> supera o limiar, o segmento é classificado como **neutral**.

### 2.6 Splits oficiais

O CMU-MOSEI define splits por **video_id** (segmentos do mesmo vídeo nunca são
divididos entre splits). As proporções aproximadas:

| Split | Segmentos | Proporção |
|---|---|---|
| **Train** | ~16.300 | ~71% |
| **Validation** | ~1.870 | ~8% |
| **Test** | ~4.660 | ~21% |
| **Total** | ~22.850 | 100% |

O loader tenta obter os folds exatos do SDK (`mmsdk.cmu_mosei.standard_folds`);
se o SDK não estiver instalado, usa partição determinística por hash do
`video_id` com as mesmas proporções.

### 2.7 Pipeline de carregamento (`src/data/mosei_loader.py`)

1. **Abertura dos `.csd` via `h5py`** (sem depender do `mmsdk`): cada arquivo
   é aberto com `h5py.File` e o grupo `data` (dentro do root da computational
   sequence) é extraído. Tokens de pausa (`sp`, `sil`) são filtrados.

2. **Iteração por vídeo**: para cada `video_id` no Labels.csd:
   - Verifica o split (train/validation/test) via SDK do MOSEI se instalado,
     ou por hash determinístico do `video_id` (proporções ~71/8/21%).
   - Lê `features` (n_seg × 7) e `intervals` (n_seg × 2) do Labels.
   - Lê `features` e `intervals` do TimestampedWords e do VisualFacet42 para
     o mesmo `video_id` (se disponíveis).

3. **Alinhamento por intervalo de tempo**: para cada segmento de label `i`:
   - O intervalo `[start_i, end_i]` define a janela temporal.
   - **Transcrição (C1)**: seleciona palavras cujos intervalos sobrepõem
     `[start_i, end_i]`, decodifica byte strings, filtra tokens de pausa e
     concatena em texto corrido.
   - **FACET (C2)**: seleciona frames FACET cujos intervalos sobrepõem a
     janela e calcula a **média temporal** (35-dim).

4. **Conversão de rótulos**: `label_vector_to_ground_truth(vec)` produz
   `GroundTruth` com:
   - `emotion_scores`: dict `{emo: intensidade}` das 6 emoções Ekman
   - `emotions`: lista multi-rótulo (emoções com intensidade > 0)
   - `sentiment`: valor contínuo \[−3, +3\]

5. **Geração de `Sample`**: cada segmento vira um `Sample` com:
   - `sample_id`: `{video_id}[{seg_idx}]` (ex.: `--qXJuDtHPw[3]`)
   - `ground_truth`: conforme item 4
   - `transcript`: texto reconstruído (para condição **C1**)
   - `facet`: vetor 35-dim médio (para condição **C2**)

> **Nota**: o MOSEI **não disponibiliza vídeo bruto** — apenas features
> pré-processadas. Por isso a condição **C3 (multimodal/imagem) não se aplica**
> a este dataset.

---

## 3. Diferenças entre as bases e impacto no protocolo

| Aspecto | OMG-Empathy | CMU-MOSEI |
|---|---|---|
| **Dado bruto** | Vídeo (.mp4) | Features pré-extraídas (.csd) |
| **Tarefa** | Regressão de valência contínua | Classificação de emoção (multi-rótulo) |
| **Métrica primária** | CCC por série temporal de vídeo | F1 multi-rótulo (micro, **macro**, ponderado) |
| **Anotação** | Valência por frame (joystick, −1 a +1) | Intensidade de 6 emoções Ekman (0–3) + sentimento |
| **Extração facial** | MediaPipe 52 blendshapes (nosso pipeline, multi-frame) | FACET 35-dim (pré-extraída, média temporal) |
| **Condição C1 (texto)** | Não aplicável (sem transcrição oficial) | Disponível (TimestampedWords) |
| **Condição C2 (facial→texto)** | Blendshapes serializados (mean/max/std) | FACET serializado (proxy) |
| **Condição C3 (visão nativa)** | 3 keyframes JPEG por janela | **Indisponível** |
| **Comparação justa (C2)** | Mesma feature (blendshapes) para LLM e baseline | Proxy (FACET ≠ blendshapes MediaPipe) |

### 3.1 Implicações

- No **OMG-Empathy**, a comparação baseline-vs-LLM na condição C2 é **perfeitamente
  justa**: ambos recebem exatamente os mesmos 52 coeficientes de blendshape
  (mesmos frames, mesma extração multi-frame).
- No **CMU-MOSEI**, a condição C2 usa FACET como proxy da entrada facial; a
  comparação mais forte é na condição **C1 (texto)**, onde o LLM recebe a
  transcrição — a mesma representação textual disponível ao baseline.
- A condição **C3** explora o teto do LLM usando visão nativa, mas fica restrita
  ao OMG (único dataset com vídeo bruto).

---

## 4. Geração de exemplos few-shot

O módulo `src/llm/fewshot.py` constrói exemplos in-context a partir do split
de **treino** (sem vazamento para o split de teste):

### 4.1 OMG — few-shot C2 (blendshapes → valência)

1. Carrega amostras de treino e embaralha com `seed` fixa.
2. Para cada amostra, extrai a sequência multi-frame de blendshapes e serializa
   com as mesmas estatísticas (mean/max/std) usadas na inferência.
3. O output de referência é `{ "valence": <gt×100>, "arousal": <baseline>, "confidence": 85 }`,
   onde a valência vem do ground truth e o arousal da baseline de regras.
4. Gera `k = 3` exemplos (configurável via `--k-shots`).

### 4.2 MOSEI — few-shot C1/C2 (texto/FACET → emoção)

1. Carrega amostras de treino e embaralha com `seed` fixa.
2. **C1**: usa a transcrição como input. **C2**: serializa o vetor FACET médio.
3. O output de referência é um JSON `EmotionPrediction` com intensidades Ekman
   normalizadas de \[0,3\] para \[0,100\], valência normalizada de \[-3,+3\] para
   \[-100,+100\], e `emotion_label` = emoção dominante do ground truth.
4. Gera `k = 5` exemplos (configurável).

Cada exemplo é injetado no prompt como par `HumanMessage` / `AIMessage`
intercalado entre o `SystemMessage` e o prompt real de inferência.

---

## 5. Protocolos de avaliação

### 5.1 OMG-Empathy — CCC por série temporal

Seguindo o protocolo oficial do OMG-Empathy Challenge (Barros et al., 2019):

1. Para cada vídeo do split de teste, a série temporal de valência predita
   (uma predição por janela de 4 s) é comparada com a série temporal de
   valência anotada (média dos frames da mesma janela).
2. O CCC é calculado **por vídeo** usando a fórmula de Lin (1989):
   CCC = 2ρσ\_xσ\_y / (σ²\_x + σ²\_y + (μ\_x − μ\_y)²).
3. O resultado reportado é a **média dos CCCs** entre todos os vídeos avaliados,
   com desvio-padrão entre vídeos como medida de dispersão.

> **Nota metodológica**: calcular CCC agregando janelas de múltiplos sujeitos
> (ignorando a estrutura temporal) produz CCC ≈ 0 — um artefato do offset
> inter-sujeito que domina a variância. O protocolo por vídeo evita isso.

### 5.2 CMU-MOSEI — F1 multi-rótulo com varredura de limiar

1. Cada emoção Ekman é tratada como rótulo binário: **presença se intensidade > 0**
   (threshold do ground truth).
2. O LLM produz scores de intensidade inteiros (0–100) para cada emoção Ekman.
   Como esses scores tendem a ser **conservadores** (concentrados em faixas
   baixas), um **threshold fixo de 0.5 (= 50/100)** penalizaria o LLM injustamente.
3. **Varredura de limiar**: testamos thresholds de 0.01 a 0.50 (passo 0.01) e
   reportamos o F1 no limiar ótimo por condição. Isso dá ao LLM a chance de
   calibrar seu range de scores.
4. Métricas reportadas:
   - **F1 micro**: favorece classes majoritárias (ex.: happiness).
   - **F1 macro**: média não-ponderada entre emoções — **métrica principal**
     por ser justa com emoções minoritárias (anger, fear, disgust, surprise).
   - **F1 ponderado**: média ponderada por suporte.
   - **F1 por emoção**: permite identificar forças/fraquezas do LLM vs baseline.

### 5.3 Baselines

| Base | Baseline | Método |
|---|---|---|
| OMG-Empathy | `face_blendshape` (regras) | Mapeamento rule-based de 52 blendshapes → valência/arousal. Multi-frame: aplica regras a cada frame e agrega (média + pico). |
| CMU-MOSEI | Regressão Logística (LogReg) | `sklearn.linear_model.LogisticRegression` (multi-saída), treinada em ~6.000 segmentos de treino usando features FACET 35-dim. Prediz 6 probabilidades de emoção Ekman independentes. |
| CMU-MOSEI | Acaso (chance) | Sorteio aleatório proporcional à prevalência de cada emoção no teste. |
| CMU-MOSEI | Majority | Prediz sempre a emoção mais prevalente (*happiness*) como ativa, todas as demais inativas. |

---

## 6. Reprodutibilidade

| Item | Valor |
|---|---|
| Config de paths | `config/config.yaml` → `data.omg_path`, `data.mosei_path` |
| Janelamento OMG | `window_s = 4.0` (configurável) |
| Frames por janela (C2) | `n_frames_per_window = 8` |
| Frames por janela (C3) | `c3_n_frames = 3` |
| Listener side (OMG) | `omg_listener_side = right` |
| Top-N blendshapes | `top_n_blendshapes = 15` |
| FPS OMG fallback | 25 fps (quando OpenCV indisponível) |
| Threshold emoção MOSEI | intensidade > 0.0 |
| Split MOSEI (sem SDK) | Hash determinístico MD5 do `video_id` mod 100 (~71/8/21%) |
| Seed do few-shot | Registrada no parquet de resultados (default: 42) |
| LLM temperatura | 0 (determinístico; leve variância MoE/vLLM) |
| Max tokens (LLM) | 512 |
| Parsing de saída | Manual: extração de JSON do texto bruto + validação Pydantic |
| Infraestrutura GPU | GCP Spot VM — NVIDIA L4 24 GB |
