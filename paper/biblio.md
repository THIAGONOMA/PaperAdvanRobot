# Bibliografia — Estudo Comparativo de Reconhecimento de Emoção

> Referências organizadas por tema: bases de dados, ferramentas/modelos e
> trabalhos relacionados (LLMs para emoção). Inclui paper, fonte, abstract e
> resumo. Para uso na seção de *Related Work* / *References* do paper
> (*Advanced Robotics*).

---

## A. Bases de dados

---

### A1. OMG-Empathy Dataset

**Paper:** The OMG-Empathy Dataset: Evaluating the Impact of Affective Behavior in Storytelling

**Autores:** Pablo V. A. Barros, Nikhil Churamani, Angelica Lim, Stefan Wermter

**Fonte:** 8th International Conference on Affective Computing and Intelligent Interaction (ACII 2019), IEEE, pp. 1–7

**Links:**
- arXiv: [1908.11706](https://arxiv.org/abs/1908.11706)
- Challenge: [OMG-Empathy Prediction](https://www2.informatik.uni-hamburg.de/wtm/omgchallenges/omg_empathy_description_19.html)

**Abstract:** We propose a novel dataset composed of dyadic interactions designed, collected and annotated with a focus on measuring the affective impact that eight different stories have on the listener. Each video of the dataset contains around 5 minutes of interaction where a speaker tells a story to a listener. After each interaction, the listener annotated, using a valence scale, how the story impacted their emotional state. To encourage the development of artificial empathy models suitable for real-world scenarios, we propose the OMG-Empathy Dataset along with two different evaluation protocols. The dataset presents a realistic approach for training and evaluating artificial empathy systems, focusing on the impact that an affective interaction has on a listener. It is composed of 7 hours of audio-visual recordings of human-human interactions, collected with 10 different participants interacting with 4 different speakers. Each participant held 2 dialogues with each speaker, based on different storylines, totaling 80 interaction videos spanning on average 5 min 12 s. Immediately after each session, the participants watched the interaction again and annotated their intrinsic affective state using a continuous valence scale (−1 to +1) via a joystick.

**Resumo:** Base de interações diádicas (ator + ouvinte) com 80 vídeos (~7 h). O ouvinte anota valência contínua (−1 a +1) após cada sessão. Dois protocolos de avaliação: Personalized (prever empatia de um sujeito específico) e Generalized (prever impacto afetivo de cada história). Métrica oficial: CCC. Relevante para o nosso estudo por fornecer **vídeo bruto** (permite extração de blendshapes via MediaPipe) e anotação contínua de valência.

**BibTeX:**
```bibtex
@inproceedings{Barros2019OMGEmpathy,
  author    = {Barros, Pablo V. A. and Churamani, Nikhil and Lim, Angelica and Wermter, Stefan},
  title     = {The {OMG}-Empathy Dataset: Evaluating the Impact of Affective Behavior in Storytelling},
  booktitle = {8th International Conference on Affective Computing and Intelligent Interaction (ACII)},
  pages     = {1--7},
  year      = {2019},
  publisher = {IEEE},
  doi       = {10.1109/ACII.2019.8925466}
}
```

---

### A2. CMU-MOSEI Dataset

**Paper:** Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph

**Autores:** AmirAli Bagher Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, Louis-Philippe Morency

**Fonte:** Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018), Vol. 1 (Long Papers), pp. 2236–2246

**Links:**
- ACL Anthology: [P18-1208](https://aclanthology.org/P18-1208/)
- PDF: [aclanthology.org/P18-1208.pdf](https://aclanthology.org/P18-1208.pdf)
- SDK: [CMU-MultimodalSDK (GitHub)](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)

**Abstract:** Analyzing human multimodal language is an emerging area of research in NLP. Intrinsically human communication is multimodal (heterogeneous), temporal and asynchronous; it consists of the language (words), visual (expressions), and acoustic (paralinguistic) modalities all in the form of asynchronous coordinated sequences. From a resource perspective, there is a genuine need for large scale datasets that allow for in-depth studies of multimodal language. In this paper we introduce CMU Multimodal Opinion Sentiment and Emotion Intensity (CMU-MOSEI), the largest dataset of sentiment analysis and emotion recognition to date. CMU-MOSEI consists of 23,453 annotated sentences from more than 1,000 online speakers and 250 different topics. Using data from CMU-MOSEI and a novel multimodal fusion technique called the Dynamic Fusion Graph (DFG), we conduct experimentation to investigate how modalities interact with each other in human multimodal language. Unlike previously proposed fusion techniques, DFG is highly interpretable and achieves competitive performance compared to the current state of the art.

**Resumo:** Maior benchmark de análise multimodal de sentimento e emoção até a data de publicação. Contém ~23 k segmentos de vídeos do YouTube com rótulos de sentimento (−3 a +3) e 6 emoções Ekman (intensidade 0–3). Features disponíveis: GloVe (texto), COVAREP (áudio), FACET/OpenFace2 (facial). Introduz também o Dynamic Fusion Graph (DFG). Relevante para o nosso estudo como **segunda base de avaliação**, com classificação multimodal de emoção; usamos as features pré-extraídas (sem vídeo bruto) nas condições C1 (texto) e C2 (FACET).

**BibTeX:**
```bibtex
@inproceedings{Zadeh2018MOSEI,
  author    = {Zadeh, AmirAli Bagher and Liang, Paul Pu and Poria, Soujanya and Cambria, Erik and Morency, Louis-Philippe},
  title     = {Multimodal Language Analysis in the Wild: {CMU-MOSEI} Dataset and Interpretable Dynamic Fusion Graph},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {2236--2246},
  year      = {2018},
  publisher = {Association for Computational Linguistics}
}
```

---

### A3. CMU Multimodal SDK

**Paper:** Multi-attention Recurrent Network for Human Communication Comprehension

**Autores:** Amir Zadeh, Paul Pu Liang, Soujanya Poria, Prateek Vij, Erik Cambria, Louis-Philippe Morency

**Fonte:** Thirty-Second AAAI Conference on Artificial Intelligence (AAAI 2018)

**Link:** [GitHub](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK)

**Resumo:** O SDK fornece a infraestrutura de "computational sequences" (.csd, HDF5) para download, alinhamento e carregamento padronizado dos datasets multimodais da CMU. Usado na comunidade como a forma canônica de acessar CMU-MOSI e CMU-MOSEI. No nosso pipeline, os `.csd` são lidos diretamente via `h5py` para maior portabilidade.

**BibTeX:**
```bibtex
@inproceedings{Zadeh2018MultiAttention,
  author    = {Zadeh, Amir and Liang, Paul Pu and Poria, Soujanya and Vij, Prateek and Cambria, Erik and Morency, Louis-Philippe},
  title     = {Multi-attention Recurrent Network for Human Communication Comprehension},
  booktitle = {Thirty-Second AAAI Conference on Artificial Intelligence},
  year      = {2018}
}
```

---

## B. Ferramentas e modelos

---

### B1. MediaPipe Face Landmarker & Blendshapes

**Paper:** Blendshapes GHUM: Real-time Monocular Facial Blendshape Prediction

**Autores:** Google Research (Razvan Surdulescu et al.)

**Fonte:** arXiv preprint, 2023

**Links:**
- arXiv: [2309.05782](https://arxiv.org/abs/2309.05782)
- MediaPipe Docs: [Face landmark detection guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)

**Abstract:** We present Blendshapes GHUM — an on-device ML pipeline that predicts 52 facial blendshape coefficients at 30+ FPS on modern mobile phones, from a single monocular RGB image and enables facial motion capture applications like virtual avatars. Our main contributions are: i) an annotation-free offline method for obtaining blendshape coefficients from real-world human scans, ii) a lightweight real-time model that predicts blendshape coefficients based on facial landmarks.

**Resumo:** Pipeline em tempo real que prediz 52 coeficientes de blendshape facial a partir de 478 landmarks 2D detectados pelo MediaPipe Face Mesh. Usa uma arquitetura leve (MLP-Mixer) treinada com dados de scans 3D reais (GHUM). Roda a 30+ FPS em celulares. No nosso estudo, é a **ferramenta de extração de features** do baseline (módulo `face_blendshape`) e a fonte de representação facial para a condição C2.

**BibTeX:**
```bibtex
@article{BlendshapesGHUM2023,
  author  = {Surdulescu, Razvan and others},
  title   = {Blendshapes {GHUM}: Real-time Monocular Facial Blendshape Prediction},
  journal = {arXiv preprint arXiv:2309.05782},
  year    = {2023}
}
```

---

### B2. Gemma 3 (base arquitetural multimodal da família)

**Paper:** Gemma 3 Technical Report

**Autores:** Gemma Team, Google DeepMind

**Fonte:** arXiv preprint, 2025

**Links:**
- arXiv: [2503.19786](https://arxiv.org/abs/2503.19786)

**Abstract:** We introduce Gemma 3, a multimodal addition to the Gemma family of lightweight open models, ranging in scale from 1 to 27 billion parameters. This version introduces vision understanding abilities, a wider coverage of languages and longer context – at least 128K tokens. We also change the architecture of the model to reduce the KV-cache memory that tends to explode with long context. This is achieved by increasing the ratio of local to global attention layers, and keeping the span on local attention short. The Gemma 3 models are trained with distillation and achieve superior performance to Gemma 2 for both pre-trained and instruction finetuned versions.

**Resumo:** Report técnico da geração multimodal da família Gemma (Google DeepMind). Introduz compreensão de imagens via encoder SigLIP congelado, contexto de 128K tokens e suporte a 140+ idiomas. Arquitetura com atenção local-global intercalada e destilação de conhecimento. Documenta a base arquitetural que o **Gemma 4** (usado no nosso estudo) herda e estende.

**BibTeX:**
```bibtex
@article{Gemma3Team2025,
  author  = {{Gemma Team, Google DeepMind}},
  title   = {Gemma 3 Technical Report},
  journal = {arXiv preprint arXiv:2503.19786},
  year    = {2025}
}
```

---

### B2b. Gemma 4 26B-A4B (modelo usado no estudo)

**Fonte:** Google DeepMind, abril 2026

**Links:**
- Model Card: [ai.google.dev/gemma/docs/core/model_card_4](https://ai.google.dev/gemma/docs/core/model_card_4)
- HuggingFace: [google/gemma-4-26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B)
- Blog: [blog.google — Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)

**Resumo:** Geração mais recente da família Gemma. O **26B-A4B-it** é um modelo MoE (Mixture of Experts) com 25.2B parâmetros totais e apenas **3.8B ativos por token** (8 experts ativos / 128 totais + 1 compartilhado). Suporta texto + imagem, contexto de 256K tokens, 140+ idiomas. No nosso estudo, usamos a variante quantizada **AWQ 4-bit** (`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`) servida via vLLM em GPU NVIDIA L4 24 GB. Licença Apache 2.0. Até junho 2026 não há report técnico formal publicado em arXiv; citamos o model card oficial e o report do Gemma 3 como base arquitetural.

| Propriedade | Valor |
|---|---|
| Parâmetros totais | 25.2B |
| Parâmetros ativos | 3.8B |
| Camadas | 30 |
| Experts | 8 ativos / 128 totais + 1 compartilhado |
| Contexto | 256K tokens |
| Modalidades | Texto + Imagem |
| Encoder de visão | ~550M parâmetros |
| Quantização usada | AWQ 4-bit |

---

### B3. vLLM (servidor de inferência)

**Paper:** Efficient Memory Management for Large Language Model Serving with PagedAttention

**Autores:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica

**Fonte:** Proceedings of the 29th Symposium on Operating Systems Principles (SOSP 2023)

**Links:**
- arXiv: [2309.06180](https://arxiv.org/abs/2309.06180)
- GitHub: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

**Abstract:** High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. We propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems. On top of it, we build vLLM, an LLM serving system that achieves near-zero waste in KV cache memory and flexible sharing of KV cache within and across requests. Our evaluations show that vLLM improves the throughput of popular LLMs by 2–4× with the same level of latency.

**Resumo:** Sistema de serving de LLMs com gerenciamento de memória KV-cache inspirado em paginação de sistemas operacionais. No nosso estudo, o vLLM serve o Gemma 4 AWQ 4-bit em GPU L4 24 GB, expondo uma API compatível com OpenAI (`/v1/chat/completions`). Permite servir o modelo MoE 26B quantizado em uma única GPU consumer-grade.

**BibTeX:**
```bibtex
@inproceedings{Kwon2023vLLM,
  author    = {Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph E. and Zhang, Hao and Stoica, Ion},
  title     = {Efficient Memory Management for Large Language Model Serving with {PagedAttention}},
  booktitle = {Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)},
  year      = {2023}
}
```

---

### B4. AWQ (quantização do modelo)

**Paper:** AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

**Autores:** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han

**Fonte:** Proceedings of MLSys 2024

**Links:**
- arXiv: [2306.00978](https://arxiv.org/abs/2306.00978)
- Projeto: [hanlab.mit.edu/projects/awq](https://hanlab.mit.edu/projects/awq)
- GitHub: [github.com/mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)

**Abstract:** We propose Activation-aware Weight Quantization (AWQ), a hardware-friendly approach for LLM low-bit weight-only quantization. Our method is based on the observation that weights are not equally important: protecting only 1% of salient weights can greatly reduce quantization error. We then propose to search for the optimal per-channel scaling that protects the salient weights by observing the activation, not weights. AWQ does not rely on any backpropagation or reconstruction, so it can well preserve LLMs' generalization ability on different domains and modalities.

**Resumo:** Técnica de quantização weight-only (4-bit) que identifica ~1% de pesos salientes via estatísticas de ativação e os protege com scaling por canal. Preserva a capacidade de generalização multimodal do modelo. Usamos AWQ 4-bit para comprimir o Gemma 4 26B de ~50 GB (bf16) para ~13 GB, viabilizando a inferência em GPU L4 24 GB.

**BibTeX:**
```bibtex
@inproceedings{Lin2024AWQ,
  author    = {Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
  title     = {{AWQ}: Activation-aware Weight Quantization for {LLM} Compression and Acceleration},
  booktitle = {MLSys},
  year      = {2024}
}
```

---

### B5. LangGraph & LangChain (orquestração)

**Fonte:** LangChain, Inc., 2024

**Links:**
- LangGraph GitHub: [github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
- LangChain Docs: [docs.langchain.com](https://docs.langchain.com)

**Resumo:** LangGraph é um framework de orquestração de baixo nível para construir agentes com estado, baseado em grafos cíclicos. Construído sobre o ecossistema LangChain (que fornece integrações com modelos e ferramentas). No nosso pipeline, o LangGraph orquestra o fluxo de inferência do LLM: construção de prompts, chamada à API OpenAI-compatível (vLLM), parsing manual de JSON e validação Pydantic. Usamos `langchain_openai.ChatOpenAI` como wrapper do endpoint.

**BibTeX:**
```bibtex
@misc{LangGraphLangChain2024,
  author = {{LangChain, Inc.}},
  title  = {{LangGraph}: Building Language Agents as Graphs},
  year   = {2024},
  url    = {https://github.com/langchain-ai/langgraph}
}
```

---

## C. Trabalhos relacionados (LLMs para emoção)

---

### C1. Survey: Affective Computing na era dos LLMs

**Paper:** Affective Computing in the Era of Large Language Models: A Survey from the NLP Perspective

**Fonte:** arXiv, 2024

**Link:** [2408.04638](https://arxiv.org/abs/2408.04638)

**Abstract:** Affective Computing (AC) integrates computer science, psychology, and cognitive science to enable machines to recognize, interpret, and simulate human emotions across domains such as social media, finance, healthcare, and education. AC commonly centers on two task families: Affective Understanding (AU) and Affective Generation (AG). While fine-tuned pre-trained language models (PLMs) have achieved solid AU performance, they often generalize poorly across tasks and remain limited for AG. The advent of Large Language Models (LLMs) has catalyzed a paradigm shift by offering in-context learning, broader world knowledge, and stronger sequence generation. This survey presents an NLP-oriented overview of AC in the LLM era. We consolidate traditional AC tasks and preliminary LLM-based studies; review adaptation techniques including Instruction Tuning, Prompt Engineering (zero/few-shot, chain-of-thought, agent-based prompting), and Reinforcement Learning; compile benchmarks and evaluation practices; and discuss open challenges from ethics and data quality to robust evaluation and resource efficiency.

**Resumo:** Survey amplo e recente que cobre como LLMs estão sendo adaptados para tarefas de computação afetiva — entendimento (classificação de emoção/sentimento) e geração (respostas empáticas). Revisa zero-shot, few-shot, CoT e RL para emoção. Referência essencial para posicionar nosso comparativo na literatura.

---

### C2. Survey: MLLMs para Reconhecimento de Emoção Multimodal

**Paper:** Multimodal Large Language Models Meet Multimodal Emotion Recognition and Reasoning: A Survey

**Fonte:** arXiv, 2025

**Link:** [2509.24322](https://arxiv.org/abs/2509.24322)

**Abstract:** In recent years, large language models (LLMs) have driven major advances in language understanding, marking a significant step toward artificial general intelligence (AGI). With increasing demands for higher-level semantics and cross-modal fusion, multimodal large language models (MLLMs) have emerged, integrating diverse information sources (text, vision, audio) to enhance modeling and reasoning in complex scenarios. Multimodal emotion recognition and reasoning has become a rapidly growing frontier. This paper provides a comprehensive survey of LLMs and MLLMs for emotion recognition and reasoning, covering model architectures, datasets, and performance benchmarks. We highlight key challenges and outline future research directions.

**Resumo:** O survey mais recente e abrangente sobre MLLMs (modelos multimodais grandes) para reconhecimento e raciocínio emocional. Cobre arquiteturas, datasets (incluindo MOSEI), benchmarks e paradigmas (zero-shot, few-shot, fine-tuning). Diretamente relevante para justificar o uso de um LLM multimodal (Gemma) como comparativo.

---

### C3. Emotional Blind Spots de LLMs

**Paper:** Fluent but Unfeeling: The Emotional Blind Spots of Language Models

**Fonte:** arXiv, 2025

**Link:** [2509.09593](https://arxiv.org/abs/2509.09593)

**Abstract:** The versatility of LLMs in natural language understanding has made them popular in mental health research. While many studies explore LLMs' capabilities in emotion recognition, a critical gap remains in evaluating whether LLMs align with human emotions at a fine-grained level. Existing research typically focuses on classifying emotions into predefined categories, overlooking nuanced expressions. We introduce EXPRESS, a benchmark dataset curated from Reddit with 251 fine-grained self-disclosed emotion labels. Systematic testing of prevalent LLMs under zero-shot, few-shot, and CoT settings reveals that accurately predicting emotions that align with human self-disclosed emotions remains challenging. Qualitative analysis shows that while certain LLMs generate emotion terms consistent with theories and definitions, they sometimes fail to capture contextual cues as effectively as human self-disclosures.

**Resumo:** Estudo crítico que mostra que LLMs ainda têm dificuldade com emoções finas — geram rótulos plausíveis em teoria, mas falham na captura de pistas contextuais sutis. Relevante para a **discussão de limitações** do nosso paper: o LLM pode acertar categorias amplas (Ekman) mas errar nuances.

---

### C4. Survey: LLMs para Análise Multimodal de Sentimento

**Paper:** Large Language Models Meet Text-Centric Multimodal Sentiment Analysis: A Survey

**Fonte:** arXiv, 2024

**Link:** [2406.08068](https://arxiv.org/abs/2406.08068)

**Abstract:** Research into multimodal sentiment analysis technologies with human-like emotion processing capabilities will provide technical support for real-world applications such as intelligent companions, customer service, e-commerce, and depression detection. Large language models (LLMs) have demonstrated astonishing conversational capabilities and showcased impressive performance across NLP tasks. Large multimodal models (LMMs) that increase the ability to understand modalities such as images also provide new ideas. They can directly perform tasks with zero-shot or few-shot context learning, requiring no supervised training. While there have been some attempts to apply LLMs in text-based sentiment analysis, there is a lack of systematic analysis regarding the application of LLMs and LMMs in multimodal sentiment analysis.

**Resumo:** Survey focado no uso de LLMs/LMMs para análise de sentimento multimodal. Cobre paradigmas parameter-frozen (zero/few-shot) vs. parameter-tuning (fine-tuning, LoRA). Importante para posicionar nosso uso do Gemma como **parameter-frozen** (zero/few-shot) vs. o baseline treinado por regras.

---

### C5. Detecção de Empatia com LLMs Multimodais

**Paper:** Thesis Proposal: Detecting Empathy Using Multimodal Language Model

**Fonte:** EACL 2024, Student Research Workshop

**Link:** [ACL Anthology](https://aclanthology.org/2024.eacl-srw.27.pdf)

**Resumo:** Proposta de tese que investiga detecção de empatia usando LLMs multimodais, citando diretamente o OMG-Empathy. Valida a relevância acadêmica do nosso estudo: aplicar LLMs ao mesmo dataset de empatia.

---

## D. Referência complementar

---

### D1. Survey: Detecção de Empatia por múltiplos sinais

**Paper:** Empathy Detection from Text, Audiovisual, Audio or Physiological Signals: Task Formulations and Machine Learning Methods

**Fonte:** arXiv, 2023 (v2)

**Link:** [2311.00721](https://arxiv.org/abs/2311.00721)

**Resumo:** Survey abrangente sobre detecção de empatia a partir de texto, audiovisual, áudio e sinais fisiológicos. Lista formulações de tarefa e métodos de ML. Cita o OMG-Empathy como base principal de valência contínua. Útil para o *Related Work* do paper.

---

### D2. Valence-Arousal Dimensional Model

**Paper (clássico):** A Circumplex Model of Affect

**Autores:** James A. Russell

**Fonte:** Journal of Personality and Social Psychology, 39(6), 1161–1178, 1980

**Resumo:** Modelo fundacional que define emoções em duas dimensões: **valência** (prazer/desprazer) e **arousal** (ativação/desativação). Base teórica do mapeamento blendshapes → V-A usado no módulo `face_blendshape` e na saída do LLM.

**BibTeX:**
```bibtex
@article{Russell1980Circumplex,
  author  = {Russell, James A.},
  title   = {A Circumplex Model of Affect},
  journal = {Journal of Personality and Social Psychology},
  volume  = {39},
  number  = {6},
  pages   = {1161--1178},
  year    = {1980}
}
```

---

### D3. Ekman Basic Emotions

**Paper (clássico):** An Argument for Basic Emotions

**Autores:** Paul Ekman

**Fonte:** Cognition & Emotion, 6(3-4), 169–200, 1992

**Resumo:** Define as 6 emoções básicas universais (happiness, sadness, anger, fear, disgust, surprise) que formam a **taxonomia categórica** usada como rótulo no CMU-MOSEI e como saída do LLM nas condições de classificação.

**BibTeX:**
```bibtex
@article{Ekman1992BasicEmotions,
  author  = {Ekman, Paul},
  title   = {An Argument for Basic Emotions},
  journal = {Cognition \& Emotion},
  volume  = {6},
  number  = {3-4},
  pages   = {169--200},
  year    = {1992}
}
```

---

### D4. Concordance Correlation Coefficient (CCC)

**Paper (clássico):** A Concordance Correlation Coefficient to Evaluate Reproducibility

**Autores:** Lawrence I-Kuei Lin

**Fonte:** Biometrics, 45(1), 255–268, 1989

**Links:**
- DOI: [10.2307/2532051](https://doi.org/10.2307/2532051)
- PubMed: [2720055](https://pubmed.ncbi.nlm.nih.gov/2720055/)

**Abstract:** A new reproducibility index is developed and studied. This index is the correlation between the two readings that fall on the 45° line through the origin. It is simple to use and possesses desirable properties.

**Resumo:** Define o coeficiente de concordância (CCC) como medida de reprodutibilidade que avalia simultaneamente **precisão** (correlação) e **acurácia** (desvio da identidade). Faixa de −1 a +1; CCC = 1 indica concordância perfeita. No nosso estudo, é a **métrica primária** para avaliar regressão de valência contínua no OMG-Empathy, seguindo o protocolo oficial do challenge.

**BibTeX:**
```bibtex
@article{Lin1989CCC,
  author  = {Lin, Lawrence I-Kuei},
  title   = {A Concordance Correlation Coefficient to Evaluate Reproducibility},
  journal = {Biometrics},
  volume  = {45},
  number  = {1},
  pages   = {255--268},
  year    = {1989},
  doi     = {10.2307/2532051}
}
```

---

### D5. FACET — Facial Action Coding System (features visuais do MOSEI)

**Fonte:** iMotions A/S

**Link:** [imotions.com/biosensor/facs-facial-action-coding-system](https://imotions.com/biosensor/facs-facial-action-coding-system/)

**Resumo:** FACET é um módulo de análise facial da iMotions baseado no Facial Action Coding System (FACS) de Ekman & Friesen. Extrai automaticamente **Action Units (AUs)** e probabilidades de emoções a partir de vídeo. No CMU-MOSEI, o arquivo `VisualFacet42.csd` contém 35 dimensões pré-extraídas (AUs + evidências de emoções) por frame, que servem como **features de entrada para a condição C2** do LLM e para o **baseline de Regressão Logística**.

---

### D6. Gemma 2 (ancestral denso)

**Paper:** Gemma 2: Improving Open Language Models at a Practical Size

**Autores:** Gemma Team, Google DeepMind

**Fonte:** arXiv preprint, 2024

**Link:** [2408.00118](https://arxiv.org/abs/2408.00118)

**Resumo:** Report técnico da geração anterior (densa) da família Gemma. Modelos de 2B a 27B parâmetros com atenção local-global intercalada, GQA e logit soft-capping. Citado aqui como referência complementar à evolução arquitetural até o Gemma 4 (MoE).

**BibTeX:**
```bibtex
@article{Gemma2Team2024,
  author  = {{Gemma Team, Google DeepMind}},
  title   = {Gemma 2: Improving Open Language Models at a Practical Size},
  journal = {arXiv preprint arXiv:2408.00118},
  year    = {2024}
}
```
