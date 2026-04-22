# PRD: Deep Learning Architectures em C para Detecção de Anomalias Vocais

## Introduction

O pipeline atual (MLP em C puro, 80.6% accuracy, Macro F1=0.612) atingiu seu teto
arquitetural — features hand-crafted fixas, overfitting persistente (train 99% vs val 80%)
e 91 amostras de Disfonia que o SMOTE não consegue compensar. Este PRD define a
implementação de 4 arquiteturas de deep learning diretamente em C, substituindo
progressivamente as features hand-crafted por representações aprendidas do sinal bruto,
com suporte a GPU via OpenCL (compatível com GPUs AMD). Todas as arquiteturas são avaliadas sob o mesmo protocolo
de 5-fold CV estratificado do pipeline original.

---

## Goals

- Implementar CNN 1D sobre espectrograma Mel como substituto das features hand-crafted
- Implementar Bidirectional GRU para capturar dinâmica temporal dos frames de áudio
- Implementar mecanismo de Attention (self-attention leve) sobre sequências de frames
- Implementar pipeline de Transfer Learning: carregar pesos pré-treinados (VGGish/wav2vec)
  e fazer fine-tuning sobre o dataset SVD
- Superar 85% accuracy no 5-fold CV estratificado mantendo o mesmo protocolo de avaliação
- Melhorar Macro F1 geral (atual 0.612), especialmente Disfonia (atual F1=0.39)
- Gerar relatório comparativo entre todas as 4 arquiteturas + baseline MLP atual
- Suportar aceleração GPU via OpenCL para treino das arquiteturas mais pesadas (compatível com GPU AMD)

---

## User Stories

### US-001: Geração de Espectrograma Mel em C
**Description:** Como desenvolvedor, preciso extrair espectrogramas Mel dos arquivos WAV
diretamente em C para alimentar a CNN 1D sem depender de features hand-crafted.

**Acceptance Criteria:**
- [ ] Função `mel_spectrogram()` em `src/feature_mel.c` que recebe sinal PCM e retorna
      matriz `[n_frames × n_mels]` com n_mels=80, hop_length=160, win_length=400
- [ ] Aplica escala Mel usando banco de filtros triangulares (filterbank) calculado em C
- [ ] Aplica log(1 + S) para compressão dinâmica
- [ ] Caching do espectrograma em `results/mel_specs/` por paciente (formato binário .bin)
- [ ] Testes unitários: comparar saída com referência gerada por librosa (tolerância 1e-3)
- [ ] Tempo de extração para todos os 918 pacientes < 60s

### US-002: Arquitetura CNN 1D em C
**Description:** Como pesquisador, quero treinar uma CNN 1D sobre espectrogramas Mel para
aprender filtros espectrais discriminativos automaticamente, superando as features fixas.

**Acceptance Criteria:**
- [ ] Estrutura `CNN1D` em `include/cnn1d.h` com camadas: Conv1D → BN → ReLU → MaxPool
      (2 blocos) → GlobalAvgPool → Dense(64) → Softmax
- [ ] Implementar `conv1d_forward()` e `conv1d_backward()` com gradientes corretos
      (verificados por gradient check numérico, tolerância 1e-5)
- [ ] Implementar MaxPool1D e GlobalAveragePool1D com backward
- [ ] Integrar com Adam e cosine LR annealing já existentes
- [ ] Treino 5-fold CV completo em < 5 min com GPU AMD (OpenCL) ou < 30 min em CPU
- [ ] Resultado reportado: accuracy, Macro F1, confusion matrix por fold
- [ ] Ganho esperado: accuracy ≥ 84% no 5-fold CV

### US-003: Arquitetura Bidirectional GRU em C
**Description:** Como pesquisador, quero uma BiGRU que processe sequências de frames
MFCC/Mel para capturar padrões temporais (ex: instabilidade de Jitter ao longo do tempo).

**Acceptance Criteria:**
- [ ] Estrutura `BiGRU` em `include/bigru.h`: GRU forward + GRU backward concatenados
- [ ] Implementar `gru_cell_forward()` e `gru_cell_backward()` com BPTT (truncated, T=50)
- [ ] Concatenar estados finais forward+backward → Dense(3) → Softmax
- [ ] Gradientes verificados numericamente (tolerância 1e-4)
- [ ] Suporte a sequências de comprimento variável (padding + masking)
- [ ] Treino 5-fold CV completo em < 10 min com GPU ou < 45 min em CPU
- [ ] Ganho esperado: accuracy ≥ 83% no 5-fold CV (efeito cumulativo com CNN)

### US-004: Mecanismo de Self-Attention em C
**Description:** Como pesquisador, quero um mecanismo de atenção leve (single-head) sobre
a sequência de frames para que o modelo identifique automaticamente regiões vocais
discriminativas (ex: instabilidades específicas da Disfonia).

**Acceptance Criteria:**
- [ ] Implementar `self_attention_forward()` em `src/attention.c`: Q=K=V projetados de
      frames Mel, `Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V`
- [ ] Implementar `self_attention_backward()` com gradientes corretos
- [ ] Camada de atenção plugável após CNN 1D ou BiGRU (arquitetura CNN+Attention)
- [ ] Salvar mapa de atenção por sample em `results/attention_maps/` para análise
- [ ] Visualização dos attention weights (output em CSV por fold) para interpretabilidade
- [ ] Ganho esperado sobre CNN sozinha: +1–3% accuracy

### US-005: Pipeline de Transfer Learning (carga de pesos pré-treinados)
**Description:** Como pesquisador, quero carregar pesos pré-treinados do VGGish (ou
equivalente leve) em C e fazer fine-tuning sobre o dataset SVD para compensar o
gargalo de dados da classe Disfonia (91 amostras).

**Acceptance Criteria:**
- [ ] Script Python auxiliar `scripts/export_weights.py` exporta pesos do VGGish
      (PyTorch/HuggingFace) para formato binário `.bin` compatível com C
- [ ] Função `load_pretrained_weights()` em `src/transfer.c` carrega os pesos exportados
      na estrutura CNN1D existente
- [ ] Fine-tuning com learning rate diferenciado: camadas convolucionais lr×0.1,
      camada Dense lr×1.0 (discriminative learning rates)
- [ ] Opção de congelar (freeze) camadas convolucionais nas primeiras N épocas
- [ ] Ganho esperado: accuracy ≥ 87% — maior ganho individual (+10–15% vs baseline)
- [ ] Documentação do formato binário de pesos em `docs/weight_format.md`

### US-006: Suporte OpenCL para aceleração GPU AMD
**Description:** Como desenvolvedor, preciso que as operações matriciais pesadas
(Conv1D, GRU, Attention) sejam aceleradas via OpenCL para tornar o treino viável
na GPU AMD disponível localmente.

**Acceptance Criteria:**
- [ ] Arquivo `src/opencl_ops.c` + kernels em `src/kernels/` (`.cl`) para: matmul,
      conv1d, softmax, relu
- [ ] `Makefile` com target `make gpu` que compila com `gcc` e linka com `-lOpenCL`
- [ ] Flag `USE_OPENCL=1` em `config.h` habilita GPU AMD; `USE_OPENCL=0` fallback CPU
- [ ] Detecção automática do dispositivo OpenCL disponível (GPU AMD preferido sobre CPU)
- [ ] Transferência automática host↔device via `clEnqueueWriteBuffer` /
      `clEnqueueReadBuffer`
- [ ] Speedup medido: ≥ 4x vs CPU para CNN 1D; ≥ 2x para BiGRU
- [ ] Testado com ROCm OpenCL (amdgpu driver) ou Mesa Clover como fallback

### US-007: Avaliação comparativa e relatório final
**Description:** Como pesquisador, quero um relatório comparativo entre todas as
arquiteturas para documentar os ganhos de cada abordagem e justificar escolhas
metodológicas (ex: para publicação acadêmica).

**Acceptance Criteria:**
- [ ] Script `make compare` executa todos os modelos sequencialmente e gera
      `results/comparison_report.csv` com: modelo, accuracy, Macro F1, F1 por classe,
      tempo de treino, nº de parâmetros
- [ ] Inclui baseline MLP atual (80.6%) como linha de referência
- [ ] Relatório em `results/REPORT_DL.md` com tabela comparativa e análise textual
- [ ] Gráficos de confusion matrix por modelo salvos em `results/plots/` (formato PNG,
      gerados via script Python auxiliar `scripts/plot_results.py`)
- [ ] Curvas de aprendizado (loss/acc por época) exportadas para CSV por modelo

---

## Functional Requirements

- **FR-1:** Espectrograma Mel com parâmetros configuráveis em `config.h`:
  `MEL_N_MELS=80`, `MEL_HOP_LENGTH=160`, `MEL_WIN_LENGTH=400`, `MEL_SAMPLE_RATE=16000`
- **FR-2:** CNN 1D com 2 blocos Conv1D(64) + BN + ReLU + MaxPool seguidos de
  GlobalAvgPool → Dense(64) → Dense(3) → Softmax
- **FR-3:** BiGRU com hidden_size=128 (64 por direção), BPTT truncado em T=50 frames
- **FR-4:** Self-Attention single-head com d_k=64, sobre sequência de frames Mel
- **FR-5:** Todos os modelos usam o mesmo protocolo: 5-fold CV estratificado,
  normalização Z-score fit no treino aplicada no val, early stopping patience=30
- **FR-6:** Transfer Learning carrega pesos VGGish exportados via script Python,
  aplica discriminative learning rates (conv: lr×0.1, head: lr×1.0)
- **FR-7:** OpenCL compilável opcionalmente via `make gpu` (linka `-lOpenCL`); CPU fallback obrigatório
- **FR-8:** Todos os modelos salvam checkpoints em `models/<arch>_fold{k}.bin`
- **FR-9:** `results/comparison_report.csv` gerado automaticamente após `make compare`
- **FR-10:** Gradient check numérico implementado em `tests/test_gradients.c`
  para CNN e GRU antes de uso em produção

---

## Non-Goals

- Não implementar Transformer completo (multi-head, positional encoding completo) —
  apenas single-head attention leve
- Não implementar wav2vec 2.0 completo em C — apenas carregar pesos exportados (VGGish)
- Não criar interface gráfica ou dashboard de visualização
- Não fazer deploy do modelo para inferência em tempo real
- Não suportar GPUs NVIDIA (CUDA) — apenas AMD via OpenCL
- Não implementar data augmentation de áudio além do que já existe (Borderline-SMOTE
  continua disponível para comparação)
- Não alterar o protocolo de avaliação (5-fold CV estratificado) — comparabilidade
  com baseline é obrigatória

---

## Technical Considerations

### Estrutura de Arquivos Proposta
```
src/
  feature_mel.c       # Espectrograma Mel (novo)
  cnn1d.c             # CNN 1D forward/backward
  bigru.c             # Bidirectional GRU
  attention.c         # Self-Attention
  transfer.c          # Carregamento de pesos pré-treinados
  opencl_ops.c        # Interface OpenCL (opcional)
  kernels/            # Kernels OpenCL (.cl): matmul, conv1d, relu, softmax
include/
  cnn1d.h
  bigru.h
  attention.h
  transfer.h
scripts/
  export_weights.py   # Exporta VGGish → .bin
  plot_results.py     # Gera gráficos PNG
tests/
  test_gradients.c    # Gradient check numérico
results/
  mel_specs/          # Cache de espectrogramas
  attention_maps/     # Attention weights por sample
  plots/              # Confusion matrices, curvas
  comparison_report.csv
  REPORT_DL.md
```

### Dependências
- C99 + gcc com `-lm -lpthread`
- OpenCL ≥ 1.2 + `-lOpenCL` (opcional, via `USE_OPENCL=1`) — ROCm ou Mesa Clover
- Python 3.8+ com `torch` e `transformers` apenas para `scripts/export_weights.py`
  (não é dependência de runtime do C)

### Ordem de Implementação Recomendada
1. `feature_mel.c` — base para todos os modelos
2. `cnn1d.c` — arquitetura mais simples, valida o pipeline
3. `tests/test_gradients.c` — verificar corretude antes de continuar
4. `bigru.c` — mais complexo (BPTT), requer gradients corretos
5. `attention.c` — plugável sobre CNN ou GRU
6. `transfer.c` + `scripts/export_weights.py` — maior ganho esperado
7. `opencl_ops.c` + kernels `.cl` — otimização final para GPU AMD

### Riscos Técnicos
- **BPTT em GRU**: gradientes que explodem/somem — mitigar com gradient clipping (já
  implementado no pipeline atual, reusar `clip_gradients()`)
- **Conv1D backward**: implementação manual é error-prone — gradient check obrigatório
- **Formato de pesos VGGish**: dependente da versão do PyTorch/HuggingFace; fixar versão
  no `scripts/export_weights.py`
- **Memória GPU**: espectrogramas de 918 pacientes × 80 mels × ~300 frames ≈ 220MB —
  usar mini-batches de 16 para caber em GPU com 4GB VRAM

---

## Success Metrics

| Métrica                        | Baseline (MLP) | Meta         |
|-------------------------------|----------------|--------------|
| Accuracy (5-fold CV)          | 80.6%          | ≥ 85.0%      |
| Macro F1                       | 0.612          | ≥ 0.700      |
| F1 Disfonia                    | 0.39           | ≥ 0.55       |
| F1 Laryngite                   | 0.55           | ≥ 0.65       |
| Tempo treino CNN (GPU AMD)     | N/A            | < 5 min      |
| Gradient check CNN/GRU         | N/A            | Passa (1e-4) |

---

## Open Questions

1. **VGGish vs HuBERT**: VGGish tem arquitetura pública bem documentada e pesos menores
   (~72MB) — preferível para exportação manual. HuBERT seria mais poderoso mas mais
   complexo de exportar para C. Confirmar qual usar antes de implementar `transfer.c`.

2. **Tamanho da sequência para GRU**: pacientes com arquivos WAV de duração variável
   (de 1s a 10s+). Definir estratégia: truncar em T=300 frames ou padded batch?

3. **BatchNorm em CNN**: o modelo atual mostrou que BN prejudica em datasets pequenos
   (918 amostras). Avaliar se BN é benéfico no contexto de CNN 1D sobre Mel ou usar
   Layer Norm como alternativa.

4. **Protocolo de avaliação para Transfer Learning**: fine-tuning com pesos externos
   pode vazar informação se os pesos pré-treinados já "viram" dados parecidos. Documentar
   limitação metodológica no relatório final.
