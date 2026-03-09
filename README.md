# Vocal Anomaly Detection — MLP in C

Pipeline de detecção de anomalias vocais implementada em C puro. Classifica pacientes em três classes — **Normal**, **Laringite** e **Disfonia Psicogênica** — a partir de gravações de vogais sustentadas.

## Dataset

Os arquivos de áudio **não estão incluídos** neste repositório.

O projeto utiliza o **SVD (Saarbrücken Voice Database)**, banco de dados de referência internacional para análise de qualidade vocal. Os arquivos WAV foram pré-processados (normalização de amplitude, recorte do segmento estável) e organizados nas pastas:

```
saudavel/               # 687 pacientes — vozes saudáveis (SVD: normal)
laringite/              # 140 pacientes — laringite (SVD: laryngitis)
disfonia_psicogênica/   #  91 pacientes — disfonia psicogênica (SVD: dysphonia)
```

Referência: Barry, W.J. & Pützer, M. (2007). *Saarbrücken Voice Database*. Institute of Phonetics, Saarland University. http://www.stimmdatenbank.coli.uni-saarland.de/

## Resultados (estado atual — v15/v20)

| Métrica | Valor |
|---|---|
| Acurácia (5-fold CV) | **80.6%** |
| Macro F1 | 0.612 |
| Weighted F1 | 0.789 |

| Classe | Precision | Recall | F1 | N |
|---|---|---|---|---|
| Normal | 0.85 | 0.94 | 0.89 | 687 |
| Laringite | 0.64 | 0.49 | 0.55 | 140 |
| Disfonia | 0.55 | 0.31 | 0.39 | 91 |

## Arquitetura

```
Input (~114 features após seleção)
  → Dense(128) + LeakyReLU + Dropout(0.5)
  → Dense(64)  + LeakyReLU + Dropout(0.4)
  → Dense(3)   + Softmax
```

**Features extraídas por vogal (A, I, U):**
- Temporais (10): Jitter, Shimmer, HNR, ZCR, energia
- Espectrais (22): F0, formantes F1–F4, MFCCs (13), entropia, centroide, rolloff
- Wavelet (18): DWT Daubechies-4, 6 níveis × {média, variância, energia}

**Total**: 3 vogais × 50 features = **150 features** → seleção por variância e correlação → ~114

**Pipeline de treinamento:**
- Normalização Z-score (fit no treino, transform no val)
- Borderline-SMOTE (balanceamento das classes minoritárias)
- Adam (lr=0.001 → 0.00001 cosine annealing), L2=0.003, label smoothing=0.05
- Early stopping (patience=30 em val_acc), gradient clipping (norm=5.0)
- 5-fold cross-validation estratificado

## Build & Execução

```bash
# Compilar
make

# Extrair features dos arquivos WAV (~5 min, cached em results/features.csv)
make extract

# Treinar com 5-fold CV (~50 s se features já extraídas)
make train

# Pipeline completa
make full
```

Requisitos: `gcc`, `make`, `libm` (padrão no Linux).

## Estrutura

```
src/            # 15 módulos .c
include/        # 15 headers .h
results/        # logs e métricas de treinamento
Makefile
CLAUDE.md       # guia para Claude Code
MELHORIAS.md    # roadmap de melhorias acadêmicas
```
