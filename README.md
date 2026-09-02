# Vocal Anomaly Detection — MLP in C

Pipeline de detecção de anomalias vocais implementada em **C puro** (C99, sem frameworks externos de ML). Classifica pacientes em **cinco classes** — **Normal**, **Laringite**, **Disfonia Psicogênica**, **Disfonia Funcional** e **Edema de Reinke** — a partir de gravações de vogais sustentadas (/a/, /i/, /u/).

Projeto desenvolvido como Iniciação Científica (PIBIC), apresentado no **XXXVIII CIC UNESP**:
- 📄 [Pôster](RonaldoChiavegatti_Poster_CIC.pdf)
- 📄 [Resumo](RonaldoChiavegatti_Resumo_CIC.pdf)
- 📋 [SPEC.md](SPEC.md) — mapeamento dos gaps entre a proposta original e a implementação, com critério de aceite A/B por gap

## Dataset

Os arquivos de áudio **não estão incluídos** neste repositório.

O projeto utiliza o **SVD (Saarbrücken Voice Database)**, banco de dados de referência internacional para análise de qualidade vocal. Os arquivos WAV são organizados nas pastas (~1098 pacientes válidos, forte desbalanceamento de classes):

```
saudavel/               # 687 pacientes — vozes saudáveis (SVD: normal)
laringite/               # 140 pacientes — laringite (SVD: laryngitis)
disfonia_psicogênica/    #  91 pacientes — disfonia psicogênica (SVD: dysphonia)
disfonia_funcional/      # 112 pacientes — disfonia funcional
edema_de_reinke/         #  68 pacientes — edema de Reinke
```

Também é necessário o arquivo de metadados `overview_merged.csv` na raiz do projeto.

Referência: Barry, W.J. & Pützer, M. (2007). *Saarbrücken Voice Database*. Institute of Phonetics, Saarland University. http://www.stimmdatenbank.coli.uni-saarland.de/

## Resultados (produção atual — Hierarchical Late Fusion)

5-fold cross-validation estratificado, `results/metrics_global.csv`:

| Métrica | Valor |
|---|---|
| Acurácia | **69.76%** |
| Macro F1 | **0.4514** |
| Weighted F1 | 0.6700 |

| Classe | Precision | Recall | F1 | N |
|---|---|---|---|---|
| Normal | 0.819 | 0.929 | 0.870 | 687 |
| Laringite | 0.478 | 0.314 | 0.379 | 140 |
| Disfonia Psicogênica | 0.325 | 0.275 | 0.298 | 91 |
| Disfonia Funcional | 0.340 | 0.152 | 0.210 | 112 |
| Edema de Reinke | 0.420 | 0.618 | 0.500 | 68 |

**Gargalo fundamental**: Disfonia Psicogênica e Disfonia Funcional são acusticamente quase indistinguíveis (ambas disfonias funcionais sem lesão estrutural, AUC one-vs-rest ≈ 0.62–0.64) — nenhuma mudança de arquitetura/hiperparâmetro rompe esse teto usando apenas features acústicas. Edema de Reinke (lesão estrutural) é a classe patológica mais separável.

### Gaps PIBIC — comparação A/B (mesma seed, mesmos 5-folds)

Todos os 3 gaps entre a proposta PIBIC original e a implementação foram fechados com comparação A/B reprodutível e decisão explícita adotar/rejeitar (`results/gap_adoption_status.csv`):

| Gap | Decisão | Δ Macro F1 | McNemar p |
|---|---|---|---|
| Borderline-SMOTE (vs. SMOTE padrão) | **ADOTADO** | +0.0235 | 0.6606 |
| Config C — 2 camadas ocultas [128,64] (vs. rasa) | **ADOTADO** (já em produção) | n/a | 0.7463 |
| Seleção Paraconsistente de features (LPA2v) | **REJEITADO** | −0.0024 | 0.8220 |

Detalhes completos de cada comparação em `CLAUDE.md` e nos logs `results/train_log_v3{2,3,4}_gap*.txt`.

## Arquitetura — Hierarchical Late Fusion

Em vez de um único classificador 5-classes, o problema é decomposto em dois estágios, replicados independentemente para cada vogal:

```
                 ┌─────────────┐
   /a/ ──────►   │   Master    │──► saudável / patológico
                 │ (binário)   │
                 └─────────────┘
                        │ se patológico
                        ▼
                 ┌─────────────┐
                 │   Expert    │──► Laringite / Disf. Psicog. /
                 │ (4 classes) │    Disf. Func. / Edema Reinke
                 └─────────────┘

  Repetido para /i/ e /u/ → 6 redes por fold (Master+Expert × 3 vogais)
  → fusão tardia por média de probabilidades entre as 3 vogais
```

Cada rede Master/Expert:

```
Input (85 features/vogal após seleção de variância+correlação)
  → Dense(128) + LeakyReLU + Dropout(0.5)
  → Dense(64)  + LeakyReLU + Dropout(0.4)
  → Dense(2 ou 4) + Softmax
```

**Features extraídas por vogal (/a/, /i/, /u/) — 83 por vogal, 251 no total (+2 metadados idade/sexo):**
- Temporais (10): Jitter (local/RAP/PPQ5), Shimmer (local/APQ3/5/11), energia, HNR, ZCR
- Espectrais (55): F0, formantes F1–F4, entropia espectral, centroide, rolloff, MFCC×13, δMFCC×13, δδMFCC×13 (desvio-padrão), CPP×3, fonte glótica (Oq/Sq/NAQ/H1-H2)×4
- Wavelet (18): DWT Daubechies-4, 6 níveis × {média, variância, energia}

**Pipeline de treinamento (por fold, por vogal, por rede):**
- Augmentation no domínio de áudio (ruído/gain/stretch) nas classes minoritárias
- Seleção de threshold via CV interna 3-fold (variância + correlação)
- Normalização Z-score (fit apenas nas amostras originais de treino)
- Borderline-SMOTE1 (Han/Wang/Mao 2005) para balancear classes minoritárias
- Adam (lr=0.001 → 0.00001, cosine annealing), L2=0.001, label smoothing=0.05
- Early stopping (patience=30 em Macro F1 de validação), gradient clipping (norm=5.0)
- 5-fold cross-validation estratificado, seed fixa (42) — reprodutível ponta a ponta

## Build & Execução

```bash
# Compilar
make

# Extrair features dos arquivos WAV (~42s com OpenMP, cacheia em results/features.csv)
make extract

# Treinar com 5-fold CV (~60-90 min com 237 features)
make train

# Pipeline completa (auto-carrega cache se results/features.csv existir)
make full
```

Requisitos: `gcc` (C99 + OpenMP), `make`, `libm`. `results/` e `models/` devem existir antes de rodar (`mkdir -p results models`).

## Estrutura

```
src/                        # 19 módulos .c
include/                    # 17 headers .h
results/                    # métricas, curvas de aprendizado, logs de treino por versão
tools/figures/               # geradores reproduzíveis das figuras do pôster/resumo
Makefile
CLAUDE.md                   # guia de arquitetura e histórico de otimização para Claude Code
SPEC.md                     # gaps PIBIC — proposta original vs. implementação atual
```
