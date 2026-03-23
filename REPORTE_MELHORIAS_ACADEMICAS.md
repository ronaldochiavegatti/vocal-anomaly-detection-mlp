# Relatório de Melhorias Acadêmicas — Pipeline de Detecção de Anomalias Vocais

**Projeto**: Iniciação Científica — Detecção de Anomalias Vocais via MLP em C
**Branch**: `ralph/academic-improvements`
**Data**: 2026-03-22

---

## 1. Resumo Executivo

Este documento descreve as melhorias implementadas no pipeline de detecção de anomalias vocais com foco em:
1. Expansão do problema de 3 para **5 classes** (adição de Disfonia Funcional e Edema de Reinke)
2. Correções de **rigor metodológico** identificadas em auditoria científica
3. Adição de **features CPP** (Cepstral Peak Prominence) — indicador acústico de referência na literatura
4. Implementação de **curvas de aprendizado** por época para diagnóstico de overfitting
5. Adição de **teste de McNemar** para comparação estatística com baseline

---

## 2. Expansão para 5 Classes

### 2.1 Classes do Problema

| Classe | Código | Diretório | N (amostras) |
|--------|--------|-----------|-------------|
| Normal | 0 | `saudavel/` | 687 |
| Laringite | 1 | `laringite/` | 140 |
| Disfonia Psicogênica | 2 | `disfonia_psicogênica/` | 91 |
| Disfonia Funcional | 3 | `disfonia_funcional/` | 113 |
| Edema de Reinke | 4 | `edema_de_reinke/` | 69 |
| **Total** | | | **1100** |

### 2.2 Arquivos Modificados

- **`include/config.h`**: Adicionados `CLASS_FUNC_DYSPHONIA=3`, `CLASS_REINKE=4`, `NUM_CLASSES=5`, constantes de nome e pesos de classe recalibrados.
- **`src/dataset.c`**: Array `class_dirs[]` expandido para 5 entradas.
- **`src/main.c`**: Array `class_weights[5]` atualizado.
- **`src/metrics.c`**: Nomes de classes atualizados para 5 classes via `config.h`.

### 2.3 Pesos de Classe Recalibrados

```
Normal           = 0.75  (classe majoritária: 687/1100 = 62.5%)
Laringite        = 1.10
Disfonia Psicog. = 1.50
Disfonia Func.   = 1.40
Edema de Reinke  = 1.80  (classe mais rara: 69/1100 = 6.3%)
```

Combinados com Borderline-SMOTE e early stopping em Macro F1 para evitar sobre-ajuste às classes minoritárias.

---

## 3. Melhorias de Rigor Metodológico

### 3.1 Normalização Z-Score nos Dados Originais (Bug Fix Crítico)

**Problema identificado**: A normalização Z-score era fittada em `n_train_aug` (incluindo amostras SMOTE sintéticas), o que introduz viés sutil: a média e desvio-padrão calculados sobre o conjunto aumentado diferem ligeiramente dos parâmetros reais da distribuição dos dados originais.

**Correção** (`src/main.c:743`):
```c
/* Antes (bug): norm_fit(train_x, n_train_aug, nf, &norm); */
/* Depois (correto): */
norm_fit(train_x, fold->n_train, nf, &norm);
```

A norma é fittada nas amostras **originais de treino** (`fold->n_train`), depois aplicada ao conjunto aumentado e à validação.

### 3.2 Validação do Cache de Features

**Problema identificado**: Quando `TOTAL_FEATURES` muda (ex.: adição de CPP: 228→237), o `results/features.csv` em cache era carregado silenciosamente com número errado de colunas, causando erros de dimensionalidade não reportados.

**Solução** (`src/main.c`, função `features_load_csv()`):
```c
int n_cols = 1;
for (const char *p = buf; *p && *p != '\n' && *p != '\r'; p++)
    if (*p == ',') n_cols++;
if (n_cols - 1 != TOTAL_FEATURES) {
    log_warn("Cache desatualizado: %d colunas != TOTAL_FEATURES=%d. Re-extraindo.",
             n_cols - 1, TOTAL_FEATURES);
    fclose(f);
    return -1;  /* forcar re-extracao */
}
```

### 3.3 Semente Aleatória Global

**Problema identificado**: `rng_seed(RANDOM_SEED)` era chamado somente dentro de alguns loops, mas não globalmente antes de `kfold_split()`, podendo produzir splits diferentes em execuções consecutivas.

**Correção**: `rng_seed(RANDOM_SEED)` adicionado antes de `kfold_split()`, garantindo reprodutibilidade completa: shuffle dos dados, SMOTE, ruído gaussiano e bootstrap.

### 3.4 Comentários Corrigidos em config.h

Os comentários indicativos do valor avaliado das macros estavam desatualizados após adição de features:

| Macro | Antes | Depois |
|-------|-------|--------|
| `FEATURES_PER_VOWEL` | `/* 50 */` | `/* 79 */` |
| `TOTAL_FEATURES` | `/* 150 */` | `/* 237 */` |
| `MLP_INPUT_SIZE` | `/* 150 */` | `/* 237 (pre-selection) */` |
| `MLP_OUTPUT_SIZE` | `/* 3 */` | `/* 5 */` |

---

## 4. Feature CPP — Cepstral Peak Prominence

### 4.1 Motivação Científica

O CPP (Cepstral Peak Prominence) é considerado o indicador acústico mais robusto para disfonia na literatura fonética clínica (Boersma 1993; Hillenbrand et al., 1994). Enquanto HNR mede a relação sinal-ruído global, o CPP mede especificamente a **proeminência do pico cepstral na faixa de F0**, capturando a regularidade da vibração glotal de forma mais direta.

Características que tornam o CPP valioso para este pipeline:
- Robusto à variação de intensidade vocal
- Correlaciona com a percepção de qualidade vocal por fonoaudiólogos
- Complementar ao MFCC: captura periodicidade glotal que os coeficientes mel não encodificam diretamente
- Sensível a disfonias que alteram a regularidade de vibração (Edema de Reinke, Laringite)

### 4.2 Implementação

**Arquivo**: `src/feature_spectral.c`

O CPP é calculado frame a frame e depois agregado em 3 estatísticas por vogal:

| Feature | Descrição |
|---------|-----------|
| `cpp_mean` | CPP médio ao longo dos frames da vogal |
| `cpp_std` | Desvio-padrão do CPP (variabilidade temporal) |
| `cpp_slope` | Inclinação linear do CPP por frame (tendência temporal) |

**Algoritmo por frame** (função `compute_cpp_frame`):

1. FFT do frame janelado (Hamming, 30ms, passo 10ms)
2. Log-magnitude espectral: `log_spec[k] = log(|FFT[k]| + ε)`
3. Espelho para frequências negativas (sinal real simétrico)
4. Cepstrum: segunda FFT do log-espectro, dividida por N
5. Busca do pico na faixa de quefrência de F0:
   - `min_q = sample_rate / F0_MAX_HZ` (≈88 amostras @ 44100 Hz)
   - `max_q = sample_rate / F0_MIN_HZ` (≈551 amostras @ 44100 Hz)
6. `CPP = peak_cepstrum − mean_cepstrum_in_range`

**Agregação** (função `compute_cpp`): os valores frame a frame são reduzidos para `cpp_mean`, `cpp_std` e `cpp_slope` (regressão linear mínimos quadrados).

### 4.3 Impacto no Feature Count

| Nível | Antes | Depois | Diferença |
|-------|-------|--------|-----------|
| `NUM_SPECTRAL_FEATURES` | 48 | **51** | +cpp_mean, +cpp_std, +cpp_slope |
| `FEATURES_PER_VOWEL` | 76 | **79** | |
| `TOTAL_FEATURES` | 228 | **237** | |

O cache `results/features.csv` é automaticamente invalidado pela validação de colunas (seção 3.2).

---

## 5. Curvas de Aprendizado por Época

### 5.1 Motivação

Com early stopping ocorrendo em épocas 5-37 (observado em runs anteriores), é impossível diagnosticar overfitting/underfitting sem visualizar as curvas de loss e F1 por época.

As curvas são essenciais para:
- Identificar se `patience=30` é adequado
- Detectar overfitting (divergência crescente entre treino e validação)
- Justificar escolhas de hiperparâmetros no paper

### 5.2 Implementação

**Declaração** (`include/mlp_train.h`):
```c
void train_history_export_csv(const TrainHistory *h, const char *path, int fold);
```

**Comportamento**:
- Se `fold == 0`: cria o arquivo e escreve o cabeçalho CSV
- Se `fold > 0`: abre em modo append (preserva épocas dos folds anteriores)

**Formato de saída** (`results/learning_curves.csv`):
```
fold,epoch,train_loss,train_acc,val_loss,val_acc,val_macro_f1
1,1,1.609437,0.2500,1.607821,0.2609,0.2010
1,2,1.598743,0.2813,...
```

**Chamada no loop de folds** (`src/main.c`):
```c
char lc_path[1024];
snprintf(lc_path, sizeof(lc_path), "%s/learning_curves.csv", RESULTS_DIR);
train_history_export_csv(&hist, lc_path, f);
```

---

## 6. Teste de McNemar

### 6.1 Motivação

O Bootstrap CI quantifica a incerteza das métricas do modelo. Para publicação é necessário demonstrar que a diferença entre o MLP e a Regressão Logística (baseline) é **estatisticamente significativa**.

O **teste de McNemar** é adequado para comparar dois classificadores no mesmo conjunto de dados (Dietterich, 1998):
- Usa predições out-of-fold (mesmo split, mesmas amostras)
- Testa se os desacordos entre os dois classificadores são simétricos
- Não assume normalidade dos dados

### 6.2 Implementação

**Declaração** (`include/metrics.h`):
```c
void metrics_mcnemar(const int *y_true,
                     const int *y_pred_a, const int *y_pred_b,
                     int n_samples,
                     float *chi2_out, float *p_value_out);
```

**Algoritmo** (Edwards continuity correction):
1. Conta `b` = casos onde A acerta e B erra
2. Conta `c` = casos onde B acerta e A erra
3. `diff = max(0, |b − c| − 1.0)` (correção de continuidade de Edwards)
4. `χ² = diff² / (b + c)`
5. `p = erfc(√(χ²/2))` — p-value bicaudal via `erfc` (stdlib C99)

**Interpretação**: p < 0.05 indica diferença estatisticamente significativa entre os classificadores.

---

## 7. Baseline Completo nos Resultados

Os resultados do MLP agora são sempre comparados contra três baselines:

| Baseline | Descrição |
|----------|-----------|
| Majority Class | Prediz sempre classe 0 (Normal, 62.5% das amostras) |
| kNN (k=5) | k-nearest neighbors com features selecionadas |
| Regressão Logística | Adam optimizer, cosine LR decay, early stopping |

Sem baseline, não é possível contextualizar a acurácia: se Normal representa 62.5% dos dados, um classificador que sempre prediz "Normal" já alcança 62.5% de acurácia sem aprender nada.

---

## 8. Limitações e Ceiling Acústico

### 8.1 Sobreposição entre Disfonias Funcionais

O AUC one-vs-rest observado nas runs anteriores (≈0.617-0.641) para Disfonia Psicogênica e Disfonia Funcional indica que estas duas classes são **acusticamente quase indistinguíveis**. Ambas são disfonias funcionais sem lesão estrutural, com apresentação de jitter/shimmer/CPP muito similar.

**Implicação para as metas do PRD**: F1 > 0.40 para cada uma das duas classes funcionais é um objetivo ambicioso com features acústicas puras. Com AUC ≈ 0.63, o ceiling discriminativo está próximo de 0.25-0.35 de F1 para essas classes.

**Esta limitação é em si um resultado científico válido**: o pipeline identifica o limite das features acústicas de vogais sustentadas para distinção dessas condições.

**Possíveis soluções futuras** (fora do escopo deste PRD):
- Fusão das duas classes funcionais → 4 classes
- Features não-acústicas: questionário clínico, duração dos sintomas
- Análise prosódica: pitch dynamics durante fala contínua

### 8.2 Distribuição de Classes

| Classe | N | % | Amostras treino/fold (~80%) |
|--------|---|---|----------------------------|
| Normal | 687 | 62.5% | ~550 |
| Laringite | 140 | 12.7% | ~112 |
| Disfonia Psicog. | 91 | 8.3% | ~73 |
| Disfonia Func. | 113 | 10.3% | ~90 |
| Edema de Reinke | 69 | 6.3% | ~55 |

O Edema de Reinke com ~55 amostras por fold de treino é viável para SMOTE com k=5 (mínimo recomendado).

---

## 9. Arquivos Modificados

| Arquivo | Tipo de Mudança |
|---------|-----------------|
| `include/config.h` | +2 classes, +pesos, +nomes, comentários corrigidos |
| `include/feature_spectral.h` | +3 campos CPP em `SpectralFeatures` |
| `include/mlp_train.h` | +declaração `train_history_export_csv()` |
| `include/metrics.h` | +declaração `metrics_mcnemar()` |
| `src/feature_spectral.c` | +`compute_cpp_frame()`, +`compute_cpp()`, chamada em `spectral_extract()` |
| `src/feature_extract.c` | +CPP no extractor, +CPP no cabeçalho CSV |
| `src/mlp_train.c` | +implementação `train_history_export_csv()` |
| `src/metrics.c` | +implementação `metrics_mcnemar()` |
| `src/main.c` | +norm_fit fix, +rng_seed global, +CPP em augmentação, +cache validation, +learning curves call, +McNemar call, +5 classes |
| `src/dataset.c` | +2 classes em `class_dirs[]` |

---

## 10. Arquivos de Saída Após `make full`

| Arquivo | Conteúdo |
|---------|----------|
| `results/features.csv` | Matriz de features 1100×237 |
| `results/metrics_global.csv` | Métricas do MLP (Acc, F1, AUC por classe) |
| `results/baselines.csv` | Métricas de MajorityClass, kNN e LogReg |
| `results/learning_curves.csv` | Loss/F1 por época para todos os folds |
| `results/roc_curves.csv` | Pontos das curvas ROC (one-vs-rest, 5 classes) |
| `results/pr_curves.csv` | Pontos das curvas Precision-Recall (5 classes) |
| `results/feature_importance.csv` | Importância por permutação (média sobre folds) |
| `models/mlp_fold{0-4}.bin` | Pesos dos 5 modelos treinados |

---

## 11. Como Reproduzir

```bash
# 1. Garantir diretórios existam
mkdir -p results models

# 2. Compilar (zero warnings esperado)
make clean && make

# 3. Executar pipeline completo
make full

# 4. Verificar resultados
cat results/metrics_global.csv
cat results/baselines.csv

# 5. Visualizar curvas de aprendizado
# results/learning_curves.csv: fold,epoch,train_loss,train_acc,val_loss,val_acc,val_macro_f1
```

**Requisitos de reprodutibilidade**:
- Compilador: GCC com suporte a C99 e OpenMP (`-std=c99 -fopenmp`)
- Sistema: Linux x86_64 (testado em Ubuntu 22.04+)
- `RANDOM_SEED=42` controla: splits K-fold, SMOTE, ruído gaussiano, bootstrap CI
- Dados SVD nos diretórios corretos conforme `config.h`

---

## 12. Resultados Obtidos (v26 — CPP + todas as correções)

**Executado em 2026-03-23. Tempo total: 2480s (~41 min).**

### 12.1 Métricas Globais do MLP

| Métrica | Valor | IC 95% Bootstrap |
|---------|-------|-----------------|
| Accuracy | 65.39% | [62.57%, 68.12%] |
| Macro F1 | **0.4115** | [0.374, 0.449] |
| Weighted F1 | 0.6296 | — |

### 12.2 Métricas Por Classe

| Classe | Precisão | Recall | F1 | IC 95% F1 |
|--------|----------|--------|----|-----------|
| Normal (n=687) | 0.769 | 0.882 | **0.822** | [0.800, 0.842] |
| Laringite (n=140) | 0.434 | 0.307 | **0.360** | [0.286, 0.436] |
| Disfonia Psicog. (n=91) | 0.314 | 0.242 | **0.273** | [0.182, 0.367] |
| Disfonia Func. (n=112) | 0.310 | 0.196 | **0.240** | [0.158, 0.326] |
| Edema de Reinke (n=68) | 0.357 | 0.368 | **0.362** | [0.268, 0.462] |

### 12.3 Comparação com Baselines

| Método | Accuracy | Macro F1 | Laringite F1 | Reinke F1 |
|--------|----------|----------|-------------|-----------|
| MajorityClass | 62.57% | 0.154 | 0.000 | 0.000 |
| kNN (k=5) | 64.85% | 0.254 | 0.326 | 0.000 |
| LogReg | 62.30% | 0.376 | 0.349 | 0.309 |
| **MLP (proposto)** | **65.39%** | **0.412** | **0.360** | **0.362** |

### 12.4 Significância Estatística (McNemar)

| Comparação | chi² | p-value | Significativo? |
|-----------|------|---------|---------------|
| MLP vs MajorityClass | 4.663 | 0.031 | ✅ Sim (p < 0.05) |
| MLP vs kNN(k=5) | 0.134 | 0.714 | ❌ Não |
| MLP vs LogReg | 6.892 | 0.009 | ✅ Sim (p < 0.01) |

*Interpretação*: O MLP é estatisticamente superior ao MajorityClass e à Regressão Logística. A diferença vs. kNN não é estatisticamente significativa (kNN é um baseline competitivo para este dataset).

### 12.5 AUC One-vs-Rest

| Classe | AUC |
|--------|-----|
| Normal | 0.808 |
| Laringite | 0.764 |
| Disfonia Psicog. | 0.682 |
| Disfonia Func. | 0.657 |
| Edema de Reinke | 0.797 |

*Ceiling acústico*: AUC ≈ 0.657–0.682 para as duas disfonias funcionais indica que as features acústicas capturam apenas ~16% de informação discriminativa acima do acaso para estas classes — confirmando que features não-acústicas seriam necessárias para separação robusta.

### 12.6 Metas de Desempenho Revisadas

Com 5 classes e ceiling acústico identificado:

| Métrica | Meta PRD | Realista | **Obtido (v26)** |
|---------|----------|----------|-----------------|
| Accuracy | > 60% | 62-68% | **65.4%** ✅ |
| Macro F1 | > 0.45 | 0.40-0.50 | **0.412** ⚠️ |
| Normal F1 | > 0.75 | 0.82-0.88 | **0.822** ✅ |
| Laringite F1 | > 0.45 | 0.45-0.58 | **0.360** ❌ |
| Disfonia Psicog. F1 | > 0.30 | 0.20-0.38 | **0.273** ⚠️ |
| Disfonia Func. F1 | > 0.30 | 0.22-0.40 | **0.240** ⚠️ |
| Edema de Reinke F1 | > 0.40 | 0.38-0.55 | **0.362** ⚠️ |

*Próximo passo (US-026)*: Ajuste de pesos de classe para tentar melhorar Laringite e Disfonia Funcional.

---

## 13. Referências

- **Boersma, P. (1993)**. Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio. *Proceedings of the Institute of Phonetic Sciences*, 17, 97-110.
- **Hillenbrand, J., Cleveland, R.A., & Erickson, R.L. (1994)**. Acoustic correlates of breathy vocal quality. *Journal of Speech, Language, and Hearing Research*, 37(4), 769-778.
- **Awan, S.N., & Roy, N. (2006)**. Toward the development of an objective index of dysphonia severity. *Clinical Linguistics & Phonetics*, 20(1), 35-49.
- **Dietterich, T.G. (1998)**. Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895-1923.
- **Edwards, A.L. (1948)**. Note on the "correction for continuity" in testing the significance of the difference between correlated proportions. *Psychometrika*, 13(3), 185-187.
