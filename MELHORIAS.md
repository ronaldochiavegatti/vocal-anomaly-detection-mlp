# MELHORIAS.md — Roadmap de Melhorias Acadêmicas

Pipeline atual: 80.6% acurácia, Macro F1=0.612, 5-fold CV estratificado.
Este documento descreve as melhorias planejadas para elevar o rigor científico e o desempenho do sistema.

---

## Melhoria 1 — Intervalos de Confiança + Repeated K-Fold

**Motivação**: Reportar apenas a média de um único 5-fold CV não é aceito em publicações científicas sem intervalos de confiança. Duas abordagens complementares:

### 1a. Bootstrap sobre as predições acumuladas
- Após o 5-fold, re-amostrar os `all_y_true` / `all_y_pred` com reposição N=1000 vezes
- Calcular acurácia, Macro F1 e F1 por classe em cada reamostragem
- Reportar IC 95% (percentis 2.5 e 97.5)
- **Custo**: ~10ms (sem re-treino), implementado em `metrics.c`

### 1b. Repeated K-Fold (10 × 5-fold)
- Repetir o 5-fold CV com 10 seeds distintas
- Reportar: `média ± std` de cada métrica
- Transforma "80.6%" em "80.6% ± 1.4%"
- **Custo**: 10× o tempo atual (~8 min)

### Arquivos a modificar
- `src/metrics.c` — adicionar `metrics_bootstrap_ci()`
- `src/main.c` — loop externo de repetições com seed variável
- `include/metrics.h` — struct `ConfidenceInterval { float mean, lower, upper; }`

---

## Melhoria 2 — Curvas ROC e AUC por Classe

**Motivação**: Para datasets desbalanceados (7.5:1.5:1), AUC é mais informativa que acurácia. Revisores de periódicos de biomedicina exigem ROC/AUC.

### O que implementar
- Salvar as probabilidades softmax (não só argmax) de cada sample de validação
- Computar curva ROC **one-vs-rest** para cada uma das 3 classes
- Calcular AUC via regra do trapézio
- Curva Precision-Recall por classe (mais informativa para classes minoritárias)
- Exportar para CSV: `results/roc_curves.csv`, `results/pr_curves.csv`

### Ponto de inserção no código
`src/main.c:499–511` — no loop de predições, além de `all_y_pred[i]`, salvar `all_y_prob[i][3]` (probabilidades softmax brutas de `mlp_forward`).

### Arquivos a modificar
- `src/metrics.c` — adicionar `metrics_roc_auc()`, `metrics_pr_curve()`
- `include/metrics.h` — `float auc[NUM_CLASSES]`
- `src/main.c` — alocar e salvar `all_y_prob`

---

## Melhoria 3 — Salvar Índices de Features Selecionadas por Fold

**Motivação**: Problema de reprodutibilidade. Atualmente, `models/mlp_fold*.bin` salva os pesos da rede, mas não quais das ~150 features foram selecionadas naquele fold. Sem isso, é impossível rodar inferência em um novo paciente sem re-extrair e re-selecionar features.

### O que implementar
- Salvar `models/selected_fold{k}.bin`: array `int selected[MAX_FEATURES]` + `int n_selected`
- Carregar no modo de inferência (futuro modo `predict`)
- Formato binário simples: `[int n_selected][int selected[0]]...[int selected[n-1]]`

### Arquivos a modificar
- `src/main.c` — após `select_features()`, salvar índices em arquivo binário
- Adicionar funções `selected_save()` / `selected_load()` (pode ser em `normalize.c` ou novo `feature_select.c`)

---

## Melhoria 4 — Feature Importance por Permutação

**Motivação**: Clinicamente, saber quais features (Jitter, F0, MFCCs, etc.) mais contribuem para cada diagnóstico é tão importante quanto a acurácia. Necessário para interpretabilidade e validação clínica.

### Algoritmo (Permutation Importance)
Para cada feature `j` (de 0 a `n_selected`):
1. Embaralhar a coluna `j` nos dados de validação (mantendo tudo mais igual)
2. Re-calcular acurácia e Macro F1
3. `importance[j] = accuracy_original - accuracy_permutada`
4. Features com maior queda = mais importantes

### Variante por grupo
Além de feature individual, permutação por grupo:
- Grupo Temporal (features 0–29)
- Grupo Espectral (features 30–95)
- Grupo Wavelet (features 96–149)
Permite comparar contribuição relativa de cada domínio físico.

### Arquivos a modificar / criar
- `src/metrics.c` — adicionar `metrics_permutation_importance()`
- `src/main.c` — chamar após avaliação do fold, exportar `results/feature_importance.csv`
- `include/metrics.h` — `float importance[MAX_FEATURES]`

---

## Melhoria 5 — Nested Cross-Validation (Rigor na Seleção de Hiperparâmetros)

**Motivação**: Os thresholds de feature selection (`var=0.01`, `corr=0.95`) são fixos. Se foram ajustados observando o resultado do CV, existe data leakage indireto nos hiperparâmetros. A solução é **nested CV**.

### Estrutura
```
Outer loop: 5-fold (avaliação imparcial)
  Inner loop: 3-fold (seleção de hiperparâmetros)
    Grid search: var_threshold ∈ {0.005, 0.01, 0.02}
                 corr_threshold ∈ {0.90, 0.95, 0.98}
```

- Cada outer fold usa os melhores hiperparâmetros encontrados no inner fold correspondente
- Aumenta custo em ~3× mas elimina bias de seleção de modelo

### Arquivos a modificar
- `src/main.c` — implementar `inner_cv_select_thresholds()`
- Refatorar `select_features()` para aceitar thresholds como parâmetros (já aceita via args)

---

## Melhoria 6 — Baselines para Comparação

**Motivação**: Qualquer paper precisa contextualizar os 80.6% contra métodos mais simples. Toda a infraestrutura já existe.

### Baselines a implementar

| Baseline | Descrição | Implementação |
|---|---|---|
| Majority class | Prediz sempre "Normal" (classe majoritária) | Trivial em `metrics.c` |
| Logistic Regression | MLP com 0 hidden layers (1 camada linear + softmax) | `MLP_HIDDEN_LAYERS=0` em `config.h` |
| kNN (k=5) | Classificação por vizinhança no espaço de features | ~50 linhas em novo `knn.c` |

A MLP já tem toda a infraestrutura — uma rede com 0 hidden layers é matematicamente equivalente a Regressão Logística Multinomial.

### Resultado esperado
Tabela comparativa para publicação:

| Método | Acurácia | Macro F1 |
|---|---|---|
| Majority class | ~74.8% | ~0.29 |
| Logistic Regression | ~? | ~? |
| kNN (k=5) | ~? | ~? |
| **MLP (proposto)** | **80.6%** | **0.612** |

---

## Melhoria 7 — Augmentação no Domínio do Áudio

**Motivação**: SMOTE interpola no espaço de features derivadas — uma aproximação. Augmentação no sinal de áudio gera amostras com variação física real (não estatística), o que é metodologicamente mais sólido.

### Técnicas a implementar em `src/wav_augment.c`

| Técnica | Parâmetro | Justificativa clínica |
|---|---|---|
| Adição de ruído branco | SNR = 20–30 dB | Simula variação de gravação |
| Pitch shifting | ±1–2 semitons | Preserva patologia, varia F0 |
| Time stretching | ±5–10% | Preserva espectro, varia duração |
| Gain perturbation | ±3 dB | Simula variação de microfone |

Aplicar **somente** às classes minoritárias (Laringite, Disfonia) e **somente** nos folds de treino (evitar data leakage).

### Experimento proposto
Ablation: comparar Borderline-SMOTE × Augmentação em áudio × combinação das duas.

### Arquivos a criar/modificar
- `src/wav_augment.c` + `include/wav_augment.h` — funções de augmentação no domínio temporal
- `src/main.c` — modo `augment`: gerar arquivos WAV aumentados antes da extração de features

---

## Sumário e Prioridade

| # | Melhoria | Impacto publicação | Impacto desempenho | Custo implementação |
|---|---|---|---|---|
| 1 | Intervalos de confiança + Repeated CV | **Essencial** | — | Baixo |
| 2 | ROC/AUC por classe | **Alto** | — | Baixo |
| 3 | Salvar features selecionadas por fold | Alto (reprodutibilidade) | — | Muito baixo |
| 4 | Feature importance (permutação) | **Alto** (interpretabilidade) | — | Médio |
| 5 | Nested cross-validation | Médio (rigor) | Pequeno | Alto |
| 6 | Baselines (LR, kNN) | **Alto** (contexto) | — | Médio |
| 7 | Augmentação em áudio | Alto (metodologia) | **Alto potencial** | Alto |

**Ordem sugerida de implementação**: 3 → 2 → 1 → 4 → 6 → 5 → 7
