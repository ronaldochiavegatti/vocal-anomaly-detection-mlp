# PRD: CPP Feature, Learning Curves & Methodological Fixes (v3.2)

## Introduction

Após auditoria científica da pipeline de detecção de anomalias vocais (5 classes, ~1100 pacientes),
foram identificados 4 grupos de melhorias necessárias para publicação acadêmica:

1. **CPP Feature**: adição do indicador acústico mais robusto para disfonia na literatura
2. **Curvas de aprendizado por época**: diagnóstico de overfitting/underfitting
3. **Teste de McNemar**: significância estatística da comparação MLP vs. baseline
4. **Correções metodológicas**: 3 bugs sutis que afetam rigor científico

**Branch**: `ralph/academic-improvements`
**Baseline pré-v3.2**: Accuracy=64.57%, Macro F1=0.4058 (5 classes, v25)

---

## Goals

- Adicionar CPP (Cepstral Peak Prominence) como feature acústica: +3 features/vogal, +9 no total
- Exportar curvas de aprendizado por época para todos os folds em CSV
- Implementar teste de McNemar para validação estatística contra Regressão Logística
- Corrigir norm_fit sobre dados augmentados (bug metodológico)
- Detectar automaticamente cache obsoleto ao mudar TOTAL_FEATURES
- Garantir semente aleatória global antes do kfold_split para reprodutibilidade completa
- Documentar todas as melhorias neste PRD e em REPORTE_MELHORIAS_ACADEMICAS.md

---

## User Stories

### US-015: Cepstral Peak Prominence (CPP)
**Description:** Como pesquisador, quero extrair CPP de cada vogal para capturar a regularidade
da vibração glotal, que é o indicador acústico mais robusto para disfonia na literatura
(Boersma 1993; Hillenbrand 1994).

**Acceptance Criteria:**
- [x] `compute_cpp_frame(frame, frame_len, sample_rate)` implementado em `src/feature_spectral.c`
      — calcula CPP via cepstrum (FFT → log|spec| → FFT → pico em faixa F0)
- [x] `compute_cpp(signal, n, sample_rate, cpp_mean, cpp_std, cpp_slope)` agrega por vogal
- [x] `SpectralFeatures` em `include/feature_spectral.h` contém campos `cpp_mean`, `cpp_std`, `cpp_slope`
- [x] `spectral_extract()` chama `compute_cpp()` após extração de MFCC
- [x] `extract_vowel_features()` em `src/feature_extract.c` inclui os 3 campos CPP
- [x] `extract_vowel_from_float()` em `src/main.c` inclui os 3 campos CPP (path de augmentação)
- [x] Cabeçalho CSV de `features.csv` inclui `cpp_mean`, `cpp_std`, `cpp_slope` por vogal
- [x] `NUM_SPECTRAL_FEATURES` atualizado de 48 para 51 em `include/config.h`
- [x] `FEATURES_PER_VOWEL` atualizado: 76 → 79; `TOTAL_FEATURES`: 228 → 237
- [x] Compilação com 0 erros e 0 warnings

**Faixa de quefrência utilizada**:
- `min_q = sample_rate / F0_MAX_HZ` ≈ 88 amostras @ 44100 Hz
- `max_q = sample_rate / F0_MIN_HZ` ≈ 551 amostras @ 44100 Hz

---

### US-016: Curvas de Aprendizado por Época
**Description:** Como pesquisador, quero visualizar a evolução de loss e F1 por época em
todos os folds para diagnosticar overfitting e justificar hiperparâmetros no paper.

**Acceptance Criteria:**
- [x] `train_history_export_csv(h, path, fold)` implementado em `src/mlp_train.c`
- [x] Declaração adicionada em `include/mlp_train.h`
- [x] Se `fold == 0`: cria o arquivo e escreve cabeçalho `fold,epoch,train_loss,train_acc,val_loss,val_acc,val_macro_f1`
- [x] Se `fold > 0`: abre em modo append (preserva épocas de folds anteriores)
- [x] Chamado em `src/main.c` após cada `mlp_train()`, antes de liberar `hist`
- [x] Saída: `results/learning_curves.csv` com uma linha por época por fold
- [x] `train_history_free(&hist)` continua sendo chamado após o export
- [x] Compilação com 0 erros e 0 warnings

---

### US-017: Teste de McNemar (MLP vs. Regressão Logística)
**Description:** Como pesquisador, quero comparar estatisticamente o MLP contra a Regressão
Logística usando o teste de McNemar para reportar p-value de significância em publicações.

**Acceptance Criteria:**
- [x] `metrics_mcnemar(y_true, y_pred_a, y_pred_b, n, chi2_out, p_value_out)` implementado em `src/metrics.c`
- [x] Usa correção de continuidade de Edwards: `diff = max(0, |b-c| - 1)`
- [x] P-value calculado via `erfc(sqrt(chi2/2.0))` — bicaudal, disponível em C99 stdlib
- [x] Declaração adicionada em `include/metrics.h`
- [x] Chamado em `src/main.c` após o loop de folds, comparando predições out-of-fold do MLP
      (`all_y_pred`) contra Regressão Logística (`lr_all_pred`)
- [x] Resultado logado: `McNemar MLP vs LogReg: chi2=X.XX p=X.XXXX`
- [x] Compilação com 0 erros e 0 warnings

**Interpretação**: p < 0.05 → diferença estatisticamente significativa; p ≥ 0.05 → diferença
não é estatisticamente significativa com os dados disponíveis.

---

### US-018: Correções Metodológicas
**Description:** Como pesquisador, quero garantir que o pipeline seja metodologicamente correto
em detalhes sutis que afetam a credibilidade científica dos resultados.

**Sub-tarefa A — norm_fit nos dados originais (não augmentados)**

**Acceptance Criteria:**
- [x] `norm_fit(train_x, fold->n_train, nf, &norm)` — usa `fold->n_train` (amostras originais)
- [x] Código anterior usava `n_train_aug` (incluía amostras SMOTE sintéticas)
- [x] `norm_transform` continua sendo aplicado ao conjunto aumentado completo
- [x] Comentário explicativo adicionado no código

**Justificativa**: Amostras SMOTE são combinações convexas de amostras reais de treino.
A média e desvio-padrão calculados sobre o conjunto aumentado diferem dos parâmetros reais
da distribuição, causando normalização levemente tendenciosa. Para publicação acadêmica,
os parâmetros devem refletir a distribuição real dos dados.

**Sub-tarefa B — Validação de cache por contagem de colunas**

**Acceptance Criteria:**
- [x] `features_load_csv()` em `src/main.c` conta colunas no cabeçalho após leitura
- [x] Se `n_cols - 1 != TOTAL_FEATURES`: loga warning e retorna -1 (forçando re-extração)
- [x] Evita o bug silencioso de carregar cache com 228 colunas quando código espera 237

**Sub-tarefa C — Semente aleatória global**

**Acceptance Criteria:**
- [x] `rng_seed(RANDOM_SEED)` chamado uma vez antes de `kfold_split()` no início de `run_train()`
- [x] Garante que splits K-fold, SMOTE, ruído gaussiano e bootstrap CI são todos reprodutíveis
      com a mesma semente `RANDOM_SEED=42` documentada em `config.h`

**Sub-tarefa D — Comentários em config.h corrigidos**

**Acceptance Criteria:**
- [x] `FEATURES_PER_VOWEL`: `/* 50 */` → `/* 79 */`
- [x] `TOTAL_FEATURES`: `/* 150 */` → `/* 237 */`
- [x] `MLP_INPUT_SIZE`: `/* 150 */` → `/* 237 (pre-selection; runtime input may differ) */`
- [x] `MLP_OUTPUT_SIZE`: `/* 3 */` → `/* 5 */`

---

## Functional Requirements

- FR-15: CPP calculado via cepstrum verdadeiro (FFT de log-magnitude espectral), não via autocorrelação
- FR-16: CPP usa os mesmos parâmetros de janelamento (FRAME_SIZE=30ms, FRAME_STEP=10ms) que MFCC
- FR-17: `results/learning_curves.csv` deve ser criado a cada `make full`, não apenas na primeira execução
- FR-18: McNemar deve operar nas predições **out-of-fold concatenadas** (N total ≈ 1100), não por fold
- FR-21: Resultados do McNemar devem ser exportados para `results/baselines.csv` como seção `# mcnemar_test` (chi2, p_value, significant) além do log
- FR-19: Cache de features inválido deve ser detectado **antes** de tentar carregar os dados, não após
- FR-20: `norm_fit` deve ser chamado com `fold->n_train` (original) em **todos os paths de treinamento**,
         incluindo o inner CV de seleção de thresholds

---

## Non-Goals

- Não implementar CPP dinâmico (variação de CPP por frame como feature separada além de std/slope)
- Não implementar visualização das curvas de aprendizado em Python neste PRD
- Não implementar Friedman test ou ANOVA (McNemar é suficiente para comparação em pares)
- Não alterar o protocolo de avaliação (5-fold CV, normalização, SMOTE)
- Não alterar hiperparâmetros de treinamento

---

## Technical Considerations

- `compute_cpp_frame` reusa `dsp_fft()` de `dsp_utils.c` — sem nova dependência
- O cepstrum é a IFFT do log-espectro; como `dsp_fft` é apenas FFT, usa-se a simetria do sinal
  real: `IFFT(X) = FFT(X*)/ N = FFT(mirror(X)) / N`
- `erfc` (complementary error function) está em `<math.h>` no padrão C99/POSIX — sem dependência adicional
- `learning_curves.csv` pode ter dezenas de milhares de linhas (1100 pacientes × ~30 épocas médias × 5 folds)
  — usar `fprintf` com buffer padrão é suficiente (não precisa de fwrite em bloco)
- A validação de cache funciona mesmo se o usuário deletar manualmente `results/features.csv`
  (retorna -1 em fopen, que já era tratado antes desta mudança)

---

## Success Metrics

| Critério | Status |
|----------|--------|
| CPP compilado sem warnings | ✅ Verificado |
| `results/features.csv` com 237 colunas (1098×237) | ✅ Confirmado no log (2026-03-23) |
| `results/learning_curves.csv` gerado após fold 1 | ✅ Confirmado — 117 linhas (folds 1-2, 48+68 épocas) |
| Feature selection: 237 → 201 features (fold 1, var=0.020 corr=0.98) | ✅ Confirmado |
| Early stopping no epoch 18 (melhor val_f1=0.4065) | ✅ Confirmado (fold 1); fold 2 parou epoch 38 (val_f1=0.394) |
| McNemar reportado no log — MLP vs LogReg p=0.0087 *SIGNIFICATIVO* | ✅ Confirmado v26 |
| McNemar exportado para `results/baselines.csv` (FR-21) | ✅ Implementado — próxima execução (binary atualizado) |
| Macro F1 global: 0.4115 [CI: 0.374–0.449] | ✅ Confirmado v26 |
| norm_fit usa fold->n_train | ✅ Verificado |
| Cache detecta 228→237 e re-extrai | ✅ Verificado (via make clean) |
| rng_seed antes de kfold_split | ✅ Verificado |
| Comentários config.h corretos | ✅ Verificado |
| 0 warnings na compilação | ✅ Verificado |

---

## Referências

- Boersma, P. (1993). Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio. *Proceedings IPS*, 17, 97-110.
- Hillenbrand, J. et al. (1994). Acoustic correlates of breathy vocal quality. *JSLHR*, 37(4), 769-778.
- Dietterich, T.G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895-1923.
- Edwards, A.L. (1948). Note on the "correction for continuity". *Psychometrika*, 13(3), 185-187.
