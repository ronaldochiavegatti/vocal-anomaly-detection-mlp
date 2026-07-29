# Requirements: Detecção de Anomalias Vocais (MLP em C)

**Defined:** 2026-07-27
**Core Value:** Fechar, com rigor metodológico comprovável por comparação A/B (mesma seed, mesmos 5-folds), os 3 gaps entre a implementação atual e a proposta PIBIC original — sem piorar o baseline de referência (Macro F1 0,4423 / Acurácia 69,4%).

## v1 Requirements

### Infraestrutura Estatística & Reprodutibilidade (INFRA) — pré-requisito, Fase 0

- [x] **INFRA-01**: `mode_train()` chama `metrics_bootstrap_ci()` e `metrics_mcnemar()` (já implementados em `metrics.c` mas hoje desconectados) sobre as predições out-of-fold agregadas, tornando o critério "adotar só se Macro F1 igual ou melhor" verificável estatisticamente
- [x] **INFRA-02**: Race condition no RNG global sob OpenMP (`precalculate_augmentations()`) corrigida — cada thread usa seu próprio stream de RNG (ou o laço deixa de ser paralelo), garantindo que `RANDOM_SEED=42` produza entradas idênticas entre execuções repetidas
- [x] **INFRA-03**: Baseline v29 reconfirmado com execução fresca, log salvo em `results/train_log_v29_baseline_reconfirmed.txt` (o log citado pelo SPEC.md está vazio); valores devem bater com `results/metrics_global.csv` (Macro F1 0,4423 / Acc 69,4%) ou a divergência é documentada explicitamente

### Gap 2 — Borderline-SMOTE (SMOTE) — Fase 1

- [x] **SMOTE-01**: `smote_oversample()` ganha parâmetro `SmoteMode {SMOTE_STANDARD, SMOTE_BORDERLINE}`; modo borderline usa `find_knn_global()` (k-NN entre todas as classes) para classificar cada amostra minoritária em safe/borderline/noise usando `2*m >= k` (não `m >= k/2` truncado em inteiro)
- [x] **SMOTE-02**: Interpolação dos sintéticos usa `find_knn()` (somente amostras da mesma classe) — nunca reaproveita a lista global de vizinhos usada para classificação, evitando ruído de rótulo cruzado
- [x] **SMOTE-03**: Fallback para pool borderline vazio (e o caso pré-existente `n_class <= 1`) é tratado com log de aviso explícito, sem crash nem geração degenerada
- [x] **SMOTE-04**: Relatório A/B produzido: tabela Macro F1 + F1 por classe (SMOTE padrão vs. borderline), mesma seed/folds, com McNemar/bootstrap CI (via INFRA-01); tabela de contagem safe/borderline/noise por classe/fold; decisão explícita de adoção/rejeição
- [x] **SMOTE-05**: `CLAUDE.md` atualizado com o resultado do Gap 2 (Optimization History / What Worked / What Didn't Work), independente do resultado

### Gap 3 — Comparação Redes Rasas × Profundas (ARCH) — Fase 2

- [x] **ARCH-01**: Rótulo de configuração do SPEC.md corrigido antes de qualquer relatório — a produção atual já é a config de 2 camadas ocultas `[128, 64]` (SPEC.md's "Config C"), não a rasa `[128]` ("Config A") assumida originalmente
- [x] **ARCH-02**: `mlp_init_dynamic()` generalizada para `mlp_init_multi()` com `hidden_sizes`/`dropout_rates` configuráveis; `Layer layers[MLP_NUM_LAYERS]` (tamanho fixo em tempo de compilação) alargado para `MLP_MAX_LAYERS` suficiente para a config mais profunda (D: 3 camadas ocultas), incluindo os buffers de tamanho fixo equivalentes em `mlp_train.c`/`mlp.c` (delta de backprop, checkpoint, BN, SWA)
- [x] **ARCH-03**: Laço de treino fold+vogal extraído para função reutilizável, permitindo treinar as 4 configs (rotulagem corrigida) sobre as mesmas partições de 5-fold
- [x] **ARCH-04**: Tabela comparativa das 4 configs (Acurácia, Macro F1, F1 por classe, nº parâmetros, tempo/época) com McNemar/bootstrap CI entre a melhor config e cada config mais simples — nunca decisão por inspeção visual
- [x] **ARCH-05**: Regra de decisão "menor complexidade não estatisticamente pior" aplicada e documentada explicitamente (regra 1-SE + McNemar)
- [x] **ARCH-06**: `CLAUDE.md` atualizado com o resultado do Gap 3, independente do resultado

### Gap 1 — Seleção Paraconsistente de Características (PARA) — Fase 3

- [x] **PARA-01**: Novo módulo `src/feature_select_paraconsistent.c` + `include/feature_select_paraconsistent.h` computando μ via estatística F da ANOVA (ou η²) — não a razão por classe do SPEC.md, matematicamente enviesada — e λ via dispersão normalizada pela variância global — não `CV = std/mean`, que explode para features de δMFCC com média ≈ zero (documentado em `CLAUDE.md`)
- [x] **PARA-02**: Cálculo de Gc/Gct com guarda de variância mínima (proteção contra divisão por zero, no padrão `MIN_STD` já usado em `normalize.c`) e laço de relaxamento do limiar com número máximo de iterações (evita loop não limitado quando zero features são selecionadas)
- [x] **PARA-03**: Seleção executada independentemente por (fold, vogal, rede) — Master (binário) e Expert (4 classes) recebem índices de features selecionadas distintos
- [x] **PARA-04**: Índices selecionados persistidos via `selected_save`/`selected_load` por (fold, vogal, rede); `predict_hierarchical_late_fusion()` e o bloco duplicado de slicing inline no loop de validação (`src/main.c`) atualizados de forma consistente para usá-los, evitando divergência entre a predição discreta e a probabilidade registrada
- [x] **PARA-05**: Relatório final inclui tabela completa (feature, μ, λ, Gc, Gct, selecionada S/N) agregada nas ~30 execuções (5 folds × 3 vogais × 2 redes) — tabela de frequência de seleção, não um snapshot de uma única execução — mais Macro F1 antes/depois
- [x] **PARA-06**: `CLAUDE.md` atualizado com o resultado do Gap 1, independente do resultado

### Transversal (CROSS)

- [x] **CROSS-01**: Tabela consolidada "Gap Adoption Status" (decisão / delta de métrica / status de citação por gap) produzida ao final, para uso direto no relatório/pôster PIBIC
- [x] **CROSS-02**: Nenhuma citação bibliográfica adicionada para técnica não efetivamente ativa no modelo final entregue (regra do próprio SPEC.md — erro já cometido uma vez com Borderline-SMOTE)

## v2 Requirements

Adiado explicitamente pelo usuário para depois do fechamento dos 3 gaps.

### Débito Técnico (achado pelo mapeamento de codebase, não pelo SPEC.md)

- **DEBT-01**: Corrigir magic number `float xv[251]` em `src/main.c:325` (deveria ser `nf_vowel`)
- **DEBT-02**: Corrigir memory leak em `mode_train()` (nunca chama `dataset_free`/`features_free`)
- **DEBT-03**: Atualizar `CLAUDE.md`/`MEMORY.md` para refletir a arquitetura real v29 (Hierarchical Late Fusion) em vez da descrição desatualizada (MLP flat 5-classes)
- **DEBT-04**: Remover ou documentar módulos mortos (`knn.c`, `logreg.c` como baselines não chamados) se não forem reconectados

### Além do escopo da defesa PIBIC atual

- **FUTURE-01**: Ablação fatorial completa (modo SMOTE × profundidade de rede × seleção de features)
- **FUTURE-02**: Validação externa dos limiares paraconsistentes em uma segunda base de dados de patologia vocal

## Out of Scope

| Item | Motivo |
|------|--------|
| Reaproveitar código WIP "nested stacked hierarchy" (v30/v31) | Colapsou para classe majoritária (Macro F1 ~0,15); SPEC.md determina partir do HEAD v29 íntegro |
| Reintroduzir ensemble averaging, SWA, focal loss, class weights fortes + SMOTE, Mixup, BatchNorm, wavelet denoising nas features iniciais | Já testados e descartados — documentados em `CLAUDE.md` como "o que não funcionou" |
| Corrigir bugs de débito técnico do mapeamento de codebase nesta fase | Pedido explícito do usuário: focar só no SPEC.md agora, bugs depois (ver v2 acima) |
| `make asan`/AddressSanitizer como target permanente de CI | Útil como verificação pontual antes do sweep do Gap 3, mas não é requisito de entrega — decisão de execução, não de escopo |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 0 | Complete |
| INFRA-02 | Phase 0 | Complete |
| INFRA-03 | Phase 0 | Complete |
| SMOTE-01 | Phase 1 | Complete |
| SMOTE-02 | Phase 1 | Complete |
| SMOTE-03 | Phase 1 | Complete |
| SMOTE-04 | Phase 1 | Complete |
| SMOTE-05 | Phase 1 | Complete |
| ARCH-01 | Phase 2 | Complete |
| ARCH-02 | Phase 2 | Complete |
| ARCH-03 | Phase 2 | Complete |
| ARCH-04 | Phase 2 | Complete |
| ARCH-05 | Phase 2 | Complete |
| ARCH-06 | Phase 2 | Complete |
| PARA-01 | Phase 3 | Complete |
| PARA-02 | Phase 3 | Complete |
| PARA-03 | Phase 3 | Complete |
| PARA-04 | Phase 3 | Complete |
| PARA-05 | Phase 3 | Complete |
| PARA-06 | Phase 3 | Complete |
| CROSS-01 | Phase 3 | Complete |
| CROSS-02 | Phase 3 | Complete |

**Coverage:**
- v1 requirements: 22 total
- Mapped to phases: 22
- Unmapped: 0 ✓

---
*Requirements defined: 2026-07-27*
*Last updated: 2026-07-29 — all v1 requirements complete, milestone closed (Plan 03-05)*
