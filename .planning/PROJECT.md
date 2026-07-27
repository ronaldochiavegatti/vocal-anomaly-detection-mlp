# Detecção de Anomalias Vocais (MLP em C)

## What This Is

Pipeline de aprendizado de máquina em C99 puro (sem frameworks externos) que classifica
pacientes em 5 classes (Normal, Laringite, Disfonia Psicogênica, Disfonia Funcional,
Edema de Reinke) a partir de gravações de voz (base SVD, ~1098 pacientes). É o produto
de uma Iniciação Científica (PIBIC) e a arquitetura HEAD atual (v29, `Hierarchical Late
Fusion`, commit `e63483a`) usa um ensemble Master binário + Expert 4 classes, replicado
por vogal (/a/, /i/, /u/), com fusão tardia por média de probabilidades.

## Core Value

Fechar, com rigor metodológico comprovável por comparação A/B (mesma seed, mesmos
5-folds), os 3 gaps entre a implementação atual e a proposta PIBIC original — sem
piorar o baseline de referência (Macro F1 0,4423 / Acurácia 69,4%, `results/metrics_global.csv`).

## Requirements

### Validated

- ✓ Extração de features (temporal + espectral + wavelet + glotais), ~251 features/paciente, cache em `results/features.csv` — pipeline v29 existente
- ✓ Split estratificado 5-fold (`RANDOM_SEED=42`) — `src/kfold.c`
- ✓ Arquitetura Hierarchical Late Fusion (Master binário + Expert 4 classes, por vogal) — `src/main.c` `mode_train()`
- ✓ SMOTE padrão para balanceamento de classes minoritárias — `smote_oversample()` em `src/main.c`
- ✓ Augmentação de áudio (ruído/gain/stretch/pitch) pré-calculada para classes patológicas
- ✓ Baseline atual documentado e reproduzível: Macro F1 0,4423, Acurácia 69,4% (`results/metrics_global.csv`)

### Active

- [ ] **GAP 2** (prioridade média, implementar primeiro por ser isolado/baixo risco): Borderline-SMOTE (Han, Wang & Mao, 2005) substituindo SMOTE padrão — `smote_oversample_ex()` com detecção safe/borderline/noise via `find_knn_global()`
- [ ] **GAP 3** (prioridade média-baixa): Comparação Redes Rasas × Profundas — generalizar `mlp_init_dynamic()` para `mlp_init_multi()` (hidden_sizes configuráveis), testar 4 configs (A:[128], B:[64], C:[128,64], D:[128,64,32]) via nested CV, escolher a de menor complexidade estatisticamente não-inferior (McNemar/bootstrap CI)
- [ ] **GAP 1** (prioridade alta, mais citado na proposta PIBIC, implementar por último): Seleção Paraconsistente de Características (LPA2v — Da Costa/Abe) — novo módulo `feature_select_paraconsistent.c` computando (μ, λ, Gc, Gct) por feature, integrado independentemente nas redes Master e Expert por vogal

### Out of Scope

- Corrigir os bugs encontrados pelo mapeamento de codebase (race condition no RNG global sob OpenMP, divisão por zero em `mlp_evaluate()`, magic number `float xv[251]`, memory leak em `mode_train()`, drift entre `CLAUDE.md`/`MEMORY.md` e o código real v29) — adiado para depois de fechar os 3 gaps do SPEC.md, por pedido explícito do usuário
- Reaproveitar o código WIP "nested stacked hierarchy" (`results/train_log_v30.txt`/`v31*.txt`) — colapsou para classe majoritária (Macro F1 ~0,15); SPEC.md determina partir do HEAD v29 íntegro
- Reintroduzir técnicas já testadas e descartadas: ensemble averaging, SWA, focal loss, class weights fortes + SMOTE, Mixup, Batch Normalization, wavelet denoising nas features iniciais — documentadas como "o que não funcionou" em `CLAUDE.md`
- Adicionar citação bibliográfica de método não efetivamente incorporado na versão final (regra do próprio SPEC.md, para evitar erro já cometido uma vez)

## Context

- Base de dados SVD: 1098 pacientes válidos, 5 classes com desbalanceamento severo (687/140/91/112/68)
- Teto fundamental documentado: Disfonia Psicogênica vs Disfonia Funcional têm AUC one-vs-rest ≈ 0,62–0,64 (quase indistinguíveis acusticamente) — nenhuma mudança de arquitetura quebra esse teto usando apenas features acústicas
- `results/train_log_v30.txt`, `v31.txt`, `v31_intel.txt` são WIP de um experimento que falhou (mode "nested stacked hierarchy") e não devem ser reaproveitados
- Mapeamento de codebase (`.planning/codebase/`, commit `f6750c3`) confirmou que `CLAUDE.md`/memória do projeto descrevem uma arquitetura desatualizada (MLP flat 5-classes) — o HEAD real é a Hierarchical Late Fusion (v29); vários módulos compilados (kNN, LogReg, feature_select, bootstrap CI, McNemar, ROC/AUC) estão **mortos** no `mode_train()` atual
- Cada gap tem critério de aceite explícito em `SPEC.md`: rodar 5-fold com/sem a mudança, mesma seed, comparar Macro F1 e F1 por classe; só adotar em definitivo se não piorar

## Constraints

- **Metodológico**: Nenhuma mudança é incorporada sem comparação A/B reprodutível (mesma seed `RANDOM_SEED=42`, mesmos 5-folds), log salvo em `results/train_log_vXX_<nome-do-gap>.txt` — exigência do SPEC.md para rigor acadêmico perante a banca PIBIC
- **Regressão**: Reverter ou manter apenas como experimento documentado qualquer mudança que derrube o Macro F1 global abaixo de 0,42
- **Tech stack**: C99 puro, sem dependências externas de ML — `gcc -O2 -Wall -Wextra -Wno-format-truncation -std=c99 -Iinclude -fopenmp -lm`
- **Ordem de implementação**: Gap 2 → Gap 3 → Gap 1 (do SPEC.md — Gap 2 valida o fluxo de comparação A/B com baixo risco; Gap 1 é o mais complexo e deve incorporar as melhores config/modo já validados nos passos anteriores)
- **Documentação**: Atualizar `CLAUDE.md` (Optimization History / What Worked / What Didn't Work) ao final de cada gap, independentemente do resultado

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Adiar correção dos bugs achados no mapeamento (race condition RNG, div/0 em mlp_evaluate, etc.) | Usuário pediu focar só no SPEC.md agora; bugs entram depois | — Pending |
| Ordem Gap 2 → Gap 3 → Gap 1 | Definida explicitamente em SPEC.md: menor risco primeiro, gap mais citado na proposta por último para incorporar melhores configs já validadas | — Pending |
| Baseline de referência = v29 (HEAD atual, `e63483a`) | SPEC.md exige partir do HEAD íntegro, não do WIP quebrado v30/v31 | ✓ Good |

## Evolution

Este documento evolui em transições de fase e marcos de milestone.

**Após cada transição de fase** (via `/gsd-transition`):
1. Requisitos invalidados? → Mover para Out of Scope com motivo
2. Requisitos validados? → Mover para Validated com referência da fase
3. Novos requisitos emergiram? → Adicionar em Active
4. Decisões a registrar? → Adicionar em Key Decisions
5. "What This Is" ainda preciso? → Atualizar se desatualizado

**Após cada milestone** (via `/gsd:complete-milestone`):
1. Revisão completa de todas as seções
2. Checagem do Core Value — ainda a prioridade certa?
3. Auditar Out of Scope — motivos ainda válidos?
4. Atualizar Context com estado atual

---
*Last updated: 2026-07-27 after initialization*
