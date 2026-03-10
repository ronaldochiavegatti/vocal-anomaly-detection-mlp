# TASK.md — Ralph Loop: Implementação das Melhorias Acadêmicas

## Objetivo
Implementar as melhorias do roadmap (MELHORIAS.md) em ordem de prioridade, uma por vez.
Cada iteração do loop deve: ler o estado atual, implementar a próxima melhoria pendente, compilar, verificar erros, e marcar como concluída.

## Estado Atual do Projeto
- Pipeline: 80.6% acurácia, Macro F1=0.612, 5-fold CV estratificado
- Linguagem: C puro, `gcc -O2 -Wall -Wextra -Wno-format-truncation -std=c99 -Iinclude -lm`
- Build: `make` → `build/vocal_detect`
- Arquivos principais: `src/main.c`, `src/metrics.c`, `include/metrics.h`, `include/config.h`

## Ordem de Implementação (MELHORIAS.md)
Seguir exatamente esta ordem:

1. [ ] **Melhoria 3** — Salvar índices de features selecionadas por fold (`models/selected_fold{k}.bin`)
2. [ ] **Melhoria 2** — Curvas ROC e AUC por classe (`results/roc_curves.csv`, `results/pr_curves.csv`)
3. [ ] **Melhoria 1** — Intervalos de confiança via Bootstrap (N=1000) sobre predições acumuladas
4. [ ] **Melhoria 4** — Feature importance por permutação (`results/feature_importance.csv`)
5. [ ] **Melhoria 6** — Baselines: Majority class + Logistic Regression (MLP 0 hidden) + kNN (k=5)
6. [ ] **Melhoria 5** — Nested cross-validation (outer 5-fold + inner 3-fold para thresholds)
7. [ ] **Melhoria 7** — Augmentação no domínio do áudio (`src/wav_augment.c`)

## Instruções por Iteração do Loop

### A cada iteração:
1. Ler este TASK.md para identificar a próxima melhoria `[ ]` não implementada
2. Ler os arquivos fonte relevantes antes de modificar qualquer coisa
3. Implementar a melhoria conforme especificado no MELHORIAS.md
4. Rodar `make` e corrigir todos os erros de compilação antes de prosseguir
5. Marcar a melhoria como `[x]` neste arquivo
6. Registrar o resultado no log abaixo
7. **Parar** — aguardar próxima iteração do loop

### Regras de implementação
- **Nunca** re-tentar: SWA, ensemble averaging, focal loss, BN, Mixup, post-hoc probability boosting
- Usar `mlp_init_dynamic(net, input_size)` para tamanhos variáveis de entrada
- `MLP_NUM_LAYERS=3` usa guard `#if` preprocessor — não alterar sem verificar mlp.c
- Preservar feature caching via `results/features.csv`
- Manter compatibilidade com `make train` (carrega CSV se presente)
- Não modificar hiperparâmetros já otimizados em `config.h` sem justificativa
- Após cada melhoria: rodar `make` e confirmar compilação limpa (0 erros, 0 warnings relevantes)

### Formato de saída esperado por melhoria
- **Melhoria 3**: `models/selected_fold{0-4}.bin` criados após `make train`
- **Melhoria 2**: `results/roc_curves.csv` e `results/pr_curves.csv` com colunas `class,threshold,tpr,fpr` e `class,threshold,precision,recall`
- **Melhoria 1**: seção adicional em `results/metrics_global.csv` com `metric,mean,ci_lower,ci_upper`
- **Melhoria 4**: `results/feature_importance.csv` com colunas `feature_idx,importance_acc,importance_f1`
- **Melhoria 6**: tabela comparativa impressa no stdout e salva em `results/baselines.csv`
- **Melhoria 5**: thresholds selecionados por fold impressos no log de treino
- **Melhoria 7**: `src/wav_augment.c` + `include/wav_augment.h` funcionais

## Log de Progresso

| Melhoria | Status | Data | Observações |
|---|---|---|---|
| 3 — Salvar features selecionadas | Pendente | — | — |
| 2 — ROC/AUC | Pendente | — | — |
| 1 — Bootstrap CI | Pendente | — | — |
| 4 — Feature importance | Pendente | — | — |
| 6 — Baselines | Pendente | — | — |
| 5 — Nested CV | Pendente | — | — |
| 7 — Audio augmentation | Pendente | — | — |

## Referências
- Detalhes de implementação: `MELHORIAS.md`
- Arquitetura e restrições: `CLAUDE.md`
- Histórico de resultados: `MEMORY.md`
