---
gsd_plan_version: 1.0
type: quick
slug: cic-resumo-poster-update
created: 2026-07-29
mode: quick (inline execution — orchestrator holds full context)
---

# Quick Task: Atualizar resumo e pôster do XXXVIII CIC Unesp

## Objetivo

Atualizar e melhorar o resumo (`Cópia de Modelo-resumo-xxxviii-cic-unesp.docx`) e o
pôster (`Cópia de Modelo-poster-xxxviii-cic-unesp.pptx`) para refletir o estado real do
pipeline após o fechamento dos 3 gaps do SPEC.md (milestone v1.0, `.planning/STATE.md`
status: complete), mantendo cada peça em **no máximo 1 página**.

## Fonte de verdade dos números (nada digitado à mão)

Configuração adotada = Hierarchical Late Fusion + Borderline-SMOTE1 + Config C [128,64],
**sem** seleção paraconsistente. Artefatos:

| Número | Arquivo |
|---|---|
| Acurácia 0,696721 · Macro F1 0,458656 · P/R/F1 por classe | `results/metrics_global_borderline_C_baseline.csv` |
| IC 95% bootstrap (N=1000, seed=42) | `results/bootstrap_ci_borderline_C_baseline.csv` |
| McNemar vs 3 baselines | `results/mcnemar_vs_baselines_borderline_C_baseline.csv` |
| Matriz de confusão agregada (out-of-fold) | `results/train_log_v32_gap2_smote_ab_console.txt` linhas 875-889 |
| A/B Gap 2 (SMOTE padrão × Borderline) | `results/smote_ab_comparison.csv` |
| A/B Gap 3 (12 braços arch × regularização) | `results/arch_compare_comparison.csv` |
| A/B Gap 1 (paraconsistente on/off) | `results/paraconsistent_ab_comparison.csv` |
| Decisões adotar/rejeitar | `results/gap_adoption_status.csv` |

## Correções de fidelidade obrigatórias

1. **Remover a alegação "+5,2 p.p. vs arquitetura de camada única (64,2%)"** — é falsa.
   `CLAUDE.md` (correção ARCH-01) registra que Config A [128] **nunca** esteve em produção.
   A comparação real e reprodutível é a varredura do Gap 3: A/light = 0,4365 vs
   C/baseline = 0,4587 Macro F1, diferenças **não** significativas por McNemar.
2. **Substituir a alegação de ganho por evidência que existe**: MLP hierárquico supera os
   3 baselines com significância — McNemar p<0,001 vs MajorityClass, kNN e Regressão Logística.
3. **Trocar a citação de SMOTE** (Chawla 2002) por Borderline-SMOTE1 (Han/Wang/Mao 2005),
   que é o método efetivamente adotado; manter Chawla como origem do SMOTE base.
4. **Não citar bibliograficamente a seleção paraconsistente** — foi REJEITADA (Gap 1),
   e `PROJECT.md` lista "citar método não incorporado" como Out of Scope. Reportá-la
   apenas como resultado negativo em texto.
5. Atualizar arquitetura documentada: 2 camadas ocultas [128, 64], 251 variáveis totais,
   85 features por rede/vogal após fatiamento.

## Tarefas

- [ ] T1 — Regenerar figuras a partir dos CSVs (matriz de confusão, F1 por classe com IC,
      varredura de arquiteturas, A/B dos 3 gaps, diagrama do pipeline), paleta validada
      pelo `validate_palette.js` da skill dataviz
- [ ] T2 — Reescrever o resumo `.docx` preservando template/fontes/margens (Times New Roman),
      ≤ 1 página
- [ ] T3 — Redesenhar o pôster `.pptx` reaproveitando template e logos (unesp, ICT, PROPe,
      FAPESP, CNPq, QR), 1 slide A0 retrato, texto mínimo e diagramação ilustrada
- [ ] T4 — Verificar 1 página em ambos via render (LibreOffice → PDF) e conferir números
      contra os CSVs
- [ ] T5 — SUMMARY.md + tabela Quick Tasks em STATE.md + commits atômicos

## Restrições

- Não alterar código-fonte C nem resultados — tarefa é somente de documentação/entrega
- Preservar os arquivos originais (`.docx`/`.pptx`) versionados como backup antes de sobrescrever
- Referências bibliográficas: todas verificadas online (autores/volume/DOI) antes de entrar
