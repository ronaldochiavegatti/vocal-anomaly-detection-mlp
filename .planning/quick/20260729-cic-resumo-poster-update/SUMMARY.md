---
gsd_summary_version: 1.0
type: quick
slug: cic-resumo-poster-update
status: complete
completed: 2026-07-29
commits:
  - ae91e71 docs(cic) — modelos originais versionados como baseline
  - 749d0c2 build(tools) — gerador reproduzível das figuras
  - b120720 docs(cic) — resumo e pôster atualizados
---

# Resumo da execução

Resumo (`.docx`) e pôster (`.pptx`) do XXXVIII CIC Unesp atualizados para a configuração
efetivamente adotada no fechamento do milestone v1.0, com todos os números lidos dos CSVs
de `results/` em vez de digitados.

## Configuração reportada

Hierarchical Late Fusion + Borderline-SMOTE1 + Config C `[128, 64]`, **sem** seleção
paraconsistente. Acurácia **69,7%** [IC 95%: 66,9–72,3%], Macro F1 **0,4587**
[0,419–0,495] — antes constavam 69,4% / 0,4435.

## Correções de fidelidade aplicadas

| Antes | Depois | Por quê |
|---|---|---|
| "+5,2 p.p. vs arquitetura de camada única (64,2%)" | Superioridade sobre os 3 baselines por McNemar (p < 0,001) + varredura real de 12 braços | A alegação era falsa: a correção ARCH-01 em `CLAUDE.md` registra que Config A `[128]` nunca esteve em produção |
| SMOTE citado como Chawla et al. 2002 | Han/Wang/Mao 2005 (Borderline-SMOTE1) | É o método de fato adotado no Gap 2 |
| Seleção paraconsistente ausente | Resultado negativo explícito, sem entrada bibliográfica | `PROJECT.md` lista citar método não incorporado como fora de escopo |
| Arquitetura implícita de 1 camada | 2 camadas ocultas `[128, 64]`, 251 variáveis | Estado real do código |
| Teto expresso como "AUC ≈ 0,62–0,64" | 37 dos 203 pacientes trocados entre si; melhor F1 de 0,36 e 0,27 nos 12 braços | O intervalo de AUC não é sustentado por nenhum artefato atual — a única medição (v26) deu 0,682/0,657, e em outra arquitetura |

## Verificações feitas

- **Uma página cada**: resumo 1 página A4 (2 colunas), pôster 1 slide A0 retrato
  (90 × 120 cm) — conferido por render `LibreOffice → PDF` e `pdfinfo`.
- **Matriz de confusão**: o parser escolhe, entre as várias matrizes do log de console,
  aquela cujos suportes por classe **e** acurácia conferem com o CSV — e o script falha
  por `assert` se nenhuma conferir.
- **Afirmações em prosa do pôster**: `load_claims()` recalcula cada número a partir dos
  CSVs e valida por `assert` (McNemar ainda p < 0,001; ainda 12 braços de arquitetura).
- **Paleta**: validada por `scripts/validate_palette.js` da skill dataviz (modo light,
  superfície `#fcfcfb`) — todos os checks PASS na dupla categórica e na rampa ordinal.
- **Referências**: cada uma conferida online (autoria, volume, DOI) antes de entrar; as
  seis do resumo são todas citadas no texto.
- **`pyflakes`** limpo nos três scripts.

## Bugs de template corrigidos de passagem

- `Tabela 1` do resumo cortava as colunas F1-Score e *n* para fora da coluna de texto: o
  `tblGrid` do modelo mantinha 3,4 cm por coluna e o `tblW`, 17 cm de largura, contra os
  8,22 cm reais da coluna.
- A figura do resumo estava sendo inserida com redução de escala, o que encolhia as
  fontes; passou a ser gerada no tamanho físico final (1:1).
- O pôster original tinha elementos além da margem direita (ex.: card em L 77,6 + 15,4 cm
  numa página de 90,01 cm); o novo layout resolve larguras a partir da largura útil.

## Pendências deixadas ao autor

- `SMOTE padrão` aparece sem citação no resumo (Chawla et al. 2002 saiu para caber em uma
  página). Se houver folga na diagramação final, vale reintroduzir.
- O diagrama do pipeline do pôster é vetorial e editável no PowerPoint — ajustes de texto
  não exigem regerar nada.
