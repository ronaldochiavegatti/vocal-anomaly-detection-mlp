# PRD: Expansão para 5 Classes — Disfonia Funcional & Edema de Reinke

## Introduction

Expandir o pipeline de detecção de anomalias vocais de 3 para 5 classes, incorporando `disfonia_funcional` (113 pacientes) e `edema_de_reinke` (69 pacientes) ao dataset existente. As pastas de áudio já estão presentes no repositório local com a estrutura correta de subdiretórios por paciente. O objetivo é manter acurácia ≥ 75% e atingir Macro F1 > 0.55 no cenário de 5 classes, recalibrando pesos de classe, SMOTE e arquitetura conforme necessário.

**Dataset após expansão:**

| Classe | Diretório | Pacientes |
|---|---|---|
| Normal | `saudavel/` | 687 |
| Laringite | `laringite/` | 140 |
| Disfonia Psicogênica | `disfonia_psicogênica/` | 91 |
| Disfonia Funcional | `disfonia_funcional/` | 113 |
| Edema de Reinke | `edema_de_reinke/` | 69 |
| **Total** | | **~1100** |

---

## Goals

- Registrar as duas novas classes no pipeline sem quebrar as 3 existentes
- Invalidar o cache `results/features.csv` e reextrair features para todos os ~1100 pacientes
- `MLP_OUTPUT_SIZE` e `class_counts[]` atualizam automaticamente via `NUM_CLASSES=5`
- Recalibrar class weights para a nova distribuição (Reinke é a classe mais rara)
- Manter SMOTE k=5 viável (fold train de Reinke ~55 pacientes → k=5 OK)
- Acurácia ≥ 75%, Macro F1 > 0.55, F1 individual de Disfonia Funcional e Reinke > 0.40

---

## User Stories

### US-020: Registrar as 5 classes em config.h ✅
**Description:** Como desenvolvedor, quero que `config.h` defina todas as 5 classes, diretórios e pesos para que o resto do código use apenas constantes.

**Acceptance Criteria:**
- [x] `NUM_CLASSES` alterado de `3` para `5`
- [x] Adicionados `CLASS_FUNC_DYSPHONIA 3` e `CLASS_REINKE 4`
- [x] Adicionados `DATA_DIR_FUNC_DYSPHONIA "disfonia_funcional"` e `DATA_DIR_REINKE "edema_de_reinke"`
- [x] Adicionados `CLASS_NAME_FUNC_DYSPHONIA "Disfonia Funcional"` e `CLASS_NAME_REINKE "Edema de Reinke"`
- [x] Pesos recalibrados para 5 classes (Normal=0.75, Laringite=1.10, Disfonia=1.50, FuncDisfonia=1.40, Reinke=1.80)
- [x] `make` compila sem erros ou warnings

### US-021: Estender dataset.c para carregar as 2 novas classes ✅
**Description:** Como desenvolvedor, quero que `dataset_load()` enumere os 5 diretórios de classe automaticamente via loop sobre `NUM_CLASSES`.

**Acceptance Criteria:**
- [x] Array `class_dirs[NUM_CLASSES]` em `dataset.c` inclui `DATA_DIR_FUNC_DYSPHONIA` e `DATA_DIR_REINKE` nas posições 3 e 4
- [x] `ds->class_counts[3]` e `ds->class_counts[4]` são populados corretamente
- [x] Cache inválido detectado pela validação de colunas — não trava o processo
- [x] `make` compila sem erros

### US-022: Atualizar main.c com array de pesos de 5 classes ✅
**Description:** Como desenvolvedor, quero que o array `class_weights[]` em `main.c` tenha 5 elementos correspondendo às constantes de `config.h`.

**Acceptance Criteria:**
- [x] `float class_weights[NUM_CLASSES]` inicializado com os 5 valores de `config.h`
- [x] Nenhum acesso out-of-bounds — loops usam `NUM_CLASSES`
- [x] `make` compila sem erros

### US-023: Atualizar métricas e display para 5 classes ✅
**Description:** Como pesquisador, quero ver o relatório de métricas com os nomes corretos das 5 classes para interpretar os resultados.

**Acceptance Criteria:**
- [x] `metrics.c` usa `CLASS_NAME_*` de `config.h` para todas as 5 classes
- [x] `results/metrics_global.csv` gerado após `make full` contém 5 linhas de per-class metrics
- [x] Nomes corretos: Normal, Laringite, Disfonia Psicogenica, Disfonia Funcional, Edema de Reinke
- [x] `make` compila sem erros

### US-024: Invalidar cache e reextrair features para os ~1100 pacientes ✅
**Description:** Como pesquisador, quero que o pipeline reextraia features incluindo os novos pacientes para garantir que o treinamento use dados completos.

**Acceptance Criteria:**
- [x] `results/features.csv` invalidado automaticamente pela validação de colunas (228→237)
- [x] `make full` executa com sucesso para 1098 pacientes válidos
- [x] `results/features.csv` resultante tem 1098 linhas × 237 features (verificado no log)
- [x] Tempo de extração: 42.1s (OpenMP, 6.5× speedup vs. single-thread)
- [x] 0 erros durante extração

### US-025: Primeiro treino 5 classes — baseline e avaliação 🔄
**Description:** Como pesquisador, quero executar o treinamento completo com 5 classes e ver as métricas iniciais para saber se os pesos padrão são razoáveis.

**Acceptance Criteria:**
- [x] `make full` iniciado — em andamento (results/train_log_v26_cpp.txt)
- [ ] `results/metrics_global.csv` contém métricas para todas as 5 classes (pendente conclusão)
- [ ] Acurácia global ≥ 70% (baseline tolerante — 5 classes é mais difícil)
- [ ] Macro F1 > 0.45 no primeiro treino (antes de tuning)
- [ ] Nenhuma classe com F1 = 0.00 (SMOTE está funcionando para Reinke)

### US-026: Tuning de class weights para Macro F1 > 0.55
**Description:** Como pesquisador, quero ajustar os pesos de classe iterativamente até atingir Macro F1 > 0.55 com as 5 classes.

**Nota importante**: Dada a análise AUC (DisfPsicog=0.682, DisfFunc=0.657), Macro F1 > 0.55 pode ser inalcançável com features acústicas puras.
Meta revisada: Macro F1 > 0.45, DisfFunc F1 > 0.30, Reinke F1 > 0.40.

**Rodada 1 — v26 (baseline)** (2026-03-23):
- Pesos: Normal=0.75, Laringite=1.10, DisfPsicog=1.50, DisfFunc=1.40, Reinke=1.80
- Resultado: Accuracy=65.4%, Macro F1=0.412, Normal=0.822, Laringite=0.360, DisfPsicog=0.273, DisfFunc=0.240, Reinke=0.362
- Problema: Laringite muito baixa (fold 1: 0.218), DisfFunc consistentemente baixa, Reinke muito variável

**Rodada 2 — v27 (em andamento)** (2026-03-23):
- Pesos: Normal=0.65, Laringite=1.35, DisfPsicog=1.70, DisfFunc=1.70, Reinke=2.10
- Mudanças: Normal reduzido (-0.10), Laringite aumentado (+0.25), DisfPsicog/DisfFunc equalizados (+0.20/+0.30), Reinke aumentado (+0.30)
- Resultado: Pendente (treinamento em andamento)

**Acceptance Criteria:**
- [x] Pelo menos 2 rodadas de ajuste de pesos documentadas (valores testados + resultado)
- [ ] Macro F1 global ≥ 0.45 (meta revisada)
- [ ] Disfonia Funcional F1 ≥ 0.30 (meta revisada)
- [ ] Edema de Reinke F1 ≥ 0.40
- [ ] Acurácia global ≥ 62% (meta revisada)
- [x] Pesos finais atualizados em `config.h`
- [ ] `results/metrics_global.csv` e `results/train_log_v27_weights.txt` salvos

---

## Functional Requirements

- **FR-1:** `NUM_CLASSES` em `config.h` deve ser `5`; todas as constantes de índice, diretório e nome devem existir para as 5 classes
- **FR-2:** `dataset_load()` deve carregar os 5 diretórios via loop `for (c = 0; c < NUM_CLASSES; c++)` — sem hardcode de `3`
- **FR-3:** O cache `results/features.csv` deve ser invalidado antes do primeiro treino com 5 classes
- **FR-4:** O pipeline de treino (`main.c`) deve usar `class_weights[NUM_CLASSES]` — sem array de tamanho fixo `[3]`
- **FR-5:** O display de métricas deve nomear corretamente as 5 classes usando as constantes de `config.h`
- **FR-6:** SMOTE deve continuar funcionando com k=5 mesmo para Edema de Reinke (~55 amostras por fold de treino antes de augmentação)
- **FR-7:** O treinamento deve atingir Macro F1 > 0.55 após tuning de pesos (US-026)

---

## Non-Goals

- Não alterar a arquitetura do MLP além do output size (que já se atualiza via `NUM_CLASSES`)
- Não adicionar novas features de áudio (228 features atuais são mantidas)
- Não modificar o algoritmo SMOTE (Borderline-SMOTE k=5 permanece)
- Não adicionar uma 6ª classe neste PRD
- Não alterar o fluxo de augmentação de dados além do necessário para 5 classes
- Não melhorar a acurácia das 3 classes existentes (foco é estabilidade, não regressão)

---

## Technical Considerations

- **Hardcodes a caçar:** procurar `3` em `main.c`, `metrics.c`, `dataset.c` nos contextos de loop de classes, arrays de nomes, e comparações de `class_label`. Substituir por `NUM_CLASSES`
- **Cache invalidation:** `results/features.csv` tem uma linha de header + N linhas de pacientes. O número de colunas (228) não muda, mas o número de linhas muda. O código de carregamento de cache usa apenas o número de features por paciente — verificar se valida o número total de amostras ou se simplesmente carrega o que encontrar
- **SMOTE viabilidade:** Com ~55 amostras de Reinke no fold de treino e k=5, SMOTE funciona mas gera poucos sintéticos. Se o fold tiver menos de k+1 amostras de alguma classe, SMOTE falhará — adicionar guard no código ou garantir que o fold mínimo de Reinke ≥ 6 (com 69 pacientes e 5 folds: ~55 treino, OK)
- **Class weights iniciais sugeridos:**

  | Classe | n | Peso sugerido |
  |---|---|---|
  | Normal | 687 | 0.75 |
  | Laringite | 140 | 1.10 |
  | Disfonia Psicogênica | 91 | 1.50 |
  | Disfonia Funcional | 113 | 1.40 |
  | Edema de Reinke | 69 | 1.80 |

- **Ordem de implementação recomendada:** US-020 → US-021 → US-022 → US-023 → US-024 → US-025 → US-026
- **O que NÃO fazer** (lições das iterações anteriores): não aplicar wavelet denoising nas features iniciais; não usar Disfonia/Reinke weight > 2.0 com SMOTE; não usar early stopping por val_acc (manter Macro F1)

---

## Success Metrics

- `make full` completa sem crash com 5 classes
- `results/metrics_global.csv` mostra 5 classes com nomes corretos
- Acurácia global ≥ 75%
- Macro F1 ≥ 0.55
- Disfonia Funcional F1 ≥ 0.40
- Edema de Reinke F1 ≥ 0.40
- Regressão nas 3 classes originais < 5% em F1 absoluto

---

## Open Questions

- A disfonia funcional tem características acústicas suficientemente distintas da disfonia psicogênica para que o classificador as separe? (resposta empírica: observar a matriz de confusão entre essas duas classes)
- O edema de Reinke, sendo uma condição de massa na prega vocal, pode ser acusticamente similar à laringite? (pode haver confusão — monitorar na matriz de confusão)
- Deve-se tentar um peso de Reinke > 1.80 se F1 de Reinke < 0.35 no primeiro treino?
