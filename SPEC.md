# SPEC.md — Aderência da Implementação à Proposta PIBIC 2025

## Contexto

A proposta de Iniciação Científica original especifica três elementos metodológicos que
**não estão implementados** na arquitetura atual (v29, `Hierarchical Late Fusion`,
commit `e63483a`, branch `ralph/academic-improvements`):

| # | Gap | Citado na proposta | Estado atual do código |
|---|-----|---------------------|--------------------------|
| 1 | Seleção paraconsistente de características | Sim, com referências próprias [12][13] | Nenhuma seleção de atributos é aplicada — todas as 251 (85/vogal) features seguem para o treino |
| 2 | Borderline-SMOTE | Sim, citada nos documentos do projeto (CLAUDE.md) | `smote_oversample()` em `src/main.c` implementa SMOTE padrão (Chawla et al., 2002), sem detecção de amostras de borda |
| 3 | Comparação entre redes rasas e profundas | Sim, com escolha da config. de menor complexidade | Apenas uma arquitetura MLP rasa (`Dense(128)`, 1 camada oculta) foi testada nas redes Mestra/Especialista |

Cada gap é especificado abaixo com rigor técnico suficiente para implementação direta,
incluindo algoritmo, assinatura de função, arquivos afetados e critério de aceite.
**Todo trabalho deve partir do HEAD atual (v29, íntegro) — não do WIP quebrado em
`results/train_log_v30.txt`/`v31*.txt` (arquitetura "nested stacked hierarchy", que
colapsou para a classe majoritária e não deve ser reaproveitada).**

Regra geral de aceite: **nenhuma mudança deste spec deve ser incorporada em definitivo
se piorar o Macro F1 global de referência (0,4435) ou a acurácia (69,4%)** obtidos no
5-fold CV atual. Toda mudança é validada por comparação A/B com os mesmos folds
(`RANDOM_SEED=42`), nunca por inspeção visual dos resultados.

---

## GAP 1 (Prioridade ALTA) — Seleção Paraconsistente de Características

### Por quê é prioridade alta
É o único gap citado com **referências bibliográficas dedicadas** na proposta original
([12] Avron/Arieli/Zamansky — *Theory of Effective Propositional Paraconsistent
Logics*; [13] Abe — *Paraconsistent Intelligent-Based Systems*). Uma banca
avaliadora tende a perguntar especificamente sobre isso.

### Fundamentação técnica
A Lógica Paraconsistente Anotada (LPA2v), usada em Análise Paraconsistente de
Evidências, atribui a cada evidência (aqui, cada *feature*) dois graus extraídos dos
dados:

- **Grau de crença/favorável (μ)** ∈ [0,1] — evidência de que a feature separa bem as
  classes.
- **Grau de descrença/contrário (λ)** ∈ [0,1] — evidência em contrário (ambiguidade,
  sobreposição entre classes).

A partir de (μ, λ) calculam-se:

```
Gc  = μ − λ                    (Grau de Certeza,       ∈ [-1, 1])
Gct = μ + λ − 1                (Grau de Contradição,    ∈ [-1, 1])
```

O plano (Gct, Gc) é dividido em 12 regiões (reticulado paraconsistente clássico de
Da Costa/Abe); a região de interesse para seleção é a de **certeza verdadeira**
(Gc alto, |Gct| baixo). Features caem em "falso", "inconsistente" ou "indeterminado"
são descartadas.

### Como derivar μ e λ das features acústicas (proposta de implementação)

Para uma feature `j` com `C` classes:

```
μ_j  = separabilidade_normalizada(j)   // ex.: razão de Fisher one-vs-rest, normalizada em [0,1]
λ_j  = 1 − consistencia_intra_classe(j) // ex.: 1 − (1 / (1 + CV médio dentro de cada classe))
```

Onde a razão de Fisher one-vs-rest para a classe `c` é:

```
F_c(j) = (média_c(j) − média_global(j))²  /  variância_c(j)
```

e `μ_j = média sobre c de F_c(j)`, normalizada por min-max sobre todas as `j` features
do conjunto de treino do fold corrente (**calculado só com dados de treino, nunca com
validação/teste** — mesma regra de `norm_fit` já documentada em CLAUDE.md).

### Especificação de implementação

**Novo arquivo**: `src/feature_select_paraconsistent.c` + header
`include/feature_select_paraconsistent.h`

```c
/* Calcula (mu, lambda, Gc, Gct) por feature e retorna os indices selecionados.
 * x: matriz de treino (n x nf), y: rotulos (n), nf: n. de features, n_classes: classes.
 * gc_thresh: limiar minimo de Grau de Certeza para aceitar a feature (ex.: 0.3-0.5).
 * gct_max: |Gct| maximo tolerado (ex.: 0.3) - controla quanto de contradicao aceitar.
 * selected: buffer de saida (tamanho >= nf), retorna numero de features selecionadas.
 */
int paraconsistent_select(const float *x, const int *y, int n, int nf, int n_classes,
                           float gc_thresh, float gct_max, int *selected);
```

Implementação interna:
1. Para cada feature `j`: computar média/variância por classe (reaproveitar padrão já
   usado em `select_features_variance`, ver WIP `src/main.c` linhas ~20-47, como
   referência de estilo, não de lógica).
2. Computar `mu_j` (Fisher one-vs-rest normalizado) e `lambda_j` (1 − consistência
   intra-classe normalizada).
3. Computar `Gc_j = mu_j - lambda_j`, `Gct_j = mu_j + lambda_j - 1`.
4. Selecionar `j` se `Gc_j >= gc_thresh` **e** `fabs(Gct_j) <= gct_max`.
5. Se nenhuma feature sobrar, relaxar `gc_thresh` em 0.05 e repetir (evita selecionar
   zero features) — logar um aviso quando isso ocorrer.

### Integração no pipeline

Em `src/main.c`, dentro do loop de fold, **antes** de `mlp_init_dynamic(&net_master[v], ...)`
e `mlp_init_dynamic(&net_expert[v], ...)`:

```c
int sel_m[FEATURES_PER_VOWEL + NUM_METADATA_FEATURES];
int ns_m = paraconsistent_select(tr_x_v, tr_y_bin, n_train_aug, nf_vowel, 2, 0.35f, 0.3f, sel_m);
/* aplicar sel_m a tr_x_v e vl_x_v antes de treinar net_master[v] */
```

Repetir de forma independente para o Expert (usa `ex_tr_y`/4 classes) — **os índices
selecionados para Mestra e Especialista podem (e devem) ser diferentes**, já que
resolvem problemas distintos (binário vs. 4 classes).

Persistir os índices selecionados por fold/rede em `models/selected_master_fold{k}_v{vogal}.bin`
e `models/selected_expert_fold{k}_v{vogal}.bin` (reaproveitar `selected_save`/`selected_load`
já existentes em `src/feature_select.c`).

### Critério de aceite
- Rodar 5-fold com e sem seleção paraconsistente, mesma seed.
- Reportar: nº de features retidas por rede/vogal, Macro F1 global, F1 por classe.
- **Só substituir a versão sem seleção se**: Macro F1 igual ou superior, **ou** Macro F1
  ligeiramente inferior (até −0.01) mas com redução de features ≥30% (trade-off
  interpretabilidade/parcimônia aceitável — documentar a escolha explicitamente).
- Adicionar ao relatório final uma tabela: feature, μ, λ, Gc, Gct, selecionada (S/N) —
  isso também alimenta uma resposta pronta para a banca sobre quais features o método
  considerou mais relevantes.

---

## GAP 2 (Prioridade MÉDIA) — Borderline-SMOTE

### Por quê é prioridade média
Mudança contida a uma função (`smote_oversample`), baixo risco de regressão, mas não é
o item mais cobrado pela proposta (SMOTE já é citado como técnica geral de balanceamento;
"Borderline" é uma variante específica mencionada apenas na documentação interna do
projeto, não na proposta PIBIC original em si — priorizar Gap 1 e Gap 3 primeiro se o
tempo for escasso).

### Fundamentação técnica (Han, Wang & Mao, 2005)
Borderline-SMOTE só gera amostras sintéticas a partir de exemplos da classe minoritária
classificados como **"borderline"** (próximos à fronteira de decisão), definidos por:

Para uma amostra `x_i` da classe minoritária, sejam seus `k` vizinhos mais próximos
(considerando **todas** as classes, não só a minoritária). Seja `m` = número desses `k`
vizinhos que pertence a classes majoritárias:

```
se      m == k:        "noise"   → descartar (não gera sintético)
se  k/2 <= m <  k:      "borderline" (danger) → gerar sintéticos
se      m <  k/2:       "safe"    → não gerar (já bem representada)
```

### Especificação de implementação

Modificar `smote_oversample()` em `src/main.c` (assinatura pode manter-se igual,
adicionando um parâmetro de modo para permitir comparação A/B):

```c
typedef enum { SMOTE_STANDARD = 0, SMOTE_BORDERLINE = 1 } SmoteMode;

static void smote_oversample_ex(const float *x_in, const int *y_in, int n_in, int nf,
                                  int num_classes, SmoteMode mode,
                                  float **x_out, int **y_out, int *n_out);
```

Alterações no corpo da função:
1. Antes do loop de geração por classe, para `mode == SMOTE_BORDERLINE`: para cada
   amostra da classe `c`, chamar uma nova `find_knn_global()` (variante de `find_knn`
   já existente, mas buscando entre **todas** as amostras `x_in`, não apenas
   `class_idx[c]`) com `k=5`, contar `m` = vizinhos de classe ≠ `c`.
2. Classificar em safe/borderline/noise conforme a regra acima; manter em
   `class_idx_borderline[c]` apenas os índices "borderline".
3. Se `class_idx_borderline[c]` ficar vazio (caso raro, classe muito isolada), cair de
   volta para todos os `class_idx[c]` (mesmo comportamento do SMOTE padrão) — logar
   aviso.
4. O laço de geração de sintéticos (`find_knn` + interpolação com `alpha`) passa a
   sortear `base_idx` apenas dentre `class_idx_borderline[c]` em vez de `class_idx[c]`;
   os vizinhos para interpolação continuam sendo buscados **dentro da própria classe**
   (`find_knn(x_in, base_idx, class_idx[c], ...)`), como já é feito — só a escolha do
   ponto-base muda.

### Critério de aceite
- Rodar 5-fold com `SMOTE_STANDARD` (baseline atual, 69,4%/0,4435) vs.
  `SMOTE_BORDERLINE`, mesma seed, mesmas demais configurações.
- Adotar Borderline-SMOTE em definitivo apenas se Macro F1 igual ou melhor —
  atenção especial ao F1 de Disfonia Psicogênica/Funcional (classes com maior
  sobreposição, teoricamente as que mais se beneficiam de oversampling de borda).
- Atualizar a citação em resumo/pôster para `HAN; WANG; MAO, 2005` **somente** se
  esta mudança for de fato incorporada — não citar método não utilizado (erro já
  corrigido uma vez nesta sessão).

---

## GAP 3 (Prioridade MÉDIA-BAIXA) — Comparação Redes Rasas × Profundas

### Por quê é prioridade média-baixa
Mais trabalhoso (requer treinar múltiplas configurações × 5 folds × 3 vogais × 2 redes
= custo computacional não trivial) e o ganho marginal esperado é incerto dado o
tamanho do dataset (1098 pacientes) — mas a proposta pede explicitamente a comparação
e a **justificativa da escolha por complexidade**, então precisa constar mesmo que a
conclusão seja "a rasa venceu".

### Especificação de implementação

`mlp_init_dynamic()` (`src/mlp.c`) hoje só suporta 1 camada oculta
(`num_layers = (hidden_size > 0) ? 3 : 2`, ou seja, Input→Hidden(128)→Output).
Adicionar uma variante configurável:

```c
/* hidden_sizes: array de tamanhos das camadas ocultas (ex.: {128, 64} para 2 camadas).
 * n_hidden: numero de camadas ocultas (0 = rede linear, sem camada oculta).
 * dropout_rates: um valor de dropout por camada oculta (mesmo tamanho de hidden_sizes).
 */
void mlp_init_multi(MLP *net, int input_size, int output_size,
                     const int *hidden_sizes, int n_hidden,
                     const float *dropout_rates);
```

Isso generaliza `mlp_init_dynamic` (que pode virar um wrapper de compatibilidade
chamando `mlp_init_multi` com `n_hidden=1`). Verificar que `net->layers[4]` (tamanho
fixo hoje) comporta `n_hidden+1` camadas — se `n_hidden` puder chegar a 3, aumentar o
array para `layers[5]` ou tornar dinâmico (`malloc`).

### Configurações a testar (Mestra e Especialista, por vogal)

| Config | Camadas ocultas | Params aprox. (85→·→2) | Dropout |
|---|---|---|---|
| A (atual, rasa) | [128] | ~11k | 0,5 |
| B (rasa menor) | [64] | ~5.5k | 0,5 |
| C (profunda 2 camadas) | [128, 64] | ~19k | 0,5 / 0,4 |
| D (profunda 3 camadas) | [128, 64, 32] | ~21k | 0,5 / 0,4 / 0,3 |

Usar os mesmos hiperparâmetros de otimização já validados (Adam, LR com cosine
annealing, class weights por rede, early stop por Macro-F1, `RANDOM_SEED=42`).

### Protocolo de comparação
1. Nested CV: outer 5-fold (o de sempre) × as 4 configs acima, mesma partição de
   dados em todas.
2. Métrica primária: Macro F1 médio dos 5 folds. Métricas secundárias: tempo de
   treino/época (custo computacional) e nº de parâmetros.
3. Critério de escolha (a documentar explicitamente no relatório final, é o que a
   proposta pede): **escolher a configuração de menor complexidade cujo Macro F1 não
   seja estatisticamente pior que a melhor configuração** (usar teste de McNemar ou
   comparação de ICs de bootstrap já implementados no projeto, ver
   `results/train_log_v27_weights.txt` para exemplo de uso do McNemar existente).

### Critério de aceite
- Tabela final com as 4 configs × 5 métricas (Acurácia, Macro F1, F1 por classe,
  tempo/época, nº params) — vira uma figura extra para resumo/pôster/relatório.
- Config vencedora documentada com a devida justificativa de complexidade×desempenho.

---

## Ordem de Implementação Recomendada

1. **Gap 2** (Borderline-SMOTE) — mudança isolada, baixo risco, valida rapidamente o
   fluxo de comparação A/B que os outros gaps também vão usar.
2. **Gap 3** (rasas × profundas) — gera a tabela comparativa que a proposta pede
   explicitamente; usa a mesma infraestrutura de nested CV já existente.
3. **Gap 1** (seleção paraconsistente) — o mais complexo e o mais citado na proposta;
   implementar por último para poder combinar com a melhor config. de rede e o melhor
   modo de SMOTE já validados nos passos 1-2, evitando retrabalho de re-tunar tudo de
   novo depois.

## Critérios de Aceite Gerais (aplicam-se aos 3 gaps)

- Nenhuma mudança é incorporada à branch principal sem comparação A/B reprodutível
  (mesma seed, mesmos folds, log salvo em `results/train_log_vXX_<nome-do-gap>.txt`).
- Toda mudança que piorar Macro F1 global abaixo de 0,42 é revertida ou mantida apenas
  como experimento documentado (não como default).
- Atualizar `CLAUDE.md` (`Optimization History` / `What Worked` / `What Didn't Work`)
  ao final de cada gap, independentemente do resultado — inclusive se um gap **não**
  funcionar, isso deve ficar documentado (é dado real e relevante para a banca).
- Nenhuma citação bibliográfica é adicionada ao resumo/pôster/relatório para um método
  que não esteja de fato implementado e ativo na versão final do modelo.
