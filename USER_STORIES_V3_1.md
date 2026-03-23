# User Stories - Pipeline Vocal MLP v3.1 (Refinamento)

As histórias abaixo visam atingir as metas de Macro F1 > 0.65 e F1-Disfonia > 0.50, com base nos resultados de 10/03/2026.

## US-010 ✅: Refinamento de Features Wavelet (Denoising)
**Como** pesquisador de processamento de sinais,
**Quero** implementar o soft-thresholding nos coeficientes de Wavelet,
**Para** que as features de frequência capturem o sinal glótico puro e reduzam o ruído de fundo.
- **Critério de Aceitação:** ✅ Implementar função `wavelet_denoise` em `dsp_utils.c`.
- **Validação:** A importância de features (Permutation Importance) do grupo Wavelet deve subir pelo menos 15% na rodada subsequente.

## US-011 ✅: Dinâmicas de MFCC (Delta e Delta-Delta)
**Como** cientista de dados,
**Quero** extrair as derivadas de 1ª e 2ª ordem dos 13 MFCCs,
**Para** capturar a instabilidade temporal da voz em disfonias psicogênicas.
- **Critério de Aceitação:** ✅ Expandir o vetor de features de 150 para 228 (3 vogais x (50 + 26 deltas)).
- **Validação:** Aumento do F1-Score na classe Disfonia em ambiente de validação cruzada (K-fold).

## US-012 ✅: Otimização de Perda (Weighted Cross-Entropy)
**Como** engenheiro de ML,
**Quero** atribuir pesos maiores para a classe Disfonia na função de erro do MLP,
**Para** compensar a baixa sensibilidade do modelo atual.
- **Critério de Aceitação:** ✅ Modificar `mlp_backward` para aceitar um vetor de `class_weights`.
- **Validação:** Redução de 30% nas falsas classificações de Disfonia como Normal.

## US-013 ✅: Early Stopping por Macro F1
**Como** engenheiro de ML,
**Quero** que o critério de parada do treino seja o Macro F1 no conjunto de validação,
**Para** garantir o melhor equilíbrio entre sensibilidade e especificidade de todas as classes.
- **Critério de Aceitação:** ✅ Alterar a lógica de monitoramento em `mlp_train.c`.
- **Validação:** Log de treinamento deve mostrar a interrupção baseada na métrica F1 (em vez da acurácia global).

## US-014 ✅: Paralelização da Extração (Performance)
**Como** desenvolvedor,
**Quero** utilizar OpenMP para paralelizar o processamento dos arquivos WAV,
**Para** reduzir o tempo total da pipeline de 21 minutos para menos de 5 minutos.
- **Critério de Aceitação:** ✅ Adicionar flags `-fopenmp` no Makefile e pragmas em `features_extract_all`.
- **Validação:** Medição de tempo total via comando `time`.
