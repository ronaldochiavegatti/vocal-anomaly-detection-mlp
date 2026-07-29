#ifndef FEATURE_SELECT_PARACONSISTENT_H
#define FEATURE_SELECT_PARACONSISTENT_H

/*
 * feature_select_paraconsistent.h - Selecao Paraconsistente de Caracteristicas (Gap 1)
 *
 * Este modulo computa (mu, lambda, Gc, Gct) por feature via Logica Paraconsistente
 * Anotada de dois valores (LPA2v/PAL2v) e seleciona features cujo Gc >= gc_thresh
 * E |Gct| <= gct_max, com relaxamento de limiar limitado por um numero maximo de
 * iteracoes (nunca relaxa indefinidamente).
 */

/*
 * Seleciona features via grau de certeza (Gc) e grau de contradicao (Gct)
 * paraconsistentes.
 *
 * x: matriz row-major n x nf contendo APENAS linhas de treino originais
 *    (nao-augmentadas, nao-sintetizadas por SMOTE) do fold atual -- callers
 *    DEVEM passar fold->n_train, nunca n_train_aug (mesmo principio ja
 *    documentado para norm_fit em CLAUDE.md).
 * y: rotulos em [0, n_classes) para cada uma das n linhas de x.
 * n: numero de linhas (amostras de treino originais) em x/y.
 * nf: numero de colunas (features) em x.
 * n_classes: numero de classes distintas presentes em y.
 * gc_thresh: limiar minimo inicial de Gc para selecionar uma feature.
 * gct_max: valor maximo tolerado de |Gct| para selecionar uma feature.
 * selected: buffer de saida alocado pelo chamador, capacidade >= nf; recebe
 *           os indices das features selecionadas.
 * mu_out: buffer opcional (pode ser NULL) alocado pelo chamador, tamanho nf;
 *         recebe o valor de mu (eta-quadrado) de TODAS as nf features,
 *         independentemente de terem sido selecionadas.
 * lambda_out: buffer opcional (pode ser NULL), tamanho nf; recebe lambda de
 *             todas as nf features.
 * gc_out: buffer opcional (pode ser NULL), tamanho nf; recebe Gc de todas as
 *         nf features.
 * gct_out: buffer opcional (pode ser NULL), tamanho nf; recebe Gct de todas
 *          as nf features.
 *
 * Retorna o numero de features selecionadas (preenchidas em selected[]),
 * garantidamente >= 1 (nunca 0): se o laco de relaxamento esgotar
 * PARA_MAX_RELAX_ITERS iteracoes sem selecionar nenhuma feature, todas as nf
 * features sao usadas como fallback.
 */
int paraconsistent_select(const float *x, const int *y, int n, int nf, int n_classes,
                           float gc_thresh, float gct_max, int *selected,
                           float *mu_out, float *lambda_out,
                           float *gc_out, float *gct_out);

#endif /* FEATURE_SELECT_PARACONSISTENT_H */
