/*
 * kfold.h - Stratified K-Fold Cross-Validation
 *
 * Divide o dataset em K folds estratificados por classe,
 * garantindo separacao por paciente (nao por gravacao).
 */

#ifndef KFOLD_H
#define KFOLD_H

#include "config.h"

/* Informacao de um fold */
typedef struct {
    int *train_indices;   /* indices dos pacientes de treino */
    int n_train;
    int *val_indices;     /* indices dos pacientes de validacao */
    int n_val;
} FoldSplit;

/* Conjunto de K folds */
typedef struct {
    FoldSplit folds[K_FOLDS];
    int k;
} KFoldSplits;

/*
 * Gera K folds estratificados.
 * labels: array de class labels [n]
 * n: numero total de amostras (pacientes)
 * seed: seed para reproducibilidade
 * splits: ponteiro para struct de saida
 */
void kfold_split(const int *labels, int n, unsigned int seed, KFoldSplits *splits);

/*
 * Libera memoria dos folds.
 */
void kfold_free(KFoldSplits *splits);

#endif /* KFOLD_H */
