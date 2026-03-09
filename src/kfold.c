/*
 * kfold.c - Stratified K-Fold Cross-Validation
 *
 * Separa indices por classe, embaralha cada grupo,
 * e distribui proporcionalmente nos K folds.
 */

#include "kfold.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>

void kfold_split(const int *labels, int n, unsigned int seed, KFoldSplits *splits)
{
    splits->k = K_FOLDS;

    /* Separar indices por classe */
    int *class_indices[NUM_CLASSES];
    int class_counts[NUM_CLASSES] = {0};

    /* Contar */
    for (int i = 0; i < n; i++) class_counts[labels[i]]++;

    for (int c = 0; c < NUM_CLASSES; c++) {
        class_indices[c] = (int *)safe_malloc(class_counts[c] * sizeof(int));
        class_counts[c] = 0;
    }

    for (int i = 0; i < n; i++) {
        int c = labels[i];
        class_indices[c][class_counts[c]++] = i;
    }

    /* Embaralhar cada classe */
    rng_seed(seed);
    for (int c = 0; c < NUM_CLASSES; c++) {
        rng_shuffle_int(class_indices[c], class_counts[c]);
    }

    /* Atribuir cada amostra a um fold (round-robin por classe) */
    int *fold_assignment = (int *)safe_malloc(n * sizeof(int));
    for (int c = 0; c < NUM_CLASSES; c++) {
        for (int i = 0; i < class_counts[c]; i++) {
            fold_assignment[class_indices[c][i]] = i % K_FOLDS;
        }
    }

    /* Montar os splits */
    for (int f = 0; f < K_FOLDS; f++) {
        /* Contar tamanhos */
        int n_val = 0, n_train = 0;
        for (int i = 0; i < n; i++) {
            if (fold_assignment[i] == f) n_val++;
            else n_train++;
        }

        splits->folds[f].n_train = n_train;
        splits->folds[f].n_val = n_val;
        splits->folds[f].train_indices = (int *)safe_malloc(n_train * sizeof(int));
        splits->folds[f].val_indices = (int *)safe_malloc(n_val * sizeof(int));

        int ti = 0, vi = 0;
        for (int i = 0; i < n; i++) {
            if (fold_assignment[i] == f) {
                splits->folds[f].val_indices[vi++] = i;
            } else {
                splits->folds[f].train_indices[ti++] = i;
            }
        }
    }

    /* Liberar temporarios */
    for (int c = 0; c < NUM_CLASSES; c++) free(class_indices[c]);
    free(fold_assignment);
}

void kfold_free(KFoldSplits *splits)
{
    for (int f = 0; f < splits->k; f++) {
        free(splits->folds[f].train_indices);
        free(splits->folds[f].val_indices);
    }
}
