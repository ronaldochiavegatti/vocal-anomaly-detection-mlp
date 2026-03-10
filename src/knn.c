#include "knn.h"
#include "config.h"
#include "utils.h"
#include <stdlib.h>

void knn_predict(const float *X_train, const int *y_train, int n_train,
                 const float *X_val,   int n_val,   int n_features,
                 int k, int *y_pred_out)
{
    int kk = (k < n_train) ? k : n_train;
    float *dists = (float *)safe_malloc((size_t)n_train * sizeof(float));
    int   *order = (int   *)safe_malloc((size_t)n_train * sizeof(int));

    for (int i = 0; i < n_val; i++) {
        /* Distancias ao quadrado para todas as amostras de treino */
        for (int j = 0; j < n_train; j++) {
            order[j] = j;
            float d = 0.0f;
            for (int f = 0; f < n_features; f++) {
                float diff = X_val[i * n_features + f] - X_train[j * n_features + f];
                d += diff * diff;
            }
            dists[j] = d;
        }

        /* Selecao parcial: encontrar os kk menores */
        for (int a = 0; a < kk; a++) {
            int min_idx = a;
            for (int b = a + 1; b < n_train; b++) {
                if (dists[order[b]] < dists[order[min_idx]])
                    min_idx = b;
            }
            int tmp = order[a]; order[a] = order[min_idx]; order[min_idx] = tmp;
        }

        /* Votacao por maioria */
        int votes[NUM_CLASSES] = {0};
        for (int a = 0; a < kk; a++)
            votes[y_train[order[a]]]++;

        int pred = 0;
        for (int c = 1; c < NUM_CLASSES; c++)
            if (votes[c] > votes[pred]) pred = c;

        y_pred_out[i] = pred;
    }

    free(dists);
    free(order);
}
