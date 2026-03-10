#ifndef KNN_H
#define KNN_H

/*
 * knn.h - Classificador k-Nearest Neighbors (baseline)
 *
 * Distancia Euclidiana + votacao por maioria.
 */

/*
 * Classifica n_val amostras usando os k vizinhos mais proximos do conjunto de treino.
 * X_train: [n_train x n_features], X_val: [n_val x n_features]
 * Resultado em y_pred_out[n_val].
 */
void knn_predict(const float *X_train, const int *y_train, int n_train,
                 const float *X_val,   int n_val,   int n_features,
                 int k, int *y_pred_out);

#endif /* KNN_H */
