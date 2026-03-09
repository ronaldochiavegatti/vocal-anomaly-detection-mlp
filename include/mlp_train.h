/*
 * mlp_train.h - Loop de treinamento do MLP
 *
 * Mini-batch training com weighted cross-entropy, regularizacao L2,
 * early stopping e logging.
 */

#ifndef MLP_TRAIN_H
#define MLP_TRAIN_H

#include "mlp.h"
#include "config.h"

/* Resultado de uma epoca de treinamento */
typedef struct {
    float train_loss;
    float train_acc;
    float val_loss;
    float val_acc;
} EpochResult;

/* Historico de treinamento */
typedef struct {
    EpochResult *epochs;
    int num_epochs;        /* epochs efetivamente executadas */
    int best_epoch;        /* epoch com menor val_loss */
    float best_val_loss;
} TrainHistory;

/*
 * Treina o MLP com os dados fornecidos.
 *
 * net: rede inicializada
 * train_x: features de treino [n_train x num_features]
 * train_y: labels de treino [n_train]
 * n_train: numero de amostras de treino
 * val_x: features de validacao [n_val x num_features]
 * val_y: labels de validacao [n_val]
 * n_val: numero de amostras de validacao
 * num_features: dimensao das features
 * history: ponteiro para historico de saida
 *
 * Retorna 0 em sucesso.
 */
int mlp_train(MLP *net,
              const float *train_x, const int *train_y, int n_train,
              const float *val_x, const int *val_y, int n_val,
              int num_features, TrainHistory *history);

/*
 * Avalia o MLP num conjunto de dados.
 * Retorna a acuracia.
 */
float mlp_evaluate(MLP *net, const float *x, const int *y, int n,
                   int num_features, float *loss_out);

/*
 * Libera a memoria do historico.
 */
void train_history_free(TrainHistory *h);

#endif /* MLP_TRAIN_H */
