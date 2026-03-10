#ifndef LOGREG_H
#define LOGREG_H

/*
 * logreg.h - Regressao Logistica Multinomial (baseline)
 *
 * Input → Linear → Softmax  (equivalente a MLP com 0 hidden layers)
 * Treinado com Adam, cosine annealing LR, early stopping, label smoothing,
 * L2 regularizacao e noise injection — mesmos hiperparametros do MLP principal.
 */

typedef struct {
    int input_size;
    int n_classes;
    float *W;         /* pesos [n_classes × input_size] */
    float *b;         /* biases [n_classes] */
    float *grad_W;
    float *grad_b;
    float *m_W, *v_W; /* momentos Adam para W */
    float *m_b, *v_b; /* momentos Adam para b */
    int timestep;
} LRModel;

void lr_init(LRModel *lr, int input_size, int n_classes);
void lr_free(LRModel *lr);

/*
 * Treina o modelo e preenche y_pred_out com predicoes na validacao.
 * Usa os mesmos hiperparametros de config.h que o MLP principal.
 */
void lr_train(LRModel *lr,
              const float *train_x, const int *train_y, int n_train,
              const float *val_x,   const int *val_y,   int n_val,
              int num_features, int *y_pred_out);

#endif /* LOGREG_H */
