/*
 * mlp.h - Rede Neural MLP (Multi-Layer Perceptron)
 *
 * Arquitetura: Input(150) -> Dense(256, BN, LeakyReLU) -> Dense(128, BN, LeakyReLU)
 *              -> Dense(64, BN, LeakyReLU) -> Dense(3, Softmax)
 * Forward pass, backward pass (backpropagation), otimizador Adam.
 */

#ifndef MLP_H
#define MLP_H

#include "config.h"

/* Batch Normalization parameters */
typedef struct {
    float *gamma;          /* scale [output_size] */
    float *beta;           /* shift [output_size] */
    float *running_mean;   /* running mean for inference [output_size] */
    float *running_var;    /* running variance for inference [output_size] */
    float *grad_gamma;     /* gradients */
    float *grad_beta;
    float *m_gamma, *v_gamma; /* Adam moments for gamma */
    float *m_beta, *v_beta;   /* Adam moments for beta */
    float *x_norm;         /* normalized values (for backprop) [output_size] */
    float *x_centered;     /* centered values (for backprop) [output_size] */
    float batch_mean;      /* current batch stats (unused, per-neuron) */
    float batch_var;
    int size;
    int enabled;
} BatchNorm;

/* Estrutura de uma camada densa */
typedef struct {
    int input_size;
    int output_size;

    float *weights;     /* [output_size x input_size] row-major */
    float *biases;      /* [output_size] */

    /* Gradientes */
    float *grad_w;      /* [output_size x input_size] */
    float *grad_b;      /* [output_size] */

    /* Momentos Adam */
    float *m_w, *v_w;   /* primeiro e segundo momento dos pesos */
    float *m_b, *v_b;   /* primeiro e segundo momento dos biases */

    /* Ativacoes (para backprop) */
    float *z;           /* pre-ativacao [output_size] */
    float *a;           /* pos-ativacao [output_size] */
    float *input;       /* ponteiro para entrada da camada */

    /* Dropout */
    float dropout_rate; /* taxa de dropout (0.0 = sem dropout) */
    int *dropout_mask;  /* mascara binaria [output_size] (1=manter, 0=zerar) */

    /* Batch Normalization */
    BatchNorm bn;
} Layer;

/* Rede MLP completa */
typedef struct {
    Layer layers[MLP_NUM_LAYERS];  /* 3 camadas: hidden1, hidden2, output */
    int num_layers;
    int timestep;                  /* contador para Adam */
} MLP;

/*
 * Inicializa a rede MLP com pesos aleatorios (He initialization).
 * Deve ser chamado apos rng_seed().
 */
void mlp_init(MLP *net);

/*
 * Inicializa MLP com tamanho de entrada dinamico (para feature selection).
 */
void mlp_init_dynamic(MLP *net, int input_size);

/*
 * Forward pass: calcula a saida da rede para uma entrada.
 * input: vetor de entrada [MLP_INPUT_SIZE]
 * output: vetor de saida (probabilidades) [MLP_OUTPUT_SIZE]
 * training: 1 para modo treino (com dropout), 0 para inferencia
 */
void mlp_forward(MLP *net, const float *input, float *output, int training);

/*
 * Backward pass: calcula gradientes via backpropagation.
 * target: vetor one-hot da classe verdadeira [MLP_OUTPUT_SIZE]
 * class_weight: peso da classe (para weighted cross-entropy)
 *
 * Os gradientes sao ACUMULADOS (nao zerados).
 * Chamar mlp_zero_gradients() antes de cada batch.
 */
void mlp_backward(MLP *net, const float *target, float class_weight);

/*
 * Zera os gradientes de todas as camadas.
 */
void mlp_zero_gradients(MLP *net);

/*
 * Atualiza pesos via Adam optimizer.
 * lr: learning rate
 */
void mlp_adam_update(MLP *net, float lr);

/*
 * Calcula a weighted cross-entropy loss para uma saida.
 * output: probabilidades da rede [MLP_OUTPUT_SIZE]
 * target: one-hot [MLP_OUTPUT_SIZE]
 * class_weight: peso da classe
 */
float mlp_loss(const float *output, const float *target, float class_weight);

/*
 * Adiciona regularizacao L2 aos gradientes e retorna o termo de loss.
 * l2_loss = (lambda/2) * sum(w^2)
 */
float mlp_l2_regularization(MLP *net, float lambda);

/*
 * Copia pesos e biases de src para buffers de checkpoint.
 * best_weights/best_biases devem ser arrays de ponteiros pre-alocados.
 */
void mlp_save_checkpoint(const MLP *net, float **best_weights, float **best_biases);

/*
 * Restaura pesos e biases dos buffers de checkpoint para a rede.
 */
void mlp_load_checkpoint(MLP *net, float **best_weights, float **best_biases);

/*
 * Salva os pesos da rede em arquivo binario.
 */
/*
 * Update BN running statistics from a batch of raw pre-activation values.
 */
void mlp_update_bn_stats(MLP *net, int layer_idx, const float *batch_z,
                         int batch_size);

int mlp_save(const MLP *net, const char *path);

/*
 * Carrega pesos de arquivo binario.
 */
int mlp_load(MLP *net, const char *path);

/*
 * Libera memoria da rede.
 */
void mlp_free(MLP *net);

#endif /* MLP_H */
