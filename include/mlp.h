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
    Layer layers[MLP_MAX_LAYERS];  /* dimensionado com folga para Config D (4 camadas); numero
                                       real de camadas em uso e sempre num_layers, nunca a macro */
    int num_layers;
    int timestep;                  /* contador para Adam */
} MLP;

/*
 * Inicializa a rede MLP com pesos aleatorios (He initialization).
 * Deve ser chamado apos rng_seed().
 */
void mlp_init(MLP *net);

/*
 * Inicializa MLP com tamanho de entrada e saida dinamicos.
 */
void mlp_init_dynamic(MLP *net, int input_size, int output_size);

/*
 * Inicializa MLP com arquitetura configuravel: hidden_sizes[n_hidden] define a largura
 * de cada camada oculta, dropout_rates[n_hidden] a taxa de dropout correspondente.
 * net->num_layers e definido internamente como (n_hidden + 1).
 */
void mlp_init_multi(MLP *net, int input_size, int output_size,
                     const int *hidden_sizes, int n_hidden,
                     const float *dropout_rates);

/*
 * Retorna o numero total de parametros treinaveis (pesos + biases) somados sobre
 * todas as net->num_layers camadas. BN esta desabilitado em todo o pipeline atual
 * (use_bn=0 em todas as chamadas de layer_init), portanto nao contribui.
 */
int mlp_count_params(const MLP *net);

/*
 * Forward pass: calcula a saida da rede para uma entrada.
 */
void mlp_forward(MLP *net, const float *input, float *output, int training);

/*
 * Backward pass: calcula gradientes via backpropagation.
 */
void mlp_backward(MLP *net, const float *target, float class_weight);

/*
 * Zera os gradientes de todas as camadas.
 */
void mlp_zero_gradients(MLP *net);

/*
 * Atualiza pesos via Adam optimizer.
 */
void mlp_adam_update(MLP *net, float lr);

/*
 * Calcula a weighted Focal Loss para uma saida.
 */
float mlp_loss(const float *output, const float *target, float class_weight, int output_size);

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
