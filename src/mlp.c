/*
 * mlp.c - Rede Neural MLP (Multi-Layer Perceptron)
 *
 * Implementacao de forward pass (LeakyReLU + BatchNorm + Softmax),
 * backward pass (backpropagation), gradient clipping, e otimizador Adam.
 *
 * Arquitetura: Input(150) -> Dense(256, BN, LeakyReLU, Dropout)
 *              -> Dense(128, BN, LeakyReLU, Dropout)
 *              -> Dense(64, BN, LeakyReLU, Dropout)
 *              -> Dense(3, Softmax)
 */

#include "mlp.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========== Batch Normalization ========== */

static void bn_init(BatchNorm *bn, int size)
{
    bn->size = size;
    bn->enabled = 1;
    bn->gamma = (float *)safe_malloc(size * sizeof(float));
    bn->beta = (float *)safe_calloc(size, sizeof(float));
    bn->running_mean = (float *)safe_calloc(size, sizeof(float));
    bn->running_var = (float *)safe_malloc(size * sizeof(float));
    bn->grad_gamma = (float *)safe_calloc(size, sizeof(float));
    bn->grad_beta = (float *)safe_calloc(size, sizeof(float));
    bn->m_gamma = (float *)safe_calloc(size, sizeof(float));
    bn->v_gamma = (float *)safe_calloc(size, sizeof(float));
    bn->m_beta = (float *)safe_calloc(size, sizeof(float));
    bn->v_beta = (float *)safe_calloc(size, sizeof(float));
    bn->x_norm = (float *)safe_calloc(size, sizeof(float));
    bn->x_centered = (float *)safe_calloc(size, sizeof(float));

    for (int i = 0; i < size; i++) {
        bn->gamma[i] = 1.0f;
        bn->running_var[i] = 1.0f;
    }
}

static void bn_free(BatchNorm *bn)
{
    free(bn->gamma); free(bn->beta);
    free(bn->running_mean); free(bn->running_var);
    free(bn->grad_gamma); free(bn->grad_beta);
    free(bn->m_gamma); free(bn->v_gamma);
    free(bn->m_beta); free(bn->v_beta);
    free(bn->x_norm); free(bn->x_centered);
}

/*
 * BatchNorm forward per-sample (simplified for SGD/mini-batch):
 * During training: use running stats (we update running stats per-batch externally)
 * Actually, for simplicity with per-sample forward, we use running stats always
 * and update running stats from batch statistics computed in the training loop.
 *
 * For our per-sample architecture, BN uses running statistics:
 *   x_norm = (x - running_mean) / sqrt(running_var + eps)
 *   out = gamma * x_norm + beta
 */
static void bn_forward(BatchNorm *bn, float *z, int n)
{
    for (int i = 0; i < n; i++) {
        bn->x_centered[i] = z[i] - bn->running_mean[i];
        float inv_std = 1.0f / sqrtf(bn->running_var[i] + BN_EPSILON);
        bn->x_norm[i] = bn->x_centered[i] * inv_std;
        z[i] = bn->gamma[i] * bn->x_norm[i] + bn->beta[i];
    }
}

/*
 * BN backward: compute grad_gamma, grad_beta, and modify delta
 * delta_in = delta * gamma / sqrt(var + eps)
 */
static void bn_backward(BatchNorm *bn, float *delta, int n)
{
    for (int i = 0; i < n; i++) {
        bn->grad_gamma[i] += delta[i] * bn->x_norm[i];
        bn->grad_beta[i] += delta[i];
        float inv_std = 1.0f / sqrtf(bn->running_var[i] + BN_EPSILON);
        delta[i] = delta[i] * bn->gamma[i] * inv_std;
    }
}

static void bn_zero_gradients(BatchNorm *bn)
{
    memset(bn->grad_gamma, 0, bn->size * sizeof(float));
    memset(bn->grad_beta, 0, bn->size * sizeof(float));
}

/*
 * Update running statistics from a batch of pre-activation values.
 * Called once per batch with accumulated z values.
 */
static void bn_update_running_stats(BatchNorm *bn, const float *batch_z,
                                     int n_samples, int n_features)
{
    for (int j = 0; j < n_features; j++) {
        float mean = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            mean += batch_z[i * n_features + j];
        }
        mean /= n_samples;

        float var = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            float diff = batch_z[i * n_features + j] - mean;
            var += diff * diff;
        }
        var /= n_samples;

        bn->running_mean[j] = (1.0f - BN_MOMENTUM) * bn->running_mean[j] + BN_MOMENTUM * mean;
        bn->running_var[j] = (1.0f - BN_MOMENTUM) * bn->running_var[j] + BN_MOMENTUM * var;
    }
}

/* ========== Inicializacao de camada ========== */

static void layer_init(Layer *l, int input_size, int output_size, int use_bn)
{
    l->input_size = input_size;
    l->output_size = output_size;

    int nw = output_size * input_size;

    l->weights = (float *)safe_malloc(nw * sizeof(float));
    l->biases = (float *)safe_calloc(output_size, sizeof(float));
    l->grad_w = (float *)safe_calloc(nw, sizeof(float));
    l->grad_b = (float *)safe_calloc(output_size, sizeof(float));
    l->m_w = (float *)safe_calloc(nw, sizeof(float));
    l->v_w = (float *)safe_calloc(nw, sizeof(float));
    l->m_b = (float *)safe_calloc(output_size, sizeof(float));
    l->v_b = (float *)safe_calloc(output_size, sizeof(float));
    l->z = (float *)safe_malloc(output_size * sizeof(float));
    l->a = (float *)safe_malloc(output_size * sizeof(float));
    l->input = NULL;
    l->dropout_rate = 0.0f;
    l->dropout_mask = (int *)safe_calloc(output_size, sizeof(int));

    /* He initialization: w ~ N(0, sqrt(2/input_size)) */
    float scale = sqrtf(2.0f / input_size);
    for (int i = 0; i < nw; i++) {
        l->weights[i] = rng_normal() * scale;
    }

    /* Batch Normalization */
    if (use_bn) {
        bn_init(&l->bn, output_size);
    } else {
        memset(&l->bn, 0, sizeof(BatchNorm));
        l->bn.enabled = 0;
    }
}

static void layer_free(Layer *l)
{
    free(l->weights); free(l->biases);
    free(l->grad_w); free(l->grad_b);
    free(l->m_w); free(l->v_w);
    free(l->m_b); free(l->v_b);
    free(l->z); free(l->a);
    free(l->dropout_mask);
    if (l->bn.enabled) bn_free(&l->bn);
}

/* ========== Funcoes de ativacao ========== */

static void leaky_relu(float *z, float *a, int n)
{
    for (int i = 0; i < n; i++) {
        a[i] = (z[i] > 0.0f) ? z[i] : LEAKY_RELU_ALPHA * z[i];
    }
}

static void softmax(float *z, float *a, int n)
{
    float max_z = z[0];
    for (int i = 1; i < n; i++) {
        if (z[i] > max_z) max_z = z[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        a[i] = expf(z[i] - max_z);
        sum += a[i];
    }
    for (int i = 0; i < n; i++) {
        a[i] /= sum;
    }
}

/* ========== Forward de uma camada ========== */

static void layer_forward(Layer *l, const float *input)
{
    l->input = (float *)input;

    /* z = W * input + b */
    for (int i = 0; i < l->output_size; i++) {
        float sum = l->biases[i];
        const float *w_row = &l->weights[i * l->input_size];
        for (int j = 0; j < l->input_size; j++) {
            sum += w_row[j] * input[j];
        }
        l->z[i] = sum;
    }
}

/* ========== Interface publica ========== */

void mlp_init(MLP *net)
{
    mlp_init_dynamic(net, MLP_INPUT_SIZE);
}

void mlp_init_dynamic(MLP *net, int input_size)
{
    net->num_layers = MLP_NUM_LAYERS;
    net->timestep = 0;

#if MLP_NUM_LAYERS == 3
    int sizes[] = { input_size, MLP_HIDDEN1_SIZE, MLP_HIDDEN2_SIZE, MLP_OUTPUT_SIZE };
    float dropout_rates[] = { DROPOUT_RATE_HIDDEN1, DROPOUT_RATE_HIDDEN2, 0.0f };
    int use_bn[] = { 0, 0, 0 };
#else
    int sizes[] = { input_size, MLP_HIDDEN1_SIZE, MLP_HIDDEN2_SIZE,
                    MLP_HIDDEN3_SIZE, MLP_OUTPUT_SIZE };
    float dropout_rates[] = { DROPOUT_RATE_HIDDEN1, DROPOUT_RATE_HIDDEN2,
                              DROPOUT_RATE_HIDDEN3, 0.0f };
    int use_bn[] = { 0, 0, 0, 0 };
#endif

    for (int i = 0; i < net->num_layers; i++) {
        layer_init(&net->layers[i], sizes[i], sizes[i + 1], use_bn[i]);
        net->layers[i].dropout_rate = dropout_rates[i];
    }
}

void mlp_forward(MLP *net, const float *input, float *output, int training)
{
    const float *current_input = input;

    for (int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        layer_forward(l, current_input);

        if (i < net->num_layers - 1) {
            /* Hidden layer: BN -> LeakyReLU -> Dropout */

            /* Batch Normalization */
            if (l->bn.enabled) {
                bn_forward(&l->bn, l->z, l->output_size);
            }

            /* LeakyReLU */
            leaky_relu(l->z, l->a, l->output_size);

            /* Inverted dropout */
            if (training && l->dropout_rate > 0.0f) {
                float scale = 1.0f / (1.0f - l->dropout_rate);
                for (int j = 0; j < l->output_size; j++) {
                    l->dropout_mask[j] = (rng_uniform() >= l->dropout_rate) ? 1 : 0;
                    l->a[j] *= l->dropout_mask[j] * scale;
                }
            }
        } else {
            /* Output layer: Softmax */
            softmax(l->z, l->a, l->output_size);
        }

        current_input = l->a;
    }

    memcpy(output, net->layers[net->num_layers - 1].a,
           MLP_OUTPUT_SIZE * sizeof(float));
}

void mlp_backward(MLP *net, const float *target, float class_weight)
{
    int nl = net->num_layers;

    /* Allocate delta buffers sized for largest hidden layer */
    int max_size = MLP_HIDDEN1_SIZE;
    float *delta = (float *)safe_malloc(max_size * sizeof(float));
    float *delta_next = (float *)safe_malloc(max_size * sizeof(float));

    /* Output layer: delta = weight * (a - target) for softmax + cross-entropy */
    Layer *out_layer = &net->layers[nl - 1];
    float *out_delta = (float *)safe_malloc(out_layer->output_size * sizeof(float));

    for (int i = 0; i < out_layer->output_size; i++) {
        out_delta[i] = class_weight * (out_layer->a[i] - target[i]);
    }

    /* Accumulate output layer gradients */
    for (int i = 0; i < out_layer->output_size; i++) {
        for (int j = 0; j < out_layer->input_size; j++) {
            out_layer->grad_w[i * out_layer->input_size + j] +=
                out_delta[i] * out_layer->input[j];
        }
        out_layer->grad_b[i] += out_delta[i];
    }

    /* Backpropagate through hidden layers */
    float *current_delta = out_delta;
    int current_delta_size = out_layer->output_size;

    for (int l = nl - 2; l >= 0; l--) {
        Layer *layer = &net->layers[l];
        Layer *next_layer = &net->layers[l + 1];

        /* delta_l = (W_{l+1}^T * delta_{l+1}) * leaky_relu'(z_l) * dropout */
        for (int i = 0; i < layer->output_size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < current_delta_size; j++) {
                sum += next_layer->weights[j * next_layer->input_size + i] *
                       current_delta[j];
            }
            /* LeakyReLU derivative */
            float deriv = (layer->z[i] > 0.0f) ? 1.0f : LEAKY_RELU_ALPHA;
            delta[i] = sum * deriv;

            /* Apply dropout mask */
            if (layer->dropout_rate > 0.0f) {
                float scale = 1.0f / (1.0f - layer->dropout_rate);
                delta[i] *= layer->dropout_mask[i] * scale;
            }
        }

        /* Batch Norm backward */
        if (layer->bn.enabled) {
            bn_backward(&layer->bn, delta, layer->output_size);
        }

        /* Accumulate gradients */
        for (int i = 0; i < layer->output_size; i++) {
            for (int j = 0; j < layer->input_size; j++) {
                layer->grad_w[i * layer->input_size + j] +=
                    delta[i] * layer->input[j];
            }
            layer->grad_b[i] += delta[i];
        }

        /* Swap delta buffers */
        float *tmp = delta;
        delta = delta_next;
        delta_next = tmp;
        current_delta = delta_next;
        current_delta_size = layer->output_size;
    }

    free(delta);
    free(delta_next);
    free(out_delta);
}

void mlp_zero_gradients(MLP *net)
{
    for (int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        memset(l->grad_w, 0, l->output_size * l->input_size * sizeof(float));
        memset(l->grad_b, 0, l->output_size * sizeof(float));
        if (l->bn.enabled) {
            bn_zero_gradients(&l->bn);
        }
    }
}

/*
 * Gradient clipping by global norm.
 * Computes the global L2 norm of all gradients and scales them down
 * if the norm exceeds max_norm.
 */
static void mlp_clip_gradients(MLP *net, float max_norm)
{
    float global_norm = 0.0f;

    for (int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        int nw = l->output_size * l->input_size;
        for (int j = 0; j < nw; j++) {
            global_norm += l->grad_w[j] * l->grad_w[j];
        }
        for (int j = 0; j < l->output_size; j++) {
            global_norm += l->grad_b[j] * l->grad_b[j];
        }
        if (l->bn.enabled) {
            for (int j = 0; j < l->bn.size; j++) {
                global_norm += l->bn.grad_gamma[j] * l->bn.grad_gamma[j];
                global_norm += l->bn.grad_beta[j] * l->bn.grad_beta[j];
            }
        }
    }

    global_norm = sqrtf(global_norm);

    if (global_norm > max_norm) {
        float scale = max_norm / global_norm;
        for (int i = 0; i < net->num_layers; i++) {
            Layer *l = &net->layers[i];
            int nw = l->output_size * l->input_size;
            for (int j = 0; j < nw; j++) l->grad_w[j] *= scale;
            for (int j = 0; j < l->output_size; j++) l->grad_b[j] *= scale;
            if (l->bn.enabled) {
                for (int j = 0; j < l->bn.size; j++) {
                    l->bn.grad_gamma[j] *= scale;
                    l->bn.grad_beta[j] *= scale;
                }
            }
        }
    }
}

void mlp_adam_update(MLP *net, float lr)
{
    /* Clip gradients before update */
    mlp_clip_gradients(net, GRAD_CLIP_NORM);

    net->timestep++;
    float bc1 = 1.0f - powf(ADAM_BETA1, (float)net->timestep);
    float bc2 = 1.0f - powf(ADAM_BETA2, (float)net->timestep);

    for (int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        int nw = l->output_size * l->input_size;

        /* Update weights */
        for (int j = 0; j < nw; j++) {
            l->m_w[j] = ADAM_BETA1 * l->m_w[j] + (1.0f - ADAM_BETA1) * l->grad_w[j];
            l->v_w[j] = ADAM_BETA2 * l->v_w[j] + (1.0f - ADAM_BETA2) * l->grad_w[j] * l->grad_w[j];

            float m_hat = l->m_w[j] / bc1;
            float v_hat = l->v_w[j] / bc2;

            l->weights[j] -= lr * m_hat / (sqrtf(v_hat) + ADAM_EPSILON);
        }

        /* Update biases */
        for (int j = 0; j < l->output_size; j++) {
            l->m_b[j] = ADAM_BETA1 * l->m_b[j] + (1.0f - ADAM_BETA1) * l->grad_b[j];
            l->v_b[j] = ADAM_BETA2 * l->v_b[j] + (1.0f - ADAM_BETA2) * l->grad_b[j] * l->grad_b[j];

            float m_hat = l->m_b[j] / bc1;
            float v_hat = l->v_b[j] / bc2;

            l->biases[j] -= lr * m_hat / (sqrtf(v_hat) + ADAM_EPSILON);
        }

        /* Update BN parameters */
        if (l->bn.enabled) {
            BatchNorm *bn = &l->bn;
            for (int j = 0; j < bn->size; j++) {
                bn->m_gamma[j] = ADAM_BETA1 * bn->m_gamma[j] + (1.0f - ADAM_BETA1) * bn->grad_gamma[j];
                bn->v_gamma[j] = ADAM_BETA2 * bn->v_gamma[j] + (1.0f - ADAM_BETA2) * bn->grad_gamma[j] * bn->grad_gamma[j];
                float m_hat = bn->m_gamma[j] / bc1;
                float v_hat = bn->v_gamma[j] / bc2;
                bn->gamma[j] -= lr * m_hat / (sqrtf(v_hat) + ADAM_EPSILON);

                bn->m_beta[j] = ADAM_BETA1 * bn->m_beta[j] + (1.0f - ADAM_BETA1) * bn->grad_beta[j];
                bn->v_beta[j] = ADAM_BETA2 * bn->v_beta[j] + (1.0f - ADAM_BETA2) * bn->grad_beta[j] * bn->grad_beta[j];
                m_hat = bn->m_beta[j] / bc1;
                v_hat = bn->v_beta[j] / bc2;
                bn->beta[j] -= lr * m_hat / (sqrtf(v_hat) + ADAM_EPSILON);
            }
        }
    }
}

float mlp_loss(const float *output, const float *target, float class_weight)
{
    float loss = 0.0f;
    for (int i = 0; i < MLP_OUTPUT_SIZE; i++) {
        if (target[i] > 0.0f) {
            float p = output[i];
            if (p < 1e-7f) p = 1e-7f;
            loss -= class_weight * target[i] * logf(p);
        }
    }
    return loss;
}

float mlp_l2_regularization(MLP *net, float lambda)
{
    float l2_loss = 0.0f;

    for (int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        int nw = l->output_size * l->input_size;

        for (int j = 0; j < nw; j++) {
            l2_loss += l->weights[j] * l->weights[j];
            l->grad_w[j] += lambda * l->weights[j];
        }
    }

    return 0.5f * lambda * l2_loss;
}

void mlp_save_checkpoint(const MLP *net, float **best_weights, float **best_biases)
{
    for (int i = 0; i < net->num_layers; i++) {
        const Layer *l = &net->layers[i];
        int nw = l->output_size * l->input_size;
        memcpy(best_weights[i], l->weights, nw * sizeof(float));
        memcpy(best_biases[i], l->biases, l->output_size * sizeof(float));
    }
}

void mlp_load_checkpoint(MLP *net, float **best_weights, float **best_biases)
{
    for (int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        int nw = l->output_size * l->input_size;
        memcpy(l->weights, best_weights[i], nw * sizeof(float));
        memcpy(l->biases, best_biases[i], l->output_size * sizeof(float));
    }
}

/*
 * Update BN running statistics from a batch of raw pre-activation values.
 * batch_z: [batch_size x output_size] raw z values for a hidden layer.
 */
void mlp_update_bn_stats(MLP *net, int layer_idx, const float *batch_z,
                         int batch_size)
{
    Layer *l = &net->layers[layer_idx];
    if (l->bn.enabled) {
        bn_update_running_stats(&l->bn, batch_z, batch_size, l->output_size);
    }
}

int mlp_save(const MLP *net, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    for (int i = 0; i < net->num_layers; i++) {
        const Layer *l = &net->layers[i];
        int nw = l->output_size * l->input_size;
        fwrite(l->weights, sizeof(float), nw, f);
        fwrite(l->biases, sizeof(float), l->output_size, f);
        if (l->bn.enabled) {
            fwrite(l->bn.gamma, sizeof(float), l->bn.size, f);
            fwrite(l->bn.beta, sizeof(float), l->bn.size, f);
            fwrite(l->bn.running_mean, sizeof(float), l->bn.size, f);
            fwrite(l->bn.running_var, sizeof(float), l->bn.size, f);
        }
    }

    fclose(f);
    return 0;
}

int mlp_load(MLP *net, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    for (int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        int nw = l->output_size * l->input_size;
        if ((int)fread(l->weights, sizeof(float), nw, f) != nw) { fclose(f); return -1; }
        if ((int)fread(l->biases, sizeof(float), l->output_size, f) != l->output_size) { fclose(f); return -1; }
        if (l->bn.enabled) {
            if ((int)fread(l->bn.gamma, sizeof(float), l->bn.size, f) != l->bn.size) { fclose(f); return -1; }
            if ((int)fread(l->bn.beta, sizeof(float), l->bn.size, f) != l->bn.size) { fclose(f); return -1; }
            if ((int)fread(l->bn.running_mean, sizeof(float), l->bn.size, f) != l->bn.size) { fclose(f); return -1; }
            if ((int)fread(l->bn.running_var, sizeof(float), l->bn.size, f) != l->bn.size) { fclose(f); return -1; }
        }
    }

    fclose(f);
    return 0;
}

void mlp_free(MLP *net)
{
    for (int i = 0; i < net->num_layers; i++) {
        layer_free(&net->layers[i]);
    }
}
