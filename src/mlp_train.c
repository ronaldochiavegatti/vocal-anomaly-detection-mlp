/*
 * mlp_train.c - Loop de treinamento do MLP
 *
 * Mini-batch com shuffle, Gaussian noise injection, focal loss,
 * label smoothing, moderate class weights, regularizacao L2,
 * cosine annealing LR, early stopping monitorando val_acc.
 */

#include "mlp_train.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float cosine_annealing_lr(int epoch, int max_epochs)
{
    float progress = (float)epoch / max_epochs;
    return LR_MIN + 0.5f * (LEARNING_RATE - LR_MIN) *
           (1.0f + cosf((float)M_PI * progress));
}

static void to_one_hot_smooth(int label, float *one_hot, float smooth)
{
    float off_value = smooth / (NUM_CLASSES - 1);
    float on_value = 1.0f - smooth;
    for (int i = 0; i < NUM_CLASSES; i++) {
        one_hot[i] = (i == label) ? on_value : off_value;
    }
}

static int argmax(const float *v, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

/* US-013: Compute Macro F1 on validation set for early stopping */
static float compute_val_macro_f1(MLP *net, const float *val_x, const int *val_y,
                                   int n_val, int num_features)
{
    int confusion[NUM_CLASSES][NUM_CLASSES];
    memset(confusion, 0, sizeof(confusion));
    float output[MLP_OUTPUT_SIZE];

    for (int i = 0; i < n_val; i++) {
        mlp_forward(net, &val_x[i * num_features], output, 0);
        int pred = argmax(output, MLP_OUTPUT_SIZE);
        if (val_y[i] >= 0 && val_y[i] < NUM_CLASSES && pred >= 0 && pred < NUM_CLASSES)
            confusion[val_y[i]][pred]++;
    }

    float f1_sum = 0.0f;
    for (int c = 0; c < NUM_CLASSES; c++) {
        int tp = confusion[c][c];
        int pred_sum = 0, true_sum = 0;
        for (int i = 0; i < NUM_CLASSES; i++) pred_sum += confusion[i][c];
        for (int j = 0; j < NUM_CLASSES; j++) true_sum += confusion[c][j];
        float p = (pred_sum > 0) ? (float)tp / pred_sum : 0.0f;
        float r = (true_sum > 0) ? (float)tp / true_sum : 0.0f;
        f1_sum += (p + r > 0.0f) ? 2.0f * p * r / (p + r) : 0.0f;
    }
    return f1_sum / NUM_CLASSES;
}

/*
 * Update BN running stats efficiently from a subsample of training data.
 */
static void update_bn_stats_from_data(MLP *net, const float *train_x,
                                       int n_train, int num_features)
{
    int n_sample = (n_train < 256) ? n_train : 256;
    int step = n_train / n_sample;

    for (int li = 0; li < net->num_layers - 1; li++) {
        Layer *l = &net->layers[li];
        if (!l->bn.enabled) continue;

        float *means = (float *)safe_calloc(l->output_size, sizeof(float));
        float *vars = (float *)safe_calloc(l->output_size, sizeof(float));

        /* First pass: compute means */
        for (int s = 0; s < n_sample; s++) {
            int idx = s * step;
            const float *cur_input = &train_x[idx * num_features];

            for (int k = 0; k <= li; k++) {
                Layer *lk = &net->layers[k];
                for (int i = 0; i < lk->output_size; i++) {
                    float sum = lk->biases[i];
                    const float *w_row = &lk->weights[i * lk->input_size];
                    for (int j = 0; j < lk->input_size; j++) {
                        sum += w_row[j] * cur_input[j];
                    }
                    lk->z[i] = sum;
                }

                if (k < li) {
                    if (lk->bn.enabled) {
                        for (int i = 0; i < lk->output_size; i++) {
                            float centered = lk->z[i] - lk->bn.running_mean[i];
                            float inv_std = 1.0f / sqrtf(lk->bn.running_var[i] + BN_EPSILON);
                            lk->z[i] = lk->bn.gamma[i] * centered * inv_std + lk->bn.beta[i];
                        }
                    }
                    for (int i = 0; i < lk->output_size; i++) {
                        lk->a[i] = (lk->z[i] > 0.0f) ? lk->z[i] : LEAKY_RELU_ALPHA * lk->z[i];
                    }
                    cur_input = lk->a;
                }
            }

            for (int i = 0; i < l->output_size; i++) {
                means[i] += l->z[i];
            }
        }

        for (int i = 0; i < l->output_size; i++) {
            means[i] /= n_sample;
        }

        /* Second pass: compute variances */
        for (int s = 0; s < n_sample; s++) {
            int idx = s * step;
            const float *cur_input = &train_x[idx * num_features];

            for (int k = 0; k <= li; k++) {
                Layer *lk = &net->layers[k];
                for (int i = 0; i < lk->output_size; i++) {
                    float sum = lk->biases[i];
                    const float *w_row = &lk->weights[i * lk->input_size];
                    for (int j = 0; j < lk->input_size; j++) {
                        sum += w_row[j] * cur_input[j];
                    }
                    lk->z[i] = sum;
                }

                if (k < li) {
                    if (lk->bn.enabled) {
                        for (int i = 0; i < lk->output_size; i++) {
                            float centered = lk->z[i] - lk->bn.running_mean[i];
                            float inv_std = 1.0f / sqrtf(lk->bn.running_var[i] + BN_EPSILON);
                            lk->z[i] = lk->bn.gamma[i] * centered * inv_std + lk->bn.beta[i];
                        }
                    }
                    for (int i = 0; i < lk->output_size; i++) {
                        lk->a[i] = (lk->z[i] > 0.0f) ? lk->z[i] : LEAKY_RELU_ALPHA * lk->z[i];
                    }
                    cur_input = lk->a;
                }
            }

            for (int i = 0; i < l->output_size; i++) {
                float diff = l->z[i] - means[i];
                vars[i] += diff * diff;
            }
        }

        for (int i = 0; i < l->output_size; i++) {
            vars[i] /= n_sample;
        }

        for (int i = 0; i < l->output_size; i++) {
            l->bn.running_mean[i] = (1.0f - BN_MOMENTUM) * l->bn.running_mean[i]
                                    + BN_MOMENTUM * means[i];
            l->bn.running_var[i] = (1.0f - BN_MOMENTUM) * l->bn.running_var[i]
                                   + BN_MOMENTUM * vars[i];
        }

        free(means);
        free(vars);
    }
}

int mlp_train(MLP *net,
              const float *train_x, const int *train_y, int n_train,
              const float *val_x, const int *val_y, int n_val,
              int num_features, TrainHistory *history)
{
    history->epochs = (EpochResult *)safe_malloc(MAX_EPOCHS * sizeof(EpochResult));
    history->num_epochs = 0;
    history->best_epoch = 0;
    history->best_val_loss = 1e30f;
    float best_val_acc = -1.0f;
    float best_val_macro_f1 = -1.0f;

    int *indices = (int *)safe_malloc(n_train * sizeof(int));
    for (int i = 0; i < n_train; i++) indices[i] = i;

    float output[MLP_OUTPUT_SIZE];
    float one_hot[MLP_OUTPUT_SIZE];
    float *x_aug = (float *)safe_malloc(num_features * sizeof(float));

    int patience_counter = 0;

    /* Checkpoint buffers */
    float *best_weights[MLP_NUM_LAYERS];
    float *best_biases[MLP_NUM_LAYERS];
    float *best_bn_gamma[MLP_NUM_LAYERS];
    float *best_bn_beta[MLP_NUM_LAYERS];
    float *best_bn_mean[MLP_NUM_LAYERS];
    float *best_bn_var[MLP_NUM_LAYERS];

    for (int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        best_weights[i] = (float *)safe_malloc(l->output_size * l->input_size * sizeof(float));
        best_biases[i] = (float *)safe_malloc(l->output_size * sizeof(float));
        if (l->bn.enabled) {
            best_bn_gamma[i] = (float *)safe_malloc(l->bn.size * sizeof(float));
            best_bn_beta[i] = (float *)safe_malloc(l->bn.size * sizeof(float));
            best_bn_mean[i] = (float *)safe_malloc(l->bn.size * sizeof(float));
            best_bn_var[i] = (float *)safe_malloc(l->bn.size * sizeof(float));
        } else {
            best_bn_gamma[i] = best_bn_beta[i] = NULL;
            best_bn_mean[i] = best_bn_var[i] = NULL;
        }
    }
    mlp_save_checkpoint(net, best_weights, best_biases);

    /* SWA (Stochastic Weight Averaging) buffers */
    int swa_start = 20;      /* start averaging after this epoch */
    int swa_freq = 5;        /* average every N epochs */
    int swa_count = 0;
    float *swa_weights[MLP_NUM_LAYERS];
    float *swa_biases[MLP_NUM_LAYERS];
    for (int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        swa_weights[i] = (float *)safe_calloc(l->output_size * l->input_size, sizeof(float));
        swa_biases[i] = (float *)safe_calloc(l->output_size, sizeof(float));
    }

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        /* Update BN stats at start of each epoch */
        update_bn_stats_from_data(net, train_x, n_train, num_features);

        rng_shuffle_int(indices, n_train);

        float epoch_loss = 0.0f;
        int epoch_correct = 0;
        int num_batches = (n_train + BATCH_SIZE - 1) / BATCH_SIZE;

        for (int b = 0; b < num_batches; b++) {
            int batch_start = b * BATCH_SIZE;
            int batch_end = batch_start + BATCH_SIZE;
            if (batch_end > n_train) batch_end = n_train;
            int batch_size = batch_end - batch_start;

            mlp_zero_gradients(net);
            float batch_loss = 0.0f;

            for (int s = batch_start; s < batch_end; s++) {
                int idx = indices[s];
                const float *x = &train_x[idx * num_features];
                int y = train_y[idx];
                /* Add Gaussian noise to input */
                for (int f = 0; f < num_features; f++) {
                    x_aug[f] = x[f] + NOISE_STDDEV * rng_normal();
                }

                /* Forward */
                mlp_forward(net, x_aug, output, 1);

                /* Cross-entropy with label smoothing and mild class weights */
                static const float cw[NUM_CLASSES] = {
                    CLASS_WEIGHT_NORMAL, CLASS_WEIGHT_LARYNGITIS, CLASS_WEIGHT_DYSPHONIA
                };
                float w = cw[y];
                to_one_hot_smooth(y, one_hot, LABEL_SMOOTHING);
                batch_loss += mlp_loss(output, one_hot, w);

                /* Backward */
                mlp_backward(net, one_hot, w);

                if (argmax(output, MLP_OUTPUT_SIZE) == y) epoch_correct++;
            }

            /* Average gradients */
            for (int l = 0; l < net->num_layers; l++) {
                Layer *layer = &net->layers[l];
                int nw = layer->output_size * layer->input_size;
                float inv_bs = 1.0f / batch_size;
                for (int j = 0; j < nw; j++) layer->grad_w[j] *= inv_bs;
                for (int j = 0; j < layer->output_size; j++) layer->grad_b[j] *= inv_bs;
                if (layer->bn.enabled) {
                    for (int j = 0; j < layer->bn.size; j++) {
                        layer->bn.grad_gamma[j] *= inv_bs;
                        layer->bn.grad_beta[j] *= inv_bs;
                    }
                }
            }

            batch_loss += mlp_l2_regularization(net, L2_LAMBDA);

            float current_lr = cosine_annealing_lr(epoch, MAX_EPOCHS);
            mlp_adam_update(net, current_lr);

            epoch_loss += batch_loss;
        }

        epoch_loss /= n_train;
        float train_acc = (float)epoch_correct / n_train;

        float val_loss;
        float val_acc = mlp_evaluate(net, val_x, val_y, n_val, num_features, &val_loss);
        float val_macro_f1 = compute_val_macro_f1(net, val_x, val_y, n_val, num_features);

        EpochResult *er = &history->epochs[epoch];
        er->train_loss = epoch_loss;
        er->train_acc = train_acc;
        er->val_loss = val_loss;
        er->val_acc = val_acc;
        er->val_macro_f1 = val_macro_f1;
        history->num_epochs = epoch + 1;

        if ((epoch + 1) % 10 == 0 || epoch == 0) {
            log_info("Epoch %3d: train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f val_f1=%.3f",
                     epoch + 1, epoch_loss, train_acc, val_loss, val_acc, val_macro_f1);
        }

        /* SWA: accumulate weights after swa_start epoch */
        if (epoch >= swa_start && (epoch - swa_start) % swa_freq == 0) {
            swa_count++;
            for (int i = 0; i < net->num_layers; i++) {
                Layer *l = &net->layers[i];
                int nw = l->output_size * l->input_size;
                for (int j = 0; j < nw; j++)
                    swa_weights[i][j] += l->weights[j];
                for (int j = 0; j < l->output_size; j++)
                    swa_biases[i][j] += l->biases[j];
            }
        }

        if (val_loss < history->best_val_loss) {
            history->best_val_loss = val_loss;
        }
        if (val_acc > best_val_acc) {
            best_val_acc = val_acc;
        }
        /* US-013: Early stopping based on Macro F1 */
        if (val_macro_f1 > best_val_macro_f1) {
            best_val_macro_f1 = val_macro_f1;
            history->best_epoch = epoch;
            patience_counter = 0;
            mlp_save_checkpoint(net, best_weights, best_biases);
            for (int i = 0; i < net->num_layers; i++) {
                Layer *l = &net->layers[i];
                if (l->bn.enabled) {
                    memcpy(best_bn_gamma[i], l->bn.gamma, l->bn.size * sizeof(float));
                    memcpy(best_bn_beta[i], l->bn.beta, l->bn.size * sizeof(float));
                    memcpy(best_bn_mean[i], l->bn.running_mean, l->bn.size * sizeof(float));
                    memcpy(best_bn_var[i], l->bn.running_var, l->bn.size * sizeof(float));
                }
            }
        } else {
            patience_counter++;
            if (patience_counter >= EARLY_STOP_PATIENCE) {
                log_info("Early stopping na epoch %d (melhor: epoch %d, val_f1=%.4f val_acc=%.4f)",
                         epoch + 1, history->best_epoch + 1, best_val_macro_f1, best_val_acc);
                break;
            }
        }
    }

    /* Try SWA weights vs best checkpoint */
    if (swa_count >= 3) {
        /* Average SWA weights */
        for (int i = 0; i < net->num_layers; i++) {
            Layer *l = &net->layers[i];
            int nw = l->output_size * l->input_size;
            for (int j = 0; j < nw; j++)
                l->weights[j] = swa_weights[i][j] / swa_count;
            for (int j = 0; j < l->output_size; j++)
                l->biases[j] = swa_biases[i][j] / swa_count;
        }

        float swa_val_loss;
        float swa_val_acc = mlp_evaluate(net, val_x, val_y, n_val, num_features, &swa_val_loss);
        float swa_macro_f1 = compute_val_macro_f1(net, val_x, val_y, n_val, num_features);
        log_info("SWA val_f1=%.4f val_acc=%.4f (avg of %d snapshots) vs best val_f1=%.4f",
                 swa_macro_f1, swa_val_acc, swa_count, best_val_macro_f1);

        if (swa_macro_f1 > best_val_macro_f1) {
            log_info("Usando pesos SWA (melhor que checkpoint)");
            /* Keep current SWA weights in net */
        } else {
            /* Restore best checkpoint */
            mlp_load_checkpoint(net, best_weights, best_biases);
            for (int i = 0; i < net->num_layers; i++) {
                Layer *l = &net->layers[i];
                if (l->bn.enabled && best_bn_gamma[i]) {
                    memcpy(l->bn.gamma, best_bn_gamma[i], l->bn.size * sizeof(float));
                    memcpy(l->bn.beta, best_bn_beta[i], l->bn.size * sizeof(float));
                    memcpy(l->bn.running_mean, best_bn_mean[i], l->bn.size * sizeof(float));
                    memcpy(l->bn.running_var, best_bn_var[i], l->bn.size * sizeof(float));
                }
            }
            log_info("Pesos restaurados da epoch %d (val_f1=%.4f)",
                     history->best_epoch + 1, best_val_macro_f1);
        }
    } else {
        /* Not enough SWA snapshots, use best checkpoint */
        mlp_load_checkpoint(net, best_weights, best_biases);
        for (int i = 0; i < net->num_layers; i++) {
            Layer *l = &net->layers[i];
            if (l->bn.enabled && best_bn_gamma[i]) {
                memcpy(l->bn.gamma, best_bn_gamma[i], l->bn.size * sizeof(float));
                memcpy(l->bn.beta, best_bn_beta[i], l->bn.size * sizeof(float));
                memcpy(l->bn.running_mean, best_bn_mean[i], l->bn.size * sizeof(float));
                memcpy(l->bn.running_var, best_bn_var[i], l->bn.size * sizeof(float));
            }
        }
        log_info("Pesos restaurados da epoch %d (val_f1=%.4f)",
                 history->best_epoch + 1, best_val_macro_f1);
    }

    for (int i = 0; i < net->num_layers; i++) {
        free(best_weights[i]);
        free(best_biases[i]);
        free(best_bn_gamma[i]);
        free(best_bn_beta[i]);
        free(best_bn_mean[i]);
        free(best_bn_var[i]);
        free(swa_weights[i]);
        free(swa_biases[i]);
    }
    free(indices);
    free(x_aug);
    return 0;
}

float mlp_evaluate(MLP *net, const float *x, const int *y, int n,
                   int num_features, float *loss_out)
{
    float output[MLP_OUTPUT_SIZE];
    float one_hot[MLP_OUTPUT_SIZE];

    int correct = 0;
    float total_loss = 0.0f;

    for (int i = 0; i < n; i++) {
        mlp_forward(net, &x[i * num_features], output, 0);

        memset(one_hot, 0, sizeof(one_hot));
        one_hot[y[i]] = 1.0f;
        total_loss += mlp_loss(output, one_hot, 1.0f);

        if (argmax(output, MLP_OUTPUT_SIZE) == y[i]) correct++;
    }

    if (loss_out) *loss_out = total_loss / n;
    return (float)correct / n;
}

void train_history_free(TrainHistory *h)
{
    if (h && h->epochs) {
        free(h->epochs);
        h->epochs = NULL;
    }
}
