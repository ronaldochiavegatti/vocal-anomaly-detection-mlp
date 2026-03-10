#include "logreg.h"
#include "config.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void lr_init(LRModel *lr, int input_size, int n_classes)
{
    lr->input_size = input_size;
    lr->n_classes  = n_classes;
    lr->timestep   = 0;

    int nw = n_classes * input_size;
    lr->W     = (float *)safe_malloc(nw * sizeof(float));
    lr->b     = (float *)safe_calloc(n_classes, sizeof(float));
    lr->grad_W = (float *)safe_calloc(nw, sizeof(float));
    lr->grad_b = (float *)safe_calloc(n_classes, sizeof(float));
    lr->m_W   = (float *)safe_calloc(nw, sizeof(float));
    lr->v_W   = (float *)safe_calloc(nw, sizeof(float));
    lr->m_b   = (float *)safe_calloc(n_classes, sizeof(float));
    lr->v_b   = (float *)safe_calloc(n_classes, sizeof(float));

    /* He initialization (sqrt(2/fan_in)) */
    float scale = sqrtf(2.0f / input_size);
    for (int i = 0; i < nw; i++)
        lr->W[i] = rng_normal() * scale;
}

void lr_free(LRModel *lr)
{
    free(lr->W); free(lr->b);
    free(lr->grad_W); free(lr->grad_b);
    free(lr->m_W); free(lr->v_W);
    free(lr->m_b); free(lr->v_b);
}

/* Softmax in-place */
static void softmax(float *z, int n)
{
    float max_z = z[0];
    for (int i = 1; i < n; i++) if (z[i] > max_z) max_z = z[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { z[i] = expf(z[i] - max_z); sum += z[i]; }
    for (int i = 0; i < n; i++) z[i] /= sum;
}

/* Forward: fills out[n_classes] with softmax probabilities */
static void lr_forward(const LRModel *lr, const float *x, float *out)
{
    for (int c = 0; c < lr->n_classes; c++) {
        float s = lr->b[c];
        const float *w = &lr->W[c * lr->input_size];
        for (int j = 0; j < lr->input_size; j++) s += w[j] * x[j];
        out[c] = s;
    }
    softmax(out, lr->n_classes);
}

/* Cosine annealing LR — same as mlp_train.c */
static float cosine_lr(int epoch, int max_epochs)
{
    float progress = (float)epoch / max_epochs;
    return LR_MIN + 0.5f * (LEARNING_RATE - LR_MIN) *
           (1.0f + cosf((float)M_PI * progress));
}

void lr_train(LRModel *lr,
              const float *train_x, const int *train_y, int n_train,
              const float *val_x,   const int *val_y,   int n_val,
              int num_features, int *y_pred_out)
{
    float *out      = (float *)safe_malloc(lr->n_classes * sizeof(float));
    float *x_aug    = (float *)safe_malloc(num_features * sizeof(float));
    int   *indices  = (int   *)safe_malloc(n_train * sizeof(int));
    for (int i = 0; i < n_train; i++) indices[i] = i;

    float best_val_acc = -1.0f;
    int   patience     = 0;
    int   nw           = lr->n_classes * num_features;

    /* Best checkpoint */
    float *best_W = (float *)safe_malloc(nw * sizeof(float));
    float *best_b = (float *)safe_malloc(lr->n_classes * sizeof(float));
    memcpy(best_W, lr->W, nw * sizeof(float));
    memcpy(best_b, lr->b, lr->n_classes * sizeof(float));

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        float lr_t = cosine_lr(epoch, MAX_EPOCHS);

        rng_shuffle_int(indices, n_train);

        int num_batches = (n_train + BATCH_SIZE - 1) / BATCH_SIZE;

        for (int bk = 0; bk < num_batches; bk++) {
            int bstart = bk * BATCH_SIZE;
            int bend   = bstart + BATCH_SIZE;
            if (bend > n_train) bend = n_train;
            int bsize  = bend - bstart;

            /* Zero gradients */
            memset(lr->grad_W, 0, nw * sizeof(float));
            memset(lr->grad_b, 0, lr->n_classes * sizeof(float));

            for (int s = bstart; s < bend; s++) {
                int idx = indices[s];
                const float *x = &train_x[idx * num_features];
                int y = train_y[idx];

                /* Noise injection */
                for (int f = 0; f < num_features; f++)
                    x_aug[f] = x[f] + NOISE_STDDEV * rng_normal();

                lr_forward(lr, x_aug, out);

                /* Class weight */
                static const float cw[NUM_CLASSES] = {
                    CLASS_WEIGHT_NORMAL, CLASS_WEIGHT_LARYNGITIS, CLASS_WEIGHT_DYSPHONIA
                };
                float w = cw[y];

                /* Label smoothing */
                float smooth = LABEL_SMOOTHING;
                float off_v  = smooth / (lr->n_classes - 1);

                /* delta = w * (out - target) */
                for (int c = 0; c < lr->n_classes; c++) {
                    float target = (c == y) ? (1.0f - smooth) : off_v;
                    float delta  = w * (out[c] - target);
                    lr->grad_b[c] += delta;
                    const float *xa = x_aug;
                    float *gw = &lr->grad_W[c * num_features];
                    for (int j = 0; j < num_features; j++)
                        gw[j] += delta * xa[j];
                }
            }

            /* Average gradients + L2 */
            lr->timestep++;
            float bc1 = 1.0f - powf(ADAM_BETA1, (float)lr->timestep);
            float bc2 = 1.0f - powf(ADAM_BETA2, (float)lr->timestep);

            for (int i = 0; i < nw; i++) {
                lr->grad_W[i] = lr->grad_W[i] / bsize + L2_LAMBDA * lr->W[i];
                lr->m_W[i] = ADAM_BETA1 * lr->m_W[i] + (1.0f - ADAM_BETA1) * lr->grad_W[i];
                lr->v_W[i] = ADAM_BETA2 * lr->v_W[i] + (1.0f - ADAM_BETA2) * lr->grad_W[i] * lr->grad_W[i];
                float mh = lr->m_W[i] / bc1;
                float vh = lr->v_W[i] / bc2;
                lr->W[i] -= lr_t * mh / (sqrtf(vh) + ADAM_EPSILON);
            }
            for (int c = 0; c < lr->n_classes; c++) {
                lr->grad_b[c] /= bsize;
                lr->m_b[c] = ADAM_BETA1 * lr->m_b[c] + (1.0f - ADAM_BETA1) * lr->grad_b[c];
                lr->v_b[c] = ADAM_BETA2 * lr->v_b[c] + (1.0f - ADAM_BETA2) * lr->grad_b[c] * lr->grad_b[c];
                float mh = lr->m_b[c] / bc1;
                float vh = lr->v_b[c] / bc2;
                lr->b[c] -= lr_t * mh / (sqrtf(vh) + ADAM_EPSILON);
            }
        }

        /* Validation accuracy for early stopping */
        int correct = 0;
        for (int i = 0; i < n_val; i++) {
            lr_forward(lr, &val_x[i * num_features], out);
            int pred = 0;
            for (int c = 1; c < lr->n_classes; c++)
                if (out[c] > out[pred]) pred = c;
            if (pred == val_y[i]) correct++;
        }
        float val_acc = (float)correct / n_val;

        if (val_acc > best_val_acc) {
            best_val_acc = val_acc;
            patience = 0;
            memcpy(best_W, lr->W, nw * sizeof(float));
            memcpy(best_b, lr->b, lr->n_classes * sizeof(float));
        } else {
            patience++;
            if (patience >= EARLY_STOP_PATIENCE) break;
        }
    }

    /* Restore best checkpoint */
    memcpy(lr->W, best_W, nw * sizeof(float));
    memcpy(lr->b, best_b, lr->n_classes * sizeof(float));

    /* Fill predictions */
    for (int i = 0; i < n_val; i++) {
        lr_forward(lr, &val_x[i * num_features], out);
        int pred = 0;
        for (int c = 1; c < lr->n_classes; c++)
            if (out[c] > out[pred]) pred = c;
        y_pred_out[i] = pred;
    }

    free(out); free(x_aug); free(indices); free(best_W); free(best_b);
}
