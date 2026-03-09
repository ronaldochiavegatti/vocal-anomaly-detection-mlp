/*
 * normalize.c - Normalizacao Z-score
 *
 * z = (x - media) / std
 * Se std == 0, a feature eh constante e fica zerada apos normalizacao.
 */

#include "normalize.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MIN_STD 1e-8f

void norm_fit(const float *features, int n, int num_features, NormParams *params)
{
    params->num_features = num_features;
    params->mean = (float *)safe_calloc(num_features, sizeof(float));
    params->std = (float *)safe_calloc(num_features, sizeof(float));

    /* Calcular media */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < num_features; j++) {
            params->mean[j] += features[i * num_features + j];
        }
    }
    for (int j = 0; j < num_features; j++) {
        params->mean[j] /= n;
    }

    /* Calcular desvio-padrao */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < num_features; j++) {
            float diff = features[i * num_features + j] - params->mean[j];
            params->std[j] += diff * diff;
        }
    }
    for (int j = 0; j < num_features; j++) {
        params->std[j] = sqrtf(params->std[j] / n);
        if (params->std[j] < MIN_STD) params->std[j] = MIN_STD;
    }
}

void norm_transform(float *features, int n, const NormParams *params)
{
    int nf = params->num_features;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < nf; j++) {
            features[i * nf + j] = (features[i * nf + j] - params->mean[j]) / params->std[j];
        }
    }
}

int norm_save(const NormParams *params, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    fwrite(&params->num_features, sizeof(int), 1, f);
    fwrite(params->mean, sizeof(float), params->num_features, f);
    fwrite(params->std, sizeof(float), params->num_features, f);

    fclose(f);
    return 0;
}

int norm_load(NormParams *params, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    if (fread(&params->num_features, sizeof(int), 1, f) != 1) {
        fclose(f);
        return -1;
    }
    params->mean = (float *)safe_malloc(params->num_features * sizeof(float));
    params->std = (float *)safe_malloc(params->num_features * sizeof(float));
    if (fread(params->mean, sizeof(float), params->num_features, f) != (size_t)params->num_features ||
        fread(params->std, sizeof(float), params->num_features, f) != (size_t)params->num_features) {
        free(params->mean);
        free(params->std);
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

void norm_free(NormParams *params)
{
    if (params) {
        free(params->mean);
        free(params->std);
        params->mean = NULL;
        params->std = NULL;
    }
}
