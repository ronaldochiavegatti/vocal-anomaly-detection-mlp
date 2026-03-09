/*
 * wav_io.h - Leitor de arquivos WAV (PCM 16-bit, mono)
 *
 * Implementacao propria com escaneamento de chunks RIFF,
 * necessario porque os WAVs do Saarbruecken Voice Database
 * possuem chunk LIST/INFO entre fmt e data.
 */

#ifndef WAV_IO_H
#define WAV_IO_H

#include <stdint.h>

/* Estrutura com os dados do audio lido */
typedef struct {
    float *samples;       /* amostras normalizadas em [-1.0, 1.0] */
    int num_samples;      /* numero total de amostras */
    int sample_rate;      /* taxa de amostragem (ex: 44100) */
    int bits_per_sample;  /* bits por amostra (ex: 16) */
    int num_channels;     /* numero de canais (ex: 1) */
} WavFile;

/*
 * Le um arquivo WAV PCM.
 * Retorna 0 em sucesso, -1 em erro.
 * O chamador deve liberar wav->samples com wav_free().
 */
int wav_read(const char *path, WavFile *wav);

/*
 * Libera a memoria alocada para as amostras.
 */
void wav_free(WavFile *wav);

#endif /* WAV_IO_H */
