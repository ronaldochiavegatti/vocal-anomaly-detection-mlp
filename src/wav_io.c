/*
 * wav_io.c - Leitor de arquivos WAV (PCM 16-bit, mono)
 *
 * Escaneia chunks RIFF sequencialmente para localizar 'fmt ' e 'data',
 * em vez de assumir offset fixo de 44 bytes. Isso eh necessario porque
 * os WAVs do dataset possuem chunk LIST/INFO entre fmt e data.
 */

#include "wav_io.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Le 4 bytes little-endian como uint32 */
static uint32_t read_u32_le(FILE *f)
{
    unsigned char buf[4];
    if (fread(buf, 1, 4, f) != 4) return 0;
    return (uint32_t)buf[0]
         | ((uint32_t)buf[1] << 8)
         | ((uint32_t)buf[2] << 16)
         | ((uint32_t)buf[3] << 24);
}

/* Le 2 bytes little-endian como uint16 */
static uint16_t read_u16_le(FILE *f)
{
    unsigned char buf[2];
    if (fread(buf, 1, 2, f) != 2) return 0;
    return (uint16_t)buf[0] | ((uint16_t)buf[1] << 8);
}

int wav_read(const char *path, WavFile *wav)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        log_error("Nao foi possivel abrir: %s", path);
        return -1;
    }

    memset(wav, 0, sizeof(WavFile));

    /* Verificar header RIFF */
    char riff_id[4];
    if (fread(riff_id, 1, 4, f) != 4 || memcmp(riff_id, "RIFF", 4) != 0) {
        log_error("Nao eh arquivo RIFF: %s", path);
        fclose(f);
        return -1;
    }

    read_u32_le(f);  /* tamanho total do arquivo - 8 (ignorado) */

    char wave_id[4];
    if (fread(wave_id, 1, 4, f) != 4 || memcmp(wave_id, "WAVE", 4) != 0) {
        log_error("Nao eh formato WAVE: %s", path);
        fclose(f);
        return -1;
    }

    /* Escanear chunks ate encontrar 'fmt ' e 'data' */
    int found_fmt = 0;
    int found_data = 0;
    uint32_t data_size = 0;

    while (!found_data) {
        char chunk_id[4];
        if (fread(chunk_id, 1, 4, f) != 4) break;

        uint32_t chunk_size = read_u32_le(f);

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            /* Ler parametros de formato */
            uint16_t audio_format = read_u16_le(f);
            wav->num_channels = read_u16_le(f);
            wav->sample_rate = (int)read_u32_le(f);
            read_u32_le(f);  /* byte rate */
            read_u16_le(f);  /* block align */
            wav->bits_per_sample = read_u16_le(f);

            if (audio_format != 1) {
                log_error("Formato nao-PCM (audio_format=%d): %s", audio_format, path);
                fclose(f);
                return -1;
            }

            /* Pular bytes extras do chunk fmt (se existirem) */
            if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
            found_fmt = 1;

        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = 1;
            /* Nao pula: os dados vem logo em seguida */

        } else {
            /* Chunk desconhecido (LIST, INFO, etc): pular */
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data) {
        log_error("Chunks fmt/data nao encontrados: %s", path);
        fclose(f);
        return -1;
    }

    if (wav->bits_per_sample != 16) {
        log_error("Apenas 16-bit suportado (encontrado %d-bit): %s",
                  wav->bits_per_sample, path);
        fclose(f);
        return -1;
    }

    /* Calcular numero de amostras */
    int bytes_per_sample = wav->bits_per_sample / 8;
    wav->num_samples = (int)(data_size / (bytes_per_sample * wav->num_channels));

    /* Ler amostras brutas */
    int16_t *raw = (int16_t *)safe_malloc(wav->num_samples * sizeof(int16_t));
    size_t read_count = fread(raw, sizeof(int16_t), wav->num_samples, f);
    if ((int)read_count < wav->num_samples) {
        /* Arquivo pode ser menor que o declarado; ajustar */
        wav->num_samples = (int)read_count;
    }

    fclose(f);

    /* Converter int16 para float normalizado [-1.0, 1.0] */
    wav->samples = (float *)safe_malloc(wav->num_samples * sizeof(float));
    for (int i = 0; i < wav->num_samples; i++) {
        wav->samples[i] = (float)raw[i] / 32768.0f;
    }

    free(raw);

    log_debug("WAV lido: %s (%d amostras, %d Hz, %d-bit, %d ch)",
              path, wav->num_samples, wav->sample_rate,
              wav->bits_per_sample, wav->num_channels);

    return 0;
}

void wav_free(WavFile *wav)
{
    if (wav && wav->samples) {
        free(wav->samples);
        wav->samples = NULL;
        wav->num_samples = 0;
    }
}
