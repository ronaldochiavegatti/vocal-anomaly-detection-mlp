/*
 * utils.h - Utilitarios gerais
 *
 * Alocacao segura de memoria, geracao de numeros aleatorios,
 * logging e medicao de tempo.
 */

#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdio.h>

/* ========== Alocacao segura ========== */

/* malloc que aborta em caso de falha */
void *safe_malloc(size_t size);

/* calloc que aborta em caso de falha */
void *safe_calloc(size_t count, size_t size);

/* realloc que aborta em caso de falha */
void *safe_realloc(void *ptr, size_t size);

/* ========== RNG ========== */

/* Inicializa o gerador com uma seed */
void rng_seed(unsigned int seed);

/* Retorna float uniforme em [0, 1) */
float rng_uniform(void);

/* Retorna float com distribuicao normal (media 0, desvio 1) via Box-Muller */
float rng_normal(void);

/* Embaralha array de inteiros (Fisher-Yates) */
void rng_shuffle_int(int *array, int n);

/* ========== Logging ========== */

typedef enum {
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR
} LogLevel;

/* Define nivel minimo de log */
void log_set_level(LogLevel level);

/* Define arquivo de saida do log (NULL = stderr) */
void log_set_file(FILE *f);

/* Funcoes de log */
void log_debug(const char *fmt, ...);
void log_info(const char *fmt, ...);
void log_warn(const char *fmt, ...);
void log_error(const char *fmt, ...);

/* ========== Timer ========== */

/* Retorna tempo atual em segundos (alta resolucao) */
double timer_now(void);

#endif /* UTILS_H */
