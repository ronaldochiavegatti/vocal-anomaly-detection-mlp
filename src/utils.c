/*
 * utils.c - Utilitarios gerais
 *
 * Implementacao de alocacao segura, RNG, logging e timer.
 */

#define _POSIX_C_SOURCE 199309L

#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ========== Alocacao segura ========== */

void *safe_malloc(size_t size)
{
    void *ptr = malloc(size);
    if (!ptr && size > 0) {
        fprintf(stderr, "[FATAL] malloc falhou para %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_calloc(size_t count, size_t size)
{
    void *ptr = calloc(count, size);
    if (!ptr && count > 0 && size > 0) {
        fprintf(stderr, "[FATAL] calloc falhou para %zu x %zu bytes\n", count, size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_realloc(void *ptr, size_t size)
{
    void *new_ptr = realloc(ptr, size);
    if (!new_ptr && size > 0) {
        fprintf(stderr, "[FATAL] realloc falhou para %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return new_ptr;
}

/* ========== RNG ========== */

/* Estado interno do gerador (xorshift32 para reproducibilidade) */
static unsigned int rng_state = 42;

void rng_seed(unsigned int seed)
{
    rng_state = seed;
    if (rng_state == 0) rng_state = 1;  /* xorshift nao aceita 0 */
}

/* xorshift32 - rapido e com periodo de 2^32 - 1 */
static unsigned int xorshift32(void)
{
    unsigned int x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

float rng_uniform(void)
{
    return (float)(xorshift32() & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

float rng_normal(void)
{
    /* Box-Muller: gera 2 normais, retorna 1 */
    static int has_spare = 0;
    static float spare;

    if (has_spare) {
        has_spare = 0;
        return spare;
    }

    float u, v, s;
    do {
        u = 2.0f * rng_uniform() - 1.0f;
        v = 2.0f * rng_uniform() - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);

    float mul = sqrtf(-2.0f * logf(s) / s);
    spare = v * mul;
    has_spare = 1;
    return u * mul;
}

void rng_shuffle_int(int *array, int n)
{
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(xorshift32() % (unsigned int)(i + 1));
        int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

/* ========== Logging ========== */

static LogLevel current_log_level = LOG_INFO;
static FILE *log_file = NULL;

void log_set_level(LogLevel level)
{
    current_log_level = level;
}

void log_set_file(FILE *f)
{
    log_file = f;
}

static const char *level_strings[] = {
    "DEBUG", "INFO", "WARN", "ERROR"
};

static void log_message(LogLevel level, const char *fmt, va_list args)
{
    if (level < current_log_level) return;

    FILE *out = log_file ? log_file : stderr;

    /* Timestamp */
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char time_buf[20];
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);

    fprintf(out, "[%s] [%s] ", time_buf, level_strings[level]);
    vfprintf(out, fmt, args);
    fprintf(out, "\n");
    fflush(out);
}

void log_debug(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    log_message(LOG_DEBUG, fmt, args);
    va_end(args);
}

void log_info(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    log_message(LOG_INFO, fmt, args);
    va_end(args);
}

void log_warn(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    log_message(LOG_WARN, fmt, args);
    va_end(args);
}

void log_error(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    log_message(LOG_ERROR, fmt, args);
    va_end(args);
}

/* ========== Timer ========== */

double timer_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
