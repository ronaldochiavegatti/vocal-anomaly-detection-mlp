/*
 * dataset.c - Gerenciamento do dataset de pacientes
 *
 * Enumera subdiretorios de cada classe, verifica existencia dos
 * WAVs de vogal necessarios (a_n, i_n, u_n), e monta a lista completa.
 */

#define _POSIX_C_SOURCE 200809L

#include "dataset.h"
#include "csv_parser.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

#define INITIAL_CAPACITY 1024

/* Nomes das vogais usadas para extracao */
static const char *vowel_names[NUM_VOWELS] = { VOWEL_A, VOWEL_I, VOWEL_U };

/*
 * Verifica se um caminho eh um diretorio.
 */
static int is_directory(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return S_ISDIR(st.st_mode);
}

/*
 * Verifica se um arquivo existe.
 */
static int file_exists(const char *path)
{
    struct stat st;
    return (stat(path, &st) == 0);
}

/*
 * Enumera pacientes de um diretorio de classe.
 * Retorna o numero de pacientes adicionados.
 */
static int enumerate_class(const char *base_dir, const char *class_dir,
                           int class_label, Patient **patients,
                           int *count, int *capacity)
{
    char dir_path[4096];
    snprintf(dir_path, sizeof(dir_path), "%s/%s", base_dir, class_dir);

    DIR *dir = opendir(dir_path);
    if (!dir) {
        log_error("Nao foi possivel abrir diretorio: %s", dir_path);
        return 0;
    }

    int added = 0;
    struct dirent *entry;

    while ((entry = readdir(dir)) != NULL) {
        /* Ignorar . e .. */
        if (entry->d_name[0] == '.') continue;

        /* Verificar se eh um diretorio de paciente (numerico) */
        char patient_dir[4096];
        snprintf(patient_dir, sizeof(patient_dir), "%s/%s", dir_path, entry->d_name);
        if (!is_directory(patient_dir)) continue;

        int patient_id = atoi(entry->d_name);
        if (patient_id <= 0) continue;

        /* Verificar se os 3 WAVs de vogal existem */
        char wav_paths[NUM_VOWELS][4096];
        int all_exist = 1;
        for (int v = 0; v < NUM_VOWELS; v++) {
            snprintf(wav_paths[v], sizeof(wav_paths[v]),
                     "%s/vowels/%s-%s.wav",
                     patient_dir, entry->d_name, vowel_names[v]);
            if (!file_exists(wav_paths[v])) {
                all_exist = 0;
                break;
            }
        }

        if (!all_exist) {
            log_debug("Paciente %d: WAVs faltando, pulando", patient_id);
            continue;
        }

        /* Expandir array se necessario */
        if (*count >= *capacity) {
            *capacity *= 2;
            *patients = (Patient *)safe_realloc(
                *patients, *capacity * sizeof(Patient));
        }

        Patient *p = &(*patients)[*count];
        p->id = patient_id;
        p->class_label = class_label;
        p->sex = '?';
        for (int v = 0; v < NUM_VOWELS; v++) {
            snprintf(p->vowel_paths[v], sizeof(p->vowel_paths[v]), "%s", wav_paths[v]);
        }

        (*count)++;
        added++;
    }

    closedir(dir);
    return added;
}

/*
 * Associa metadados do CSV (sexo) com os pacientes carregados.
 */
static void associate_csv_metadata(Patient *patients, int count,
                                   const CsvData *csv)
{
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < csv->count; j++) {
            if (csv->records[j].recording_id == patients[i].id) {
                patients[i].sex = csv->records[j].sex;
                break;
            }
        }
    }
}

int dataset_load(const char *base_dir, const char *csv_path, Dataset *ds)
{
    memset(ds, 0, sizeof(Dataset));

    int capacity = INITIAL_CAPACITY;
    ds->patients = (Patient *)safe_malloc(capacity * sizeof(Patient));
    ds->count = 0;

    /* Estrutura: nome_do_diretorio -> classe */
    const char *class_dirs[NUM_CLASSES] = {
        DATA_DIR_NORMAL, DATA_DIR_LARYNGITIS, DATA_DIR_DYSPHONIA
    };

    for (int c = 0; c < NUM_CLASSES; c++) {
        int added = enumerate_class(base_dir, class_dirs[c], c,
                                    &ds->patients, &ds->count, &capacity);
        ds->class_counts[c] = added;
        log_info("Classe %d (%s): %d pacientes", c, class_dirs[c], added);
    }

    log_info("Dataset total: %d pacientes", ds->count);

    /* Associar metadados do CSV (opcional) */
    if (csv_path) {
        CsvData csv;
        if (csv_parse(csv_path, &csv) == 0) {
            associate_csv_metadata(ds->patients, ds->count, &csv);
            csv_free(&csv);
        }
    }

    return (ds->count > 0) ? 0 : -1;
}

void dataset_free(Dataset *ds)
{
    if (ds && ds->patients) {
        free(ds->patients);
        ds->patients = NULL;
        ds->count = 0;
    }
}
