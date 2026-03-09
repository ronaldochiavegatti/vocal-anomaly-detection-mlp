/*
 * dataset.h - Gerenciamento do dataset de pacientes
 *
 * Enumera diretorios de pacientes, associa com classe,
 * e monta a lista completa para treinamento.
 */

#ifndef DATASET_H
#define DATASET_H

#include "config.h"

/* Informacoes de um paciente */
typedef struct {
    int id;                           /* ID do paciente (nome do diretorio) */
    int class_label;                  /* CLASS_NORMAL, CLASS_LARYNGITIS, CLASS_DYSPHONIA */
    char sex;                         /* 'm' ou 'w', ou '?' se desconhecido */
    char vowel_paths[NUM_VOWELS][4096]; /* caminhos dos 3 WAVs (a_n, i_n, u_n) */
} Patient;

/* Dataset completo */
typedef struct {
    Patient *patients;
    int count;
    int class_counts[NUM_CLASSES];    /* contagem por classe */
} Dataset;

/*
 * Carrega o dataset a partir do diretorio base.
 * Enumera os 3 subdiretorios de classe e monta a lista de pacientes.
 *
 * base_dir: diretorio raiz do projeto (contendo saudavel/, laringite/, etc.)
 * csv_path: caminho do overview_merged.csv (para metadados, pode ser NULL)
 * ds: ponteiro para struct de saida
 *
 * Retorna 0 em sucesso, -1 em erro.
 */
int dataset_load(const char *base_dir, const char *csv_path, Dataset *ds);

/*
 * Libera a memoria do dataset.
 */
void dataset_free(Dataset *ds);

#endif /* DATASET_H */
