/*
 * csv_parser.h - Parser CSV compativel com RFC 4180
 *
 * Le o arquivo overview_merged.csv e extrai metadados dos pacientes:
 * SprecherID (ID do paciente), Geschlecht (sexo), Pathologien (patologia).
 */

#ifndef CSV_PARSER_H
#define CSV_PARSER_H

/* Registro de metadados de um paciente no CSV */
typedef struct {
    int recording_id;      /* AufnahmeID */
    int speaker_id;        /* SprecherID */
    char sex;              /* 'm' ou 'w' (Geschlecht) */
    char pathology[64];    /* Pathologien */
} CsvRecord;

/* Resultado do parsing do CSV completo */
typedef struct {
    CsvRecord *records;
    int count;
} CsvData;

/*
 * Le e parseia o arquivo CSV.
 * Retorna 0 em sucesso, -1 em erro.
 * O chamador deve liberar com csv_free().
 */
int csv_parse(const char *path, CsvData *data);

/*
 * Libera a memoria alocada pelo parser.
 */
void csv_free(CsvData *data);

#endif /* CSV_PARSER_H */
