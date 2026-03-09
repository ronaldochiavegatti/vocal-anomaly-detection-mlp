/*
 * csv_parser.c - Parser CSV compativel com RFC 4180
 *
 * Lida com campos entre aspas duplas (necessario porque o campo
 * Diagnose pode conter virgulas). Extrai apenas os campos relevantes:
 * AufnahmeID (col 0), SprecherID (col 4), Geschlecht (col 6),
 * Pathologien (col 7).
 */

#include "csv_parser.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 2048
#define MAX_FIELDS 16
#define INITIAL_CAPACITY 1200

/*
 * Parseia uma linha CSV respeitando campos entre aspas.
 * Preenche fields[] com ponteiros para o inicio de cada campo (in-place).
 * Retorna o numero de campos encontrados.
 */
static int parse_csv_line(char *line, char *fields[], int max_fields)
{
    int count = 0;
    char *p = line;

    while (*p && count < max_fields) {
        if (*p == '"') {
            /* Campo entre aspas */
            p++;
            fields[count++] = p;
            /* Buscar aspas de fechamento */
            while (*p) {
                if (*p == '"') {
                    if (*(p + 1) == '"') {
                        /* Aspas escapadas: pular */
                        p += 2;
                    } else {
                        /* Fim do campo */
                        *p = '\0';
                        p++;
                        if (*p == ',') p++;
                        break;
                    }
                } else {
                    p++;
                }
            }
        } else {
            /* Campo sem aspas */
            fields[count++] = p;
            while (*p && *p != ',' && *p != '\n' && *p != '\r') p++;
            if (*p == ',') {
                *p = '\0';
                p++;
            } else {
                /* Fim da linha: remover \n\r */
                if (*p == '\r' || *p == '\n') *p = '\0';
                break;
            }
        }
    }

    return count;
}

int csv_parse(const char *path, CsvData *data)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        log_error("Nao foi possivel abrir CSV: %s", path);
        return -1;
    }

    data->records = (CsvRecord *)safe_malloc(INITIAL_CAPACITY * sizeof(CsvRecord));
    data->count = 0;
    int capacity = INITIAL_CAPACITY;

    char line[MAX_LINE];

    /* Pular cabecalho */
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return -1;
    }

    while (fgets(line, sizeof(line), f)) {
        char *fields[MAX_FIELDS];
        int nf = parse_csv_line(line, fields, MAX_FIELDS);

        /* Precisamos de pelo menos 8 campos */
        if (nf < 8) continue;

        /* Expandir array se necessario */
        if (data->count >= capacity) {
            capacity *= 2;
            data->records = (CsvRecord *)safe_realloc(
                data->records, capacity * sizeof(CsvRecord));
        }

        CsvRecord *rec = &data->records[data->count];
        rec->recording_id = atoi(fields[0]);
        rec->speaker_id = atoi(fields[4]);
        rec->sex = fields[6][0];
        strncpy(rec->pathology, fields[7], sizeof(rec->pathology) - 1);
        rec->pathology[sizeof(rec->pathology) - 1] = '\0';

        /* Remover \n\r do final da patologia */
        int len = (int)strlen(rec->pathology);
        while (len > 0 && (rec->pathology[len - 1] == '\n' ||
                           rec->pathology[len - 1] == '\r')) {
            rec->pathology[--len] = '\0';
        }

        data->count++;
    }

    fclose(f);
    log_info("CSV lido: %d registros de %s", data->count, path);
    return 0;
}

void csv_free(CsvData *data)
{
    if (data && data->records) {
        free(data->records);
        data->records = NULL;
        data->count = 0;
    }
}
