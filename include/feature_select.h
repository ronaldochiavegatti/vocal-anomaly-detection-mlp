#ifndef FEATURE_SELECT_H
#define FEATURE_SELECT_H

/*
 * feature_select.h - Persistencia dos indices de features selecionadas por fold
 *
 * Formato binario: [int n_selected][int idx_0]...[int idx_{n-1}]
 */

/* Salva n indices em path. Retorna 0 em sucesso, -1 em erro. */
int selected_save(const char *path, const int *indices, int n);

/* Carrega indices de path. Preenche indices[] e *n. Retorna 0 em sucesso, -1 em erro. */
int selected_load(const char *path, int *indices, int *n);

#endif /* FEATURE_SELECT_H */
