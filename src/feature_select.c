#include "feature_select.h"
#include <stdio.h>

int selected_save(const char *path, const int *indices, int n)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(&n, sizeof(int), 1, f);
    fwrite(indices, sizeof(int), (size_t)n, f);
    fclose(f);
    return 0;
}

int selected_load(const char *path, int *indices, int *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fread(n, sizeof(int), 1, f) != 1) { fclose(f); return -1; }
    if (fread(indices, sizeof(int), (size_t)*n, f) != (size_t)*n) { fclose(f); return -1; }
    fclose(f);
    return 0;
}
