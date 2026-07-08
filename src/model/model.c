#include "model/model.h"

#include <stdlib.h>

/* TODO: implement during model module migration */

int model_config_load(model_config_t *cfg, const char *model_dir) {
    (void)cfg;
    (void)model_dir;
    return -1;
}

void model_config_free(model_config_t *cfg) {
    if (!cfg)
        return;
    free(cfg->model_type);
    free(cfg->architectures);
}

char **model_discover(int *count) {
    *count = 0;
    return NULL;
}
