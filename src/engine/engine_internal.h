#ifndef MLXD_ENGINE_INTERNAL_H
#define MLXD_ENGINE_INTERNAL_H

#include "mlxbridge/mlxbridge.h"
#include <stdint.h>

/* Exported for GPU unit tests (test_engine_gpu). Not part of the public engine API. */
int greedy_next_id(mlx_array logits, mlx_stream s, int32_t *id_out);

#endif
