#ifndef MLXD_MLXBRIDGE_H
#define MLXD_MLXBRIDGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Opaque mlx-c handle types */
typedef void *mlx_array;
typedef void *mlx_stream;
typedef void *mlx_device;
typedef void *mlx_string;
typedef void *mlx_map_string_to_array;
typedef void *mlx_vector_array;
typedef void *mlx_closure;

typedef enum {
    MLX_BOOL,
    MLX_UINT8,
    MLX_UINT16,
    MLX_UINT32,
    MLX_UINT64,
    MLX_INT8,
    MLX_INT16,
    MLX_INT32,
    MLX_INT64,
    MLX_FLOAT16,
    MLX_BFLOAT16,
    MLX_FLOAT32,
    MLX_FLOAT64,
    MLX_COMPLEX64,
} mlx_dtype_t;

/* GPU stream helper */
mlx_stream mlxbridge_gpu_stream(void);

/* Array helpers */
void mlxbridge_print_array(mlx_array arr);
int  mlxbridge_get_shape(mlx_array arr, int *dims, int max_dims);
bool mlxbridge_check(const char *context);

/* TODO: remaining ~190 extern declarations will be added during mlxbridge migration */

#endif
