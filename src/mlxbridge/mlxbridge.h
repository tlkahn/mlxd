#ifndef MLXD_MLXBRIDGE_H
#define MLXD_MLXBRIDGE_H

#include <mlx/c/mlx.h>
#include <stdbool.h>

mlx_stream mlxbridge_gpu_stream(void);
void mlxbridge_print_array(const char *label, mlx_array arr);
int mlxbridge_get_shape(mlx_array arr, int *dims, int max_dims);
bool mlxbridge_check(int ret, const char *context);

#define MLXB_CHECK(call) mlxbridge_check((call), #call)

#endif
