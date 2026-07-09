#include "mlxbridge/mlxbridge.h"
#include "core/log.h"

#include <string.h>

mlx_stream mlxbridge_gpu_stream(void) {
    return mlx_default_gpu_stream_new();
}

bool mlxbridge_check(int ret, const char *context) {
    if (ret != 0) {
        log_error("mlx-c call failed: %s (ret=%d)", context, ret);
        return false;
    }
    return true;
}

int mlxbridge_get_shape(mlx_array arr, int *dims, int max_dims) {
    size_t ndim = mlx_array_ndim(arr);
    if ((int)ndim > max_dims) {
        return -1;
    }
    if (ndim > 0) {
        const int *shape = mlx_array_shape(arr);
        memcpy(dims, shape, ndim * sizeof(int));
    }
    return (int)ndim;
}

void mlxbridge_print_array(const char *label, mlx_array arr) {
    mlx_array_eval(arr);
    mlx_string s = mlx_string_new();
    mlx_array_tostring(&s, arr);
    log_debug("%s: %s", label, mlx_string_data(s));
    mlx_string_free(s);
}
