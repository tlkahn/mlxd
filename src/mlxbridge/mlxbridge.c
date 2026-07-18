#include "mlxbridge/mlxbridge.h"
#include "core/log.h"

#include <string.h>

mlx_stream mlxbridge_gpu_stream(void) {
    return mlx_default_gpu_stream_new();
}

mlx_stream mlxbridge_cpu_stream(void) {
    return mlx_default_cpu_stream_new();
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
    if (!MLXB_CHECK(mlx_array_eval(arr))) return;
    mlx_string s = mlx_string_new();
    if (!MLXB_CHECK(mlx_array_tostring(&s, arr))) {
        mlx_string_free(s);
        return;
    }
    log_debug("%s: %s", label, mlx_string_data(s));
    mlx_string_free(s);
}

int mlxbridge_load_safetensors(mlx_map_string_to_array *params,
                               mlx_map_string_to_string *meta,
                               const char *path) {
    mlx_stream cpu = mlx_default_cpu_stream_new();
    int rc = mlx_load_safetensors(params, meta, path, cpu);
    mlx_stream_free(cpu);
    if (rc != 0) {
        log_error("mlxbridge: failed to load safetensors: %s", path);
        return -1;
    }
    return 0;
}

int mlxbridge_map_get(mlx_array *out, mlx_map_string_to_array map,
                      const char *key) {
    int rc = mlx_map_string_to_array_get(out, map, key);
    if (rc != 0)
        return -1;
    return 0;
}

size_t mlxbridge_map_count(mlx_map_string_to_array map) {
    mlx_map_string_to_array_iterator it =
        mlx_map_string_to_array_iterator_new(map);
    size_t count = 0;
    const char *key = NULL;
    mlx_array val = mlx_array_new();
    while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0 &&
           key != NULL) {
        count++;
        key = NULL;
    }
    mlx_array_free(val);
    mlx_map_string_to_array_iterator_free(it);
    return count;
}

void mlxbridge_map_free(mlx_map_string_to_array params,
                        mlx_map_string_to_string meta) {
    mlx_map_string_to_array_free(params);
    mlx_map_string_to_string_free(meta);
}

int mlxbridge_async_eval_n(const mlx_array *arrs, size_t n) {
    if (n == 0)
        return 0;
    if (!arrs)
        return -1;
    /* mlx_vector_array_new_data takes const mlx_array*; cast is API-faithful. */
    mlx_vector_array va = mlx_vector_array_new_data(arrs, n);
    int rc = mlx_async_eval(va);
    mlx_vector_array_free(va);
    if (rc != 0) {
        log_error("mlxbridge: async_eval failed");
        return -1;
    }
    return 0;
}

int mlxbridge_async_eval(mlx_array a) {
    return mlxbridge_async_eval_n(&a, 1);
}

int mlxbridge_item_int32(int32_t *out, mlx_array a) {
    if (!out) return -1;
    if (mlx_array_eval(a) != 0)
        return -1;
    if (mlx_array_item_int32(out, a) != 0)
        return -1;
    return 0;
}

int mlxbridge_synchronize(mlx_stream s) {
    if (mlx_synchronize(s) != 0) {
        log_error("mlxbridge: synchronize failed");
        return -1;
    }
    return 0;
}

int mlxbridge_set_wired_limit(size_t *old, size_t bytes) {
    size_t prev = 0;
    if (mlx_set_wired_limit(&prev, bytes) != 0) {
        log_error("mlxbridge: set_wired_limit failed");
        return -1;
    }
    if (old)
        *old = prev;
    return 0;
}

int mlxbridge_set_cache_limit(size_t *old, size_t bytes) {
    size_t prev = 0;
    if (mlx_set_cache_limit(&prev, bytes) != 0) {
        log_error("mlxbridge: set_cache_limit failed");
        return -1;
    }
    if (old)
        *old = prev;
    return 0;
}

int mlxbridge_get_active_memory(size_t *out) {
    if (mlx_get_active_memory(out) != 0) {
        log_error("mlxbridge: get_active_memory failed");
        return -1;
    }
    return 0;
}

int mlxbridge_get_peak_memory(size_t *out) {
    if (mlx_get_peak_memory(out) != 0) {
        log_error("mlxbridge: get_peak_memory failed");
        return -1;
    }
    return 0;
}

int mlxbridge_clear_cache(void) {
    if (mlx_clear_cache() != 0) {
        log_error("mlxbridge: clear_cache failed");
        return -1;
    }
    return 0;
}
