#ifndef MLXD_MLXBRIDGE_H
#define MLXD_MLXBRIDGE_H

#include <mlx/c/mlx.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

mlx_stream mlxbridge_gpu_stream(void);
void mlxbridge_print_array(const char *label, mlx_array arr);
int mlxbridge_get_shape(mlx_array arr, int *dims, int max_dims);
bool mlxbridge_check(int ret, const char *context);

#define MLXB_CHECK(call) mlxbridge_check((call), #call)

/* All frees engine-thread-only.
   Loads pinned to CPU stream internally (mlx_load_safetensors on GPU aborts).
   mlx_map_string_to_array_insert copies the underlying C++ array handle
   (insert_or_assign in mlx-c map.cpp), so the caller retains ownership of
   its mlx_array and must free it independently. */
mlx_stream mlxbridge_cpu_stream(void);
int    mlxbridge_load_safetensors(mlx_map_string_to_array *params,
                                  mlx_map_string_to_string *meta,
                                  const char *path);
int    mlxbridge_map_get(mlx_array *out, mlx_map_string_to_array map,
                         const char *key);
size_t mlxbridge_map_count(mlx_map_string_to_array map);
void   mlxbridge_map_free(mlx_map_string_to_array params,
                           mlx_map_string_to_string meta);

int    mlxbridge_async_eval(mlx_array a);
/* Async-eval n arrays. n == 0 is a success no-op. Returns -1 and logs on failure. */
int    mlxbridge_async_eval_n(const mlx_array *arrs, size_t n);
int    mlxbridge_item_int32(int32_t *out, mlx_array a);
int    mlxbridge_synchronize(mlx_stream s);

int    mlxbridge_set_wired_limit(size_t *old, size_t bytes);
int    mlxbridge_set_cache_limit(size_t *old, size_t bytes);
int    mlxbridge_get_active_memory(size_t *out);
int    mlxbridge_get_peak_memory(size_t *out);
int    mlxbridge_clear_cache(void);

#endif
