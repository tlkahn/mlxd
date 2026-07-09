#include "mlxbridge/mlxbridge.h"

#include <assert.h>
#include <stdio.h>

static void test_extern_presence(void) {
    /* Take address of key mlx-c symbols across subsystems to verify
       they resolve at link time. */
    void *syms[] = {
        /* array */
        (void *)mlx_array_new,
        (void *)mlx_array_new_int,
        (void *)mlx_array_new_float,
        (void *)mlx_array_new_data,
        (void *)mlx_array_free,
        (void *)mlx_array_eval,
        (void *)mlx_array_ndim,
        (void *)mlx_array_shape,
        (void *)mlx_array_size,
        (void *)mlx_array_dtype,
        (void *)mlx_array_itemsize,
        (void *)mlx_array_tostring,
        (void *)mlx_array_set,
        (void *)mlx_array_item_int32,
        (void *)mlx_array_item_float32,
        (void *)mlx_array_item_bool,
        (void *)mlx_array_data_float32,
        /* ops */
        (void *)mlx_add,
        (void *)mlx_matmul,
        (void *)mlx_sqrt,
        (void *)mlx_less,
        (void *)mlx_sum_axis,
        (void *)mlx_mean,
        (void *)mlx_arange,
        (void *)mlx_reshape,
        (void *)mlx_transpose,
        (void *)mlx_slice,
        (void *)mlx_concatenate_axis,
        (void *)mlx_broadcast_to,
        (void *)mlx_ones,
        (void *)mlx_astype,
        (void *)mlx_split,
        (void *)mlx_stack_axis,
        (void *)mlx_contiguous,
        (void *)mlx_cummax,
        (void *)mlx_std,
        (void *)mlx_std_axes,
        (void *)mlx_mean_axes,
        (void *)mlx_all,
        (void *)mlx_logsumexp_axis,
        (void *)mlx_quantize,
        (void *)mlx_dequantize,
        (void *)mlx_quantized_matmul,
        (void *)mlx_gather_qmm,
        (void *)mlx_gather_mm,
        (void *)mlx_conv1d,
        (void *)mlx_conv2d,
        (void *)mlx_conv3d,
        (void *)mlx_conv_transpose1d,
        (void *)mlx_conv_transpose2d,
        (void *)mlx_conv_transpose3d,
        /* stream */
        (void *)mlx_default_cpu_stream_new,
        (void *)mlx_default_gpu_stream_new,
        (void *)mlx_stream_new_device,
        (void *)mlx_stream_free,
        (void *)mlx_synchronize,
        /* device */
        (void *)mlx_device_new_type,
        (void *)mlx_device_free,
        (void *)mlx_set_default_device,
        (void *)mlx_get_default_device,
        /* metal */
        (void *)mlx_metal_is_available,
        /* fast */
        (void *)mlx_fast_rms_norm,
        (void *)mlx_fast_layer_norm,
        (void *)mlx_fast_rope,
        (void *)mlx_fast_rope_dynamic,
        (void *)mlx_fast_scaled_dot_product_attention,
        /* io */
        (void *)mlx_save_safetensors,
        (void *)mlx_load_safetensors,
        /* closure + compile */
        (void *)mlx_closure_new_func_payload,
        (void *)mlx_closure_apply,
        (void *)mlx_closure_free,
        (void *)mlx_compile,
        (void *)mlx_detail_compile_clear_cache,
        /* map */
        (void *)mlx_map_string_to_array_new,
        (void *)mlx_map_string_to_array_free,
        (void *)mlx_map_string_to_array_insert,
        (void *)mlx_map_string_to_array_get,
        (void *)mlx_map_string_to_array_iterator_new,
        (void *)mlx_map_string_to_array_iterator_free,
        (void *)mlx_map_string_to_array_iterator_next,
        (void *)mlx_map_string_to_string_new,
        (void *)mlx_map_string_to_string_free,
        /* vector */
        (void *)mlx_vector_array_new,
        (void *)mlx_vector_array_new_value,
        (void *)mlx_vector_array_append_value,
        (void *)mlx_vector_array_size,
        (void *)mlx_vector_array_get,
        (void *)mlx_vector_array_free,
        /* random */
        (void *)mlx_random_key,
        (void *)mlx_random_categorical,
        (void *)mlx_random_normal,
        (void *)mlx_random_randint,
        /* memory */
        (void *)mlx_clear_cache,
        (void *)mlx_get_active_memory,
        (void *)mlx_get_peak_memory,
        (void *)mlx_reset_peak_memory,
        (void *)mlx_set_cache_limit,
        (void *)mlx_set_memory_limit,
        (void *)mlx_set_wired_limit,
        /* transforms */
        (void *)mlx_eval,
        (void *)mlx_async_eval,
        /* fft */
        (void *)mlx_fft_rfft,
        /* error */
        (void *)mlx_set_error_handler,
        /* version */
        (void *)mlx_version,
        /* custom metal kernel */
        (void *)mlx_fast_metal_kernel_new,
        (void *)mlx_fast_metal_kernel_free,
        (void *)mlx_fast_metal_kernel_apply,
        (void *)mlx_fast_metal_kernel_config_new,
        (void *)mlx_fast_metal_kernel_config_free,
        (void *)mlx_fast_metal_kernel_config_add_output_arg,
        (void *)mlx_fast_metal_kernel_config_set_grid,
        (void *)mlx_fast_metal_kernel_config_set_thread_group,
        /* device info */
        (void *)mlx_device_info_new,
        (void *)mlx_device_info_get,
        (void *)mlx_device_info_free,
        (void *)mlx_device_info_get_size,
        /* string */
        (void *)mlx_string_new,
        (void *)mlx_string_data,
        (void *)mlx_string_free,
        /* vector string */
        (void *)mlx_vector_string_new,
        (void *)mlx_vector_string_new_data,
        (void *)mlx_vector_string_free,
    };
    for (size_t i = 0; i < sizeof(syms) / sizeof(syms[0]); i++) {
        assert(syms[i] != NULL);
    }
}

static void test_check_success(void) {
    assert(mlxbridge_check(0, "test_ok") == true);
}

static void test_check_failure(void) {
    assert(mlxbridge_check(1, "test_fail") == false);
}

static void test_check_macro(void) {
    int zero = 0;
    assert(MLXB_CHECK(zero));
}

int main(void) {
    test_extern_presence();
    test_check_success();
    test_check_failure();
    test_check_macro();
    printf("test_mlxbridge: all passed\n");
    return 0;
}
