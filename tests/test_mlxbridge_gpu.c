#include "mlxbridge/mlxbridge.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* ---- Task 3: gpu_stream helper ---- */

static void test_gpu_stream(void) {
    mlx_stream s = mlxbridge_gpu_stream();
    assert(s.ctx != NULL);
    mlx_stream_free(s);
}

/* ---- Task 4: get_shape helper ---- */

static void test_get_shape(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    float data[] = {1, 2, 3, 4, 5, 6};
    int shape[] = {2, 3};
    mlx_array arr = mlx_array_new_data(data, shape, 2, MLX_FLOAT32);

    int dims[8];
    int ndim = mlxbridge_get_shape(arr, dims, 8);
    assert(ndim == 2);
    assert(dims[0] == 2);
    assert(dims[1] == 3);

    mlx_array scalar = mlx_array_new_int(42);
    ndim = mlxbridge_get_shape(scalar, dims, 8);
    assert(ndim == 0);

    ndim = mlxbridge_get_shape(arr, dims, 1);
    assert(ndim == -1);

    mlx_array_free(scalar);
    mlx_array_free(arr);
    mlx_stream_free(s);
}

/* ---- Task 5: print_array helper ---- */

static void test_print_array(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    mlx_array scalar = mlx_array_new_float(3.14f);
    mlxbridge_print_array("scalar", scalar);

    float data[] = {1, 2, 3};
    mlx_array vec = mlx_array_new_data(data, (int[]){3}, 1, MLX_FLOAT32);
    mlxbridge_print_array("vector", vec);

    mlx_array_free(vec);
    mlx_array_free(scalar);
    mlx_stream_free(s);
}

/* ---- Task 6: version, device, stream, metal ---- */

static void test_version(void) {
    mlx_string ver = mlx_string_new();
    assert(MLXB_CHECK(mlx_version(&ver)));
    const char *data = mlx_string_data(ver);
    assert(data != NULL);
    assert(strlen(data) > 0);
    mlx_string_free(ver);
}

static void test_device(void) {
    mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
    assert(gpu.ctx != NULL);
    assert(MLXB_CHECK(mlx_set_default_device(gpu)));

    mlx_device got = mlx_device_new();
    assert(MLXB_CHECK(mlx_get_default_device(&got)));
    assert(got.ctx != NULL);

    mlx_device_free(got);
    mlx_device_free(gpu);
}

static void test_stream(void) {
    mlx_stream cpu = mlx_default_cpu_stream_new();
    assert(cpu.ctx != NULL);
    assert(MLXB_CHECK(mlx_synchronize(cpu)));

    mlx_device gpu_dev = mlx_device_new_type(MLX_GPU, 0);
    mlx_stream gpu = mlx_stream_new_device(gpu_dev);
    assert(gpu.ctx != NULL);
    assert(MLXB_CHECK(mlx_synchronize(gpu)));

    mlx_stream_free(gpu);
    mlx_device_free(gpu_dev);
    mlx_stream_free(cpu);
}

static void test_metal_available(void) {
    bool avail = false;
    assert(MLXB_CHECK(mlx_metal_is_available(&avail)));
    assert(avail);
}

/* ---- Task 7: array basics ---- */

static void test_array_basics(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    mlx_array iscalar = mlx_array_new_int(42);
    assert(MLXB_CHECK(mlx_array_eval(iscalar)));
    int ival = 0;
    assert(MLXB_CHECK(mlx_array_item_int32(&ival, iscalar)));
    assert(ival == 42);

    float data[] = {1, 2, 3, 4, 5, 6};
    int shape[] = {2, 3};
    mlx_array arr = mlx_array_new_data(data, shape, 2, MLX_FLOAT32);
    assert(MLXB_CHECK(mlx_array_eval(arr)));
    assert(mlx_array_ndim(arr) == 2);
    assert(mlx_array_shape(arr)[0] == 2);
    assert(mlx_array_shape(arr)[1] == 3);
    assert(mlx_array_size(arr) == 6);
    assert(mlx_array_dtype(arr) == MLX_FLOAT32);
    assert(mlx_array_itemsize(arr) == 4);
    const float *raw = mlx_array_data_float32(arr);
    assert(raw != NULL);
    for (int i = 0; i < 6; i++) {
        assert(raw[i] == data[i]);
    }

    mlx_array casted = mlx_array_new();
    assert(MLXB_CHECK(mlx_astype(&casted, iscalar, MLX_FLOAT32, s)));
    assert(MLXB_CHECK(mlx_array_eval(casted)));
    assert(mlx_array_dtype(casted) == MLX_FLOAT32);
    float fval = 0;
    assert(MLXB_CHECK(mlx_array_item_float32(&fval, casted)));
    assert(fval == 42.0f);

    mlx_array copy = mlx_array_new();
    assert(MLXB_CHECK(mlx_array_set(&copy, iscalar)));
    assert(MLXB_CHECK(mlx_array_eval(copy)));
    int cval = 0;
    assert(MLXB_CHECK(mlx_array_item_int32(&cval, copy)));
    assert(cval == 42);

    mlx_array_free(copy);
    mlx_array_free(casted);
    mlx_array_free(arr);
    mlx_array_free(iscalar);
    mlx_stream_free(s);
}

/* ---- Task 8: arithmetic ops ---- */

static void test_ops(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    mlx_array a = mlx_array_new_int(2);
    mlx_array b = mlx_array_new_int(3);

    mlx_array sum = mlx_array_new();
    assert(MLXB_CHECK(mlx_add(&sum, a, b, s)));
    assert(MLXB_CHECK(mlx_array_eval(sum)));
    int v = 0;
    assert(MLXB_CHECK(mlx_array_item_int32(&v, sum)));
    assert(v == 5);

    int ones_shape[] = {2, 2};
    mlx_array m1 = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&m1, ones_shape, 2, MLX_FLOAT32, s)));
    mlx_array mm = mlx_array_new();
    assert(MLXB_CHECK(mlx_matmul(&mm, m1, m1, s)));
    mlx_array mm_mean = mlx_array_new();
    assert(MLXB_CHECK(mlx_mean(&mm_mean, mm, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(mm_mean)));
    float fv = 0;
    assert(MLXB_CHECK(mlx_array_item_float32(&fv, mm_mean)));
    assert(fv == 2.0f);

    mlx_array four = mlx_array_new_float(4.0f);
    mlx_array sq = mlx_array_new();
    assert(MLXB_CHECK(mlx_sqrt(&sq, four, s)));
    assert(MLXB_CHECK(mlx_array_eval(sq)));
    assert(MLXB_CHECK(mlx_array_item_float32(&fv, sq)));
    assert(fv == 2.0f);

    mlx_array lt = mlx_array_new();
    assert(MLXB_CHECK(mlx_less(&lt, a, b, s)));
    assert(MLXB_CHECK(mlx_array_eval(lt)));
    bool bv = false;
    assert(MLXB_CHECK(mlx_array_item_bool(&bv, lt)));
    assert(bv == true);

    float sdata[] = {1, 2, 3};
    mlx_array sarr = mlx_array_new_data(sdata, (int[]){3}, 1, MLX_FLOAT32);
    mlx_array sax = mlx_array_new();
    assert(MLXB_CHECK(mlx_sum_axis(&sax, sarr, 0, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(sax)));
    assert(MLXB_CHECK(mlx_array_item_float32(&fv, sax)));
    assert(fv == 6.0f);

    mlx_array mn = mlx_array_new();
    assert(MLXB_CHECK(mlx_mean(&mn, sarr, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(mn)));
    assert(MLXB_CHECK(mlx_array_item_float32(&fv, mn)));
    assert(fv == 2.0f);

    mlx_array_free(mn);
    mlx_array_free(sax);
    mlx_array_free(sarr);
    mlx_array_free(lt);
    mlx_array_free(sq);
    mlx_array_free(four);
    mlx_array_free(mm_mean);
    mlx_array_free(mm);
    mlx_array_free(m1);
    mlx_array_free(sum);
    mlx_array_free(b);
    mlx_array_free(a);
    mlx_stream_free(s);
}

/* ---- Task 9: shape ops ---- */

static void test_shape_ops(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    mlx_array r = mlx_array_new();
    assert(MLXB_CHECK(mlx_arange(&r, 0, 6, 1, MLX_FLOAT32, s)));
    mlx_array mat = mlx_array_new();
    assert(MLXB_CHECK(mlx_reshape(&mat, r, (int[]){2, 3}, 2, s)));
    assert(mlx_array_ndim(mat) == 2);
    assert(mlx_array_shape(mat)[0] == 2);
    assert(mlx_array_shape(mat)[1] == 3);

    mlx_array t = mlx_array_new();
    assert(MLXB_CHECK(mlx_transpose(&t, mat, s)));
    assert(mlx_array_shape(t)[0] == 3);
    assert(mlx_array_shape(t)[1] == 2);

    mlx_array sl = mlx_array_new();
    assert(MLXB_CHECK(mlx_slice(&sl, mat, (int[]){0, 0}, 2,
                                 (int[]){1, 2}, 2, (int[]){1, 1}, 2, s)));
    assert(mlx_array_shape(sl)[0] == 1);
    assert(mlx_array_shape(sl)[1] == 2);

    mlx_vector_array cat_in = mlx_vector_array_new_value(mat);
    assert(MLXB_CHECK(mlx_vector_array_append_value(cat_in, mat)));
    mlx_array cat = mlx_array_new();
    assert(MLXB_CHECK(mlx_concatenate_axis(&cat, cat_in, 0, s)));
    assert(mlx_array_shape(cat)[0] == 4);
    assert(mlx_array_shape(cat)[1] == 3);

    mlx_array bc = mlx_array_new();
    assert(MLXB_CHECK(mlx_broadcast_to(&bc, sl, (int[]){4, 2}, 2, s)));
    assert(mlx_array_shape(bc)[0] == 4);
    assert(mlx_array_shape(bc)[1] == 2);

    mlx_array_free(bc);
    mlx_array_free(cat);
    mlx_vector_array_free(cat_in);
    mlx_array_free(sl);
    mlx_array_free(t);
    mlx_array_free(mat);
    mlx_array_free(r);
    mlx_stream_free(s);
}

/* ---- Task 10: vector_array + split + stack ---- */

static void test_vector_array(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    mlx_array a1 = mlx_array_new_int(7);
    mlx_array a2 = mlx_array_new_int(9);
    mlx_vector_array va = mlx_vector_array_new_value(a1);
    assert(MLXB_CHECK(mlx_vector_array_append_value(va, a2)));
    assert(mlx_vector_array_size(va) == 2);

    mlx_array got = mlx_array_new();
    assert(MLXB_CHECK(mlx_vector_array_get(&got, va, 1)));
    assert(MLXB_CHECK(mlx_array_eval(got)));
    int v = 0;
    assert(MLXB_CHECK(mlx_array_item_int32(&v, got)));
    assert(v == 9);

    mlx_array rng = mlx_array_new();
    assert(MLXB_CHECK(mlx_arange(&rng, 0, 4, 1, MLX_FLOAT32, s)));
    mlx_vector_array halves = mlx_vector_array_new();
    assert(MLXB_CHECK(mlx_split(&halves, rng, 2, 0, s)));
    assert(mlx_vector_array_size(halves) == 2);

    mlx_array h0 = mlx_array_new();
    assert(MLXB_CHECK(mlx_vector_array_get(&h0, halves, 0)));
    assert(mlx_array_size(h0) == 2);

    mlx_array stacked = mlx_array_new();
    assert(MLXB_CHECK(mlx_stack_axis(&stacked, halves, 0, s)));
    assert(mlx_array_ndim(stacked) == 2);
    assert(mlx_array_shape(stacked)[0] == 2);
    assert(mlx_array_shape(stacked)[1] == 2);

    mlx_array_free(stacked);
    mlx_array_free(h0);
    mlx_vector_array_free(halves);
    mlx_array_free(rng);
    mlx_array_free(got);
    mlx_vector_array_free(va);
    mlx_array_free(a2);
    mlx_array_free(a1);
    mlx_stream_free(s);
}

/* ---- Task 11: closure + compile ---- */

static int double_closure_fn(mlx_vector_array *res, const mlx_vector_array input, void *payload) {
    (void)payload;
    mlx_array x = mlx_array_new();
    if (mlx_vector_array_get(&x, input, 0) != 0) {
        mlx_array_free(x);
        return 1;
    }
    mlx_stream cpu = mlx_default_cpu_stream_new();
    mlx_array out = mlx_array_new();
    if (mlx_add(&out, x, x, cpu) != 0) {
        mlx_array_free(out);
        mlx_array_free(x);
        mlx_stream_free(cpu);
        return 1;
    }
    *res = mlx_vector_array_new_value(out);
    mlx_array_free(out);
    mlx_array_free(x);
    mlx_stream_free(cpu);
    return 0;
}

static void test_closure_compile(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    mlx_closure cls = mlx_closure_new_func_payload(double_closure_fn, NULL, NULL);
    assert(cls.ctx != NULL);

    mlx_array three = mlx_array_new_int(3);
    mlx_vector_array input = mlx_vector_array_new_value(three);
    mlx_vector_array output = mlx_vector_array_new();
    assert(MLXB_CHECK(mlx_closure_apply(&output, cls, input)));

    mlx_array r = mlx_array_new();
    assert(MLXB_CHECK(mlx_vector_array_get(&r, output, 0)));
    assert(MLXB_CHECK(mlx_array_eval(r)));
    int v = 0;
    assert(MLXB_CHECK(mlx_array_item_int32(&v, r)));
    assert(v == 6);

    mlx_closure compiled = mlx_closure_new();
    assert(MLXB_CHECK(mlx_compile(&compiled, cls, false)));

    mlx_vector_array output2 = mlx_vector_array_new();
    assert(MLXB_CHECK(mlx_closure_apply(&output2, compiled, input)));
    mlx_array r2 = mlx_array_new();
    assert(MLXB_CHECK(mlx_vector_array_get(&r2, output2, 0)));
    assert(MLXB_CHECK(mlx_array_eval(r2)));
    int v2 = 0;
    assert(MLXB_CHECK(mlx_array_item_int32(&v2, r2)));
    assert(v2 == 6);

    assert(MLXB_CHECK(mlx_detail_compile_clear_cache()));

    mlx_array_free(r2);
    mlx_vector_array_free(output2);
    mlx_closure_free(compiled);
    mlx_array_free(r);
    mlx_vector_array_free(output);
    mlx_vector_array_free(input);
    mlx_array_free(three);
    mlx_closure_free(cls);
    mlx_stream_free(s);
}

/* ---- Task 12: maps ---- */

static void test_maps(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    mlx_map_string_to_array m = mlx_map_string_to_array_new();
    mlx_array v1 = mlx_array_new_int(1);
    mlx_array v2 = mlx_array_new_int(2);
    assert(MLXB_CHECK(mlx_map_string_to_array_insert(m, "a", v1)));
    assert(MLXB_CHECK(mlx_map_string_to_array_insert(m, "b", v2)));

    mlx_array got = mlx_array_new();
    assert(MLXB_CHECK(mlx_map_string_to_array_get(&got, m, "b")));
    assert(MLXB_CHECK(mlx_array_eval(got)));
    int iv = 0;
    assert(MLXB_CHECK(mlx_array_item_int32(&iv, got)));
    assert(iv == 2);

    mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(m);
    int count = 0;
    const char *key = NULL;
    mlx_array val = mlx_array_new();
    while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0 && key != NULL) {
        count++;
        key = NULL;
    }
    assert(count == 2);

    mlx_map_string_to_array_iterator_free(it);
    mlx_array_free(val);
    mlx_array_free(got);
    mlx_array_free(v2);
    mlx_array_free(v1);
    mlx_map_string_to_array_free(m);

    mlx_map_string_to_string ms = mlx_map_string_to_string_new();
    mlx_map_string_to_string_free(ms);

    mlx_stream_free(s);
}

/* ---- Task 13: safetensors IO ---- */

static void test_safetensors_io(void) {
    mlx_stream cpu = mlx_default_cpu_stream_new();

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    mlx_array arr = mlx_array_new_data(data, (int[]){2, 2}, 2, MLX_FLOAT32);
    assert(MLXB_CHECK(mlx_array_eval(arr)));

    mlx_map_string_to_array params = mlx_map_string_to_array_new();
    assert(MLXB_CHECK(mlx_map_string_to_array_insert(params, "w", arr)));
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();

    char path[128];
    snprintf(path, sizeof(path), "/tmp/mlxbridge_test_%d.safetensors", getpid());
    assert(MLXB_CHECK(mlx_save_safetensors(path, params, meta)));

    mlx_map_string_to_array loaded_params = mlx_map_string_to_array_new();
    mlx_map_string_to_string loaded_meta = mlx_map_string_to_string_new();
    assert(MLXB_CHECK(mlx_load_safetensors(&loaded_params, &loaded_meta, path, cpu)));

    mlx_array loaded = mlx_array_new();
    assert(MLXB_CHECK(mlx_map_string_to_array_get(&loaded, loaded_params, "w")));
    assert(MLXB_CHECK(mlx_array_eval(loaded)));
    assert(mlx_array_ndim(loaded) == 2);
    assert(mlx_array_shape(loaded)[0] == 2);
    assert(mlx_array_shape(loaded)[1] == 2);
    const float *ld = mlx_array_data_float32(loaded);
    for (int i = 0; i < 4; i++) {
        assert(ld[i] == data[i]);
    }

    unlink(path);

    mlx_array_free(loaded);
    mlx_map_string_to_string_free(loaded_meta);
    mlx_map_string_to_array_free(loaded_params);
    mlx_map_string_to_string_free(meta);
    mlx_map_string_to_array_free(params);
    mlx_array_free(arr);
    mlx_stream_free(cpu);
}

/* ---- Task 14: fast ops ---- */

static void test_fast_ops(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    mlx_array ones_2x4 = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&ones_2x4, (int[]){2, 4}, 2, MLX_FLOAT32, s)));

    /* rms_norm */
    mlx_array rms = mlx_array_new();
    assert(MLXB_CHECK(mlx_fast_rms_norm(&rms, ones_2x4, mlx_array_empty, 1e-5f, s)));
    assert(MLXB_CHECK(mlx_array_eval(rms)));
    mlx_array rms_mean = mlx_array_new();
    assert(MLXB_CHECK(mlx_mean(&rms_mean, rms, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(rms_mean)));
    float fv = 0;
    assert(MLXB_CHECK(mlx_array_item_float32(&fv, rms_mean)));
    assert(fabsf(fv - 1.0f) < 1e-4f);

    /* layer_norm */
    mlx_array weight = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&weight, (int[]){4}, 1, MLX_FLOAT32, s)));
    mlx_array bias = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&bias, (int[]){4}, 1, MLX_FLOAT32, s)));
    mlx_array ln = mlx_array_new();
    assert(MLXB_CHECK(mlx_fast_layer_norm(&ln, ones_2x4, weight, bias, 1e-5f, s)));
    assert(MLXB_CHECK(mlx_array_eval(ln)));
    mlx_array ln_mean = mlx_array_new();
    assert(MLXB_CHECK(mlx_mean(&ln_mean, ln, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(ln_mean)));
    assert(MLXB_CHECK(mlx_array_item_float32(&fv, ln_mean)));
    assert(fabsf(fv - 1.0f) < 1e-4f);

    /* rope */
    mlx_array rope_in = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&rope_in, (int[]){1, 1, 2, 4}, 4, MLX_FLOAT32, s)));
    mlx_array rope_out = mlx_array_new();
    assert(MLXB_CHECK(mlx_fast_rope(&rope_out, rope_in, 4, false,
                                     (mlx_optional_float){.value = 10000.0f, .has_value = true},
                                     1.0f, 0, mlx_array_empty, s)));
    assert(MLXB_CHECK(mlx_array_eval(rope_out)));
    assert(mlx_array_ndim(rope_out) == 4);
    assert(mlx_array_shape(rope_out)[0] == 1);
    assert(mlx_array_shape(rope_out)[3] == 4);

    /* rope_dynamic with [0] offset should match static rope */
    mlx_array offsets = mlx_array_new_data((int[]){0}, (int[]){1}, 1, MLX_INT32);
    mlx_array rope_dyn = mlx_array_new();
    assert(MLXB_CHECK(mlx_fast_rope_dynamic(&rope_dyn, rope_in, 4, false,
                                             (mlx_optional_float){.value = 10000.0f, .has_value = true},
                                             1.0f, offsets, mlx_array_empty, s)));
    assert(MLXB_CHECK(mlx_array_eval(rope_dyn)));

    /* check max abs diff < 1e-5 */
    mlx_array diff = mlx_array_new();
    mlx_array absdiff = mlx_array_new();
    assert(MLXB_CHECK(mlx_subtract(&diff, rope_out, rope_dyn, s)));
    assert(MLXB_CHECK(mlx_abs(&absdiff, diff, s)));
    mlx_array maxval = mlx_array_new();
    assert(MLXB_CHECK(mlx_max(&maxval, absdiff, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(maxval)));
    assert(MLXB_CHECK(mlx_array_item_float32(&fv, maxval)));
    assert(fv < 1e-5f);

    /* scaled_dot_product_attention */
    mlx_array qkv = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&qkv, (int[]){1, 1, 2, 4}, 4, MLX_FLOAT32, s)));
    mlx_array attn = mlx_array_new();
    assert(MLXB_CHECK(mlx_fast_scaled_dot_product_attention(
        &attn, qkv, qkv, qkv, 0.5f, "", mlx_array_empty, mlx_array_empty, s)));
    assert(MLXB_CHECK(mlx_array_eval(attn)));
    assert(mlx_array_ndim(attn) == 4);
    assert(mlx_array_shape(attn)[2] == 2);
    assert(mlx_array_shape(attn)[3] == 4);

    mlx_array_free(attn);
    mlx_array_free(qkv);
    mlx_array_free(maxval);
    mlx_array_free(absdiff);
    mlx_array_free(diff);
    mlx_array_free(rope_dyn);
    mlx_array_free(offsets);
    mlx_array_free(rope_out);
    mlx_array_free(rope_in);
    mlx_array_free(ln_mean);
    mlx_array_free(ln);
    mlx_array_free(bias);
    mlx_array_free(weight);
    mlx_array_free(rms_mean);
    mlx_array_free(rms);
    mlx_array_free(ones_2x4);
    mlx_stream_free(s);
}

/* ---- Task 15: quantization + gather_qmm ---- */

static void test_quantization(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    /* 8x64 weight matrix with repeating 0..15 pattern */
    float wdata[8 * 64];
    for (int i = 0; i < 8 * 64; i++) {
        wdata[i] = (float)(i % 16);
    }
    mlx_array w = mlx_array_new_data(wdata, (int[]){8, 64}, 2, MLX_FLOAT32);

    mlx_vector_array qresult = mlx_vector_array_new();
    assert(MLXB_CHECK(mlx_quantize(&qresult, w,
                                    (mlx_optional_int){.value = 64, .has_value = true},
                                    (mlx_optional_int){.value = 4, .has_value = true},
                                    "affine", mlx_array_empty, s)));
    assert(mlx_vector_array_size(qresult) == 3);

    mlx_array q = mlx_array_new();
    mlx_array scales = mlx_array_new();
    mlx_array biases = mlx_array_new();
    assert(MLXB_CHECK(mlx_vector_array_get(&q, qresult, 0)));
    assert(MLXB_CHECK(mlx_vector_array_get(&scales, qresult, 1)));
    assert(MLXB_CHECK(mlx_vector_array_get(&biases, qresult, 2)));

    /* dequantize roundtrip */
    mlx_array dq = mlx_array_new();
    assert(MLXB_CHECK(mlx_dequantize(&dq, q, scales, biases,
                                      (mlx_optional_int){.value = 64, .has_value = true},
                                      (mlx_optional_int){.value = 4, .has_value = true},
                                      "affine", mlx_array_empty,
                                      (mlx_optional_dtype){0},
                                      s)));
    assert(MLXB_CHECK(mlx_array_eval(dq)));
    assert(MLXB_CHECK(mlx_array_eval(w)));

    mlx_array err = mlx_array_new();
    mlx_array abserr = mlx_array_new();
    mlx_array maxerr = mlx_array_new();
    assert(MLXB_CHECK(mlx_subtract(&err, w, dq, s)));
    assert(MLXB_CHECK(mlx_abs(&abserr, err, s)));
    assert(MLXB_CHECK(mlx_max(&maxerr, abserr, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(maxerr)));
    float max_err_val = 0;
    assert(MLXB_CHECK(mlx_array_item_float32(&max_err_val, maxerr)));
    assert(max_err_val < 0.01f);

    /* quantized_matmul */
    mlx_array x = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&x, (int[]){1, 64}, 2, MLX_FLOAT32, s)));
    mlx_array qmm = mlx_array_new();
    assert(MLXB_CHECK(mlx_quantized_matmul(&qmm, x, q, scales, biases, true,
                                            (mlx_optional_int){.value = 64, .has_value = true},
                                            (mlx_optional_int){.value = 4, .has_value = true},
                                            "affine", s)));
    assert(MLXB_CHECK(mlx_array_eval(qmm)));
    assert(mlx_array_ndim(qmm) == 2);
    assert(mlx_array_shape(qmm)[0] == 1);
    assert(mlx_array_shape(qmm)[1] == 8);

    mlx_array qmm_mean = mlx_array_new();
    assert(MLXB_CHECK(mlx_mean(&qmm_mean, qmm, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(qmm_mean)));
    float qmm_fv = 0;
    assert(MLXB_CHECK(mlx_array_item_float32(&qmm_fv, qmm_mean)));
    assert(fabsf(qmm_fv - 480.0f) < 1.0f);

    mlx_array_free(qmm_mean);
    mlx_array_free(qmm);
    mlx_array_free(x);
    mlx_array_free(maxerr);
    mlx_array_free(abserr);
    mlx_array_free(err);
    mlx_array_free(dq);
    mlx_array_free(biases);
    mlx_array_free(scales);
    mlx_array_free(q);
    mlx_vector_array_free(qresult);
    mlx_array_free(w);
    mlx_stream_free(s);
}

static void test_gather_qmm(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    /* Quantized gather: 2 experts, 8x64 weights each */
    float wdata[2 * 8 * 64];
    for (int i = 0; i < 2 * 8 * 64; i++) {
        wdata[i] = (float)(i % 16);
    }
    mlx_array w3d = mlx_array_new_data(wdata, (int[]){2, 8, 64}, 3, MLX_FLOAT32);
    mlx_vector_array qr = mlx_vector_array_new();
    assert(MLXB_CHECK(mlx_quantize(&qr, w3d,
                                    (mlx_optional_int){.value = 64, .has_value = true},
                                    (mlx_optional_int){.value = 4, .has_value = true},
                                    "affine", mlx_array_empty, s)));

    mlx_array q = mlx_array_new();
    mlx_array sc = mlx_array_new();
    mlx_array bi = mlx_array_new();
    assert(MLXB_CHECK(mlx_vector_array_get(&q, qr, 0)));
    assert(MLXB_CHECK(mlx_vector_array_get(&sc, qr, 1)));
    assert(MLXB_CHECK(mlx_vector_array_get(&bi, qr, 2)));

    uint32_t idx_data[] = {0, 1};
    mlx_array lhs_idx = mlx_array_new_data(idx_data, (int[]){2}, 1, MLX_UINT32);
    mlx_array rhs_idx = mlx_array_new_data(idx_data, (int[]){2}, 1, MLX_UINT32);
    mlx_array x = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&x, (int[]){2, 1, 64}, 3, MLX_FLOAT32, s)));

    mlx_array gqmm = mlx_array_new();
    assert(MLXB_CHECK(mlx_gather_qmm(&gqmm, x, q, sc, bi, lhs_idx, rhs_idx, true,
                                       (mlx_optional_int){.value = 64, .has_value = true},
                                       (mlx_optional_int){.value = 4, .has_value = true},
                                       "affine", false, s)));
    assert(MLXB_CHECK(mlx_array_eval(gqmm)));
    assert(mlx_array_ndim(gqmm) == 3);
    assert(mlx_array_shape(gqmm)[0] == 2);
    assert(mlx_array_shape(gqmm)[1] == 1);
    assert(mlx_array_shape(gqmm)[2] == 8);

    mlx_array gqmm_mean = mlx_array_new();
    assert(MLXB_CHECK(mlx_mean(&gqmm_mean, gqmm, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(gqmm_mean)));
    float fv = 0;
    assert(MLXB_CHECK(mlx_array_item_float32(&fv, gqmm_mean)));
    assert(fabsf(fv - 480.0f) < 1.0f);

    /* Dense gather_mm analog */
    mlx_array da = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&da, (int[]){2, 1, 4}, 3, MLX_FLOAT32, s)));
    mlx_array db = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&db, (int[]){2, 4, 8}, 3, MLX_FLOAT32, s)));
    mlx_array gmm = mlx_array_new();
    assert(MLXB_CHECK(mlx_gather_mm(&gmm, da, db, lhs_idx, rhs_idx, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(gmm)));
    assert(mlx_array_shape(gmm)[0] == 2);
    assert(mlx_array_shape(gmm)[1] == 1);
    assert(mlx_array_shape(gmm)[2] == 8);

    mlx_array gmm_mean = mlx_array_new();
    assert(MLXB_CHECK(mlx_mean(&gmm_mean, gmm, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(gmm_mean)));
    assert(MLXB_CHECK(mlx_array_item_float32(&fv, gmm_mean)));
    assert(fabsf(fv - 4.0f) < 1e-4f);

    mlx_array_free(gmm_mean);
    mlx_array_free(gmm);
    mlx_array_free(db);
    mlx_array_free(da);
    mlx_array_free(gqmm_mean);
    mlx_array_free(gqmm);
    mlx_array_free(x);
    mlx_array_free(rhs_idx);
    mlx_array_free(lhs_idx);
    mlx_array_free(bi);
    mlx_array_free(sc);
    mlx_array_free(q);
    mlx_vector_array_free(qr);
    mlx_array_free(w3d);
    mlx_stream_free(s);
}

/* ---- Task 16: custom Metal kernel ---- */

static void test_custom_metal_kernel(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    const char *in_names[] = {"inp"};
    const char *out_names[] = {"out"};
    mlx_vector_string in_vs = mlx_vector_string_new_data(in_names, 1);
    mlx_vector_string out_vs = mlx_vector_string_new_data(out_names, 1);

    const char *source =
        "uint elem = thread_position_in_grid.x;\n"
        "out[elem] = inp[elem];\n";

    mlx_fast_metal_kernel kern = mlx_fast_metal_kernel_new(
        "mlxbridge_test_copy", in_vs, out_vs, source, "", true, false);
    assert(kern.ctx != NULL);

    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    assert(MLXB_CHECK(mlx_fast_metal_kernel_config_add_output_arg(
        cfg, (int[]){4}, 1, MLX_FLOAT32)));
    assert(MLXB_CHECK(mlx_fast_metal_kernel_config_set_grid(cfg, 4, 1, 1)));
    assert(MLXB_CHECK(mlx_fast_metal_kernel_config_set_thread_group(cfg, 4, 1, 1)));

    float in_data[] = {10, 20, 30, 40};
    mlx_array in_arr = mlx_array_new_data(in_data, (int[]){4}, 1, MLX_FLOAT32);
    mlx_vector_array inputs = mlx_vector_array_new_value(in_arr);

    mlx_vector_array outputs = mlx_vector_array_new();
    assert(MLXB_CHECK(mlx_fast_metal_kernel_apply(&outputs, kern, inputs, cfg, s)));

    mlx_array out_arr = mlx_array_new();
    assert(MLXB_CHECK(mlx_vector_array_get(&out_arr, outputs, 0)));
    assert(MLXB_CHECK(mlx_array_eval(out_arr)));
    assert(mlx_array_size(out_arr) == 4);
    const float *out_data = mlx_array_data_float32(out_arr);
    for (int i = 0; i < 4; i++) {
        assert(out_data[i] == in_data[i]);
    }

    mlx_array_free(out_arr);
    mlx_vector_array_free(outputs);
    mlx_vector_array_free(inputs);
    mlx_array_free(in_arr);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_fast_metal_kernel_free(kern);
    mlx_vector_string_free(out_vs);
    mlx_vector_string_free(in_vs);
    mlx_stream_free(s);
}

/* ---- Task 17: conv + fft ---- */

static void test_conv_fft(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    /* conv1d: in=[1,4,1], w=[1,2,1] -> out=[1,3,1], mean=2 */
    mlx_array c1_in = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&c1_in, (int[]){1, 4, 1}, 3, MLX_FLOAT32, s)));
    mlx_array c1_w = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&c1_w, (int[]){1, 2, 1}, 3, MLX_FLOAT32, s)));
    mlx_array c1 = mlx_array_new();
    assert(MLXB_CHECK(mlx_conv1d(&c1, c1_in, c1_w, 1, 0, 1, 1, s)));
    assert(MLXB_CHECK(mlx_array_eval(c1)));
    mlx_array c1_mean = mlx_array_new();
    assert(MLXB_CHECK(mlx_mean(&c1_mean, c1, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(c1_mean)));
    float fv = 0;
    assert(MLXB_CHECK(mlx_array_item_float32(&fv, c1_mean)));
    assert(fabsf(fv - 2.0f) < 1e-4f);

    /* conv2d: in=[1,3,3,1], w=[1,2,2,1] -> out=[1,2,2,1], mean=4 */
    mlx_array c2_in = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&c2_in, (int[]){1, 3, 3, 1}, 4, MLX_FLOAT32, s)));
    mlx_array c2_w = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&c2_w, (int[]){1, 2, 2, 1}, 4, MLX_FLOAT32, s)));
    mlx_array c2 = mlx_array_new();
    assert(MLXB_CHECK(mlx_conv2d(&c2, c2_in, c2_w, 1, 1, 0, 0, 1, 1, 1, s)));
    assert(MLXB_CHECK(mlx_array_eval(c2)));
    mlx_array c2_mean = mlx_array_new();
    assert(MLXB_CHECK(mlx_mean(&c2_mean, c2, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(c2_mean)));
    assert(MLXB_CHECK(mlx_array_item_float32(&fv, c2_mean)));
    assert(fabsf(fv - 4.0f) < 1e-4f);

    /* conv3d: in=[1,3,3,3,1], w=[1,2,2,2,1] -> out=[1,2,2,2,1], mean=8 */
    mlx_array c3_in = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&c3_in, (int[]){1, 3, 3, 3, 1}, 5, MLX_FLOAT32, s)));
    mlx_array c3_w = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&c3_w, (int[]){1, 2, 2, 2, 1}, 5, MLX_FLOAT32, s)));
    mlx_array c3 = mlx_array_new();
    assert(MLXB_CHECK(mlx_conv3d(&c3, c3_in, c3_w, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, s)));
    assert(MLXB_CHECK(mlx_array_eval(c3)));
    mlx_array c3_mean = mlx_array_new();
    assert(MLXB_CHECK(mlx_mean(&c3_mean, c3, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(c3_mean)));
    assert(MLXB_CHECK(mlx_array_item_float32(&fv, c3_mean)));
    assert(fabsf(fv - 8.0f) < 1e-4f);

    /* conv_transpose1d: in=[1,2,1], w=[1,2,1] -> out=[1,3,1], mean=4/3 */
    mlx_array ct1_in = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&ct1_in, (int[]){1, 2, 1}, 3, MLX_FLOAT32, s)));
    mlx_array ct1_w = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&ct1_w, (int[]){1, 2, 1}, 3, MLX_FLOAT32, s)));
    mlx_array ct1 = mlx_array_new();
    assert(MLXB_CHECK(mlx_conv_transpose1d(&ct1, ct1_in, ct1_w, 1, 0, 1, 0, 1, s)));
    assert(MLXB_CHECK(mlx_array_eval(ct1)));
    mlx_array ct1_mean = mlx_array_new();
    assert(MLXB_CHECK(mlx_mean(&ct1_mean, ct1, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(ct1_mean)));
    assert(MLXB_CHECK(mlx_array_item_float32(&fv, ct1_mean)));
    assert(fabsf(fv - 4.0f / 3.0f) < 1e-4f);

    /* conv_transpose2d: in=[1,2,2,1], w=[1,2,2,1] -> out=[1,3,3,1], mean=16/9 */
    mlx_array ct2_in = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&ct2_in, (int[]){1, 2, 2, 1}, 4, MLX_FLOAT32, s)));
    mlx_array ct2_w = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&ct2_w, (int[]){1, 2, 2, 1}, 4, MLX_FLOAT32, s)));
    mlx_array ct2 = mlx_array_new();
    assert(MLXB_CHECK(mlx_conv_transpose2d(&ct2, ct2_in, ct2_w,
                                            1, 1, 0, 0, 1, 1, 0, 0, 1, s)));
    assert(MLXB_CHECK(mlx_array_eval(ct2)));
    mlx_array ct2_mean = mlx_array_new();
    assert(MLXB_CHECK(mlx_mean(&ct2_mean, ct2, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(ct2_mean)));
    assert(MLXB_CHECK(mlx_array_item_float32(&fv, ct2_mean)));
    assert(fabsf(fv - 16.0f / 9.0f) < 1e-4f);

    /* conv_transpose3d: in=[1,2,2,2,1], w=[1,2,2,2,1] -> out=[1,3,3,3,1], mean=64/27 */
    mlx_array ct3_in = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&ct3_in, (int[]){1, 2, 2, 2, 1}, 5, MLX_FLOAT32, s)));
    mlx_array ct3_w = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&ct3_w, (int[]){1, 2, 2, 2, 1}, 5, MLX_FLOAT32, s)));
    mlx_array ct3 = mlx_array_new();
    assert(MLXB_CHECK(mlx_conv_transpose3d(&ct3, ct3_in, ct3_w,
                                            1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, s)));
    assert(MLXB_CHECK(mlx_array_eval(ct3)));
    mlx_array ct3_mean = mlx_array_new();
    assert(MLXB_CHECK(mlx_mean(&ct3_mean, ct3, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(ct3_mean)));
    assert(MLXB_CHECK(mlx_array_item_float32(&fv, ct3_mean)));
    assert(fabsf(fv - 64.0f / 27.0f) < 1e-4f);

    /* rfft of constant signal [1,1,1,1] */
    float fft_data[] = {1, 1, 1, 1};
    mlx_array fft_in = mlx_array_new_data(fft_data, (int[]){4}, 1, MLX_FLOAT32);
    mlx_array fft_out = mlx_array_new();
    assert(MLXB_CHECK(mlx_fft_rfft(&fft_out, fft_in, 4, 0, MLX_FFT_NORM_BACKWARD, s)));
    assert(MLXB_CHECK(mlx_array_eval(fft_out)));
    assert(mlx_array_size(fft_out) == 3);
    assert(mlx_array_dtype(fft_out) == MLX_COMPLEX64);

    mlx_array_free(fft_out);
    mlx_array_free(fft_in);
    mlx_array_free(ct3_mean);
    mlx_array_free(ct3);
    mlx_array_free(ct3_w);
    mlx_array_free(ct3_in);
    mlx_array_free(ct2_mean);
    mlx_array_free(ct2);
    mlx_array_free(ct2_w);
    mlx_array_free(ct2_in);
    mlx_array_free(ct1_mean);
    mlx_array_free(ct1);
    mlx_array_free(ct1_w);
    mlx_array_free(ct1_in);
    mlx_array_free(c3_mean);
    mlx_array_free(c3);
    mlx_array_free(c3_w);
    mlx_array_free(c3_in);
    mlx_array_free(c2_mean);
    mlx_array_free(c2);
    mlx_array_free(c2_w);
    mlx_array_free(c2_in);
    mlx_array_free(c1_mean);
    mlx_array_free(c1);
    mlx_array_free(c1_w);
    mlx_array_free(c1_in);
    mlx_stream_free(s);
}

/* ---- Task 18: sampler/reduction ops ---- */

static void test_sampler_reduction_ops(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    /* cummax([1,3,2], axis=0) = [1,3,3] */
    float cdata[] = {1, 3, 2};
    mlx_array ca = mlx_array_new_data(cdata, (int[]){3}, 1, MLX_FLOAT32);
    mlx_array cm = mlx_array_new();
    assert(MLXB_CHECK(mlx_cummax(&cm, ca, 0, false, true, s)));
    assert(MLXB_CHECK(mlx_array_eval(cm)));
    const float *cmd = mlx_array_data_float32(cm);
    assert(cmd[0] == 1.0f);
    assert(cmd[1] == 3.0f);
    assert(cmd[2] == 3.0f);

    /* logsumexp */
    mlx_array lse = mlx_array_new();
    assert(MLXB_CHECK(mlx_logsumexp_axis(&lse, ca, 0, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(lse)));
    float lse_v = 0;
    assert(MLXB_CHECK(mlx_array_item_float32(&lse_v, lse)));
    assert(fabsf(lse_v - 3.40760596f) < 1e-4f);

    /* std */
    mlx_array sd = mlx_array_new();
    assert(MLXB_CHECK(mlx_std(&sd, ca, false, 0, s)));
    assert(MLXB_CHECK(mlx_array_eval(sd)));
    float sd_v = 0;
    assert(MLXB_CHECK(mlx_array_item_float32(&sd_v, sd)));
    assert(fabsf(sd_v - 0.81649658f) < 1e-4f);

    /* all: [1,3,2] -> true; [1,1,0] -> false */
    mlx_array all_true = mlx_array_new();
    assert(MLXB_CHECK(mlx_all(&all_true, ca, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(all_true)));
    bool bv = false;
    assert(MLXB_CHECK(mlx_array_item_bool(&bv, all_true)));
    assert(bv == true);

    float zdata[] = {1, 1, 0};
    mlx_array za = mlx_array_new_data(zdata, (int[]){3}, 1, MLX_FLOAT32);
    mlx_array all_false = mlx_array_new();
    assert(MLXB_CHECK(mlx_all(&all_false, za, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(all_false)));
    assert(MLXB_CHECK(mlx_array_item_bool(&bv, all_false)));
    assert(bv == false);

    /* contiguous: transpose then contiguous forces row-major layout */
    float m2data[] = {0, 1, 2, 3, 4, 5};
    mlx_array m2 = mlx_array_new_data(m2data, (int[]){2, 3}, 2, MLX_FLOAT32);
    mlx_array m2t = mlx_array_new();
    assert(MLXB_CHECK(mlx_transpose(&m2t, m2, s)));
    mlx_array m2c = mlx_array_new();
    assert(MLXB_CHECK(mlx_contiguous(&m2c, m2t, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(m2c)));
    const float *cd = mlx_array_data_float32(m2c);
    float expected[] = {0, 3, 1, 4, 2, 5};
    for (int i = 0; i < 6; i++) {
        assert(cd[i] == expected[i]);
    }

    /* std_axes on 2x3 over both axes */
    mlx_array sda = mlx_array_new();
    assert(MLXB_CHECK(mlx_std_axes(&sda, m2, (int[]){0, 1}, 2, false, 0, s)));
    assert(MLXB_CHECK(mlx_array_eval(sda)));
    float sda_v = 0;
    assert(MLXB_CHECK(mlx_array_item_float32(&sda_v, sda)));
    assert(fabsf(sda_v - 1.70782513f) < 1e-4f);

    /* mean_axes on 2x3 over both axes */
    mlx_array ma = mlx_array_new();
    assert(MLXB_CHECK(mlx_mean_axes(&ma, m2, (int[]){0, 1}, 2, false, s)));
    assert(MLXB_CHECK(mlx_array_eval(ma)));
    float ma_v = 0;
    assert(MLXB_CHECK(mlx_array_item_float32(&ma_v, ma)));
    assert(fabsf(ma_v - 2.5f) < 1e-4f);

    mlx_array_free(ma);
    mlx_array_free(sda);
    mlx_array_free(m2c);
    mlx_array_free(m2t);
    mlx_array_free(m2);
    mlx_array_free(all_false);
    mlx_array_free(za);
    mlx_array_free(all_true);
    mlx_array_free(sd);
    mlx_array_free(lse);
    mlx_array_free(cm);
    mlx_array_free(ca);
    mlx_stream_free(s);
}

/* ---- Task 19: random ---- */

static void test_random(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    mlx_array key = mlx_array_new();
    assert(MLXB_CHECK(mlx_random_key(&key, 42)));
    assert(MLXB_CHECK(mlx_array_eval(key)));
    assert(mlx_array_size(key) > 0);

    /* categorical on uniform [2,4] logits */
    mlx_array logits = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&logits, (int[]){2, 4}, 2, MLX_FLOAT32, s)));
    mlx_array cat = mlx_array_new();
    assert(MLXB_CHECK(mlx_random_categorical(&cat, logits, -1, mlx_array_empty, s)));
    assert(MLXB_CHECK(mlx_array_eval(cat)));
    assert(mlx_array_ndim(cat) == 1);
    assert(mlx_array_shape(cat)[0] == 2);

    /* normal with shape [2,3] */
    mlx_array norm = mlx_array_new();
    assert(MLXB_CHECK(mlx_random_normal(&norm, (int[]){2, 3}, 2,
                                         MLX_FLOAT32, 0.0f, 1.0f,
                                         mlx_array_empty, s)));
    assert(MLXB_CHECK(mlx_array_eval(norm)));
    assert(mlx_array_ndim(norm) == 2);
    assert(mlx_array_shape(norm)[0] == 2);
    assert(mlx_array_shape(norm)[1] == 3);

    /* randint(0, 10, shape=[3]) -> shape [3], dtype int32 */
    mlx_array lo = mlx_array_new_int(0);
    mlx_array hi = mlx_array_new_int(10);
    mlx_array ri = mlx_array_new();
    assert(MLXB_CHECK(mlx_random_randint(&ri, lo, hi, (int[]){3}, 1,
                                          MLX_INT32, mlx_array_empty, s)));
    assert(MLXB_CHECK(mlx_array_eval(ri)));
    assert(mlx_array_ndim(ri) == 1);
    assert(mlx_array_shape(ri)[0] == 3);
    assert(mlx_array_dtype(ri) == MLX_INT32);

    mlx_array_free(ri);
    mlx_array_free(hi);
    mlx_array_free(lo);
    mlx_array_free(norm);
    mlx_array_free(cat);
    mlx_array_free(logits);
    mlx_array_free(key);
    mlx_stream_free(s);
}

/* ---- Task 20: batch eval + memory management ---- */

static void test_batch_eval_memory(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    mlx_array a = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&a, (int[]){10, 10}, 2, MLX_FLOAT32, s)));
    mlx_array b = mlx_array_new();
    assert(MLXB_CHECK(mlx_ones(&b, (int[]){10, 10}, 2, MLX_FLOAT32, s)));

    mlx_vector_array va = mlx_vector_array_new();
    assert(MLXB_CHECK(mlx_vector_array_append_value(va, a)));
    assert(MLXB_CHECK(mlx_vector_array_append_value(va, b)));
    assert(MLXB_CHECK(mlx_async_eval(va)));
    assert(MLXB_CHECK(mlx_eval(va)));

    size_t active = 0, peak = 0;
    assert(MLXB_CHECK(mlx_get_active_memory(&active)));
    assert(MLXB_CHECK(mlx_get_peak_memory(&peak)));
    assert(peak >= active);

    assert(MLXB_CHECK(mlx_reset_peak_memory()));

    size_t old_cache = 0;
    assert(MLXB_CHECK(mlx_set_cache_limit(&old_cache, 1024 * 1024)));
    size_t restore_cache = 0;
    assert(MLXB_CHECK(mlx_set_cache_limit(&restore_cache, old_cache)));

    size_t one_tb = (size_t)1024 * 1024 * 1024 * 1024;
    size_t old_mem = 0;
    assert(MLXB_CHECK(mlx_set_memory_limit(&old_mem, one_tb)));
    size_t restored_mem = 0;
    assert(MLXB_CHECK(mlx_set_memory_limit(&restored_mem, old_mem)));
    assert(restored_mem == one_tb);

    size_t old_wired = 0;
    assert(MLXB_CHECK(mlx_set_wired_limit(&old_wired, 0)));
    size_t restore_wired = 0;
    assert(MLXB_CHECK(mlx_set_wired_limit(&restore_wired, old_wired)));

    assert(MLXB_CHECK(mlx_clear_cache()));

    mlx_vector_array_free(va);
    mlx_array_free(b);
    mlx_array_free(a);
    mlx_stream_free(s);
}

/* ---- Task 21: device info ---- */

static void test_device_info(void) {
    mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
    mlx_device_info info = mlx_device_info_new();
    assert(MLXB_CHECK(mlx_device_info_get(&info, gpu)));

    size_t mem_size = 0;
    assert(MLXB_CHECK(mlx_device_info_get_size(&mem_size, info, "memory_size")));
    assert(mem_size > 0);

    mlx_device_info_free(info);
    mlx_device_free(gpu);
}

/* ---- Task 22: error handler ---- */

static bool test_error_handler_fired = false;

static void record_test_error(const char *msg, void *data) {
    (void)msg;
    (void)data;
    test_error_handler_fired = true;
}

static void test_error_handler(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    mlx_set_error_handler(record_test_error, NULL, NULL);

    /* Trigger error: add arrays of incompatible shapes [2] + [3] */
    mlx_array a = mlx_array_new_data((float[]){1, 2}, (int[]){2}, 1, MLX_FLOAT32);
    mlx_array b = mlx_array_new_data((float[]){1, 2, 3}, (int[]){3}, 1, MLX_FLOAT32);
    mlx_array result = mlx_array_new();
    int ret = mlx_add(&result, a, b, s);
    if (ret == 0) {
        ret = mlx_array_eval(result);
    }
    assert(ret != 0);
    assert(test_error_handler_fired);

    /* Restore default handler */
    mlx_set_error_handler(NULL, NULL, NULL);

    mlx_array_free(result);
    mlx_array_free(b);
    mlx_array_free(a);
    mlx_stream_free(s);
}

/* ---- Cycle 12: cpu_stream + load_safetensors ---- */

static void test_cpu_stream_and_load_safetensors(void) {
    mlx_stream cpu = mlxbridge_cpu_stream();
    assert(cpu.ctx != NULL);

    mlx_device dev = mlx_device_new();
    assert(MLXB_CHECK(mlx_stream_get_device(&dev, cpu)));
    mlx_device_type dtype;
    assert(MLXB_CHECK(mlx_device_get_type(&dtype, dev)));
    assert(dtype == MLX_CPU);
    mlx_device_free(dev);

    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    mlx_array arr = mlx_array_new_data(data, (int[]){2, 3}, 2, MLX_FLOAT32);
    assert(MLXB_CHECK(mlx_array_eval(arr)));

    mlx_map_string_to_array save_params = mlx_map_string_to_array_new();
    assert(MLXB_CHECK(mlx_map_string_to_array_insert(save_params, "weight", arr)));
    mlx_map_string_to_string save_meta = mlx_map_string_to_string_new();

    char path[128];
    snprintf(path, sizeof(path), "/tmp/mlxbridge_a2_test_%d.safetensors", getpid());
    assert(MLXB_CHECK(mlx_save_safetensors(path, save_params, save_meta)));

    mlx_map_string_to_array loaded_params = mlx_map_string_to_array_new();
    mlx_map_string_to_string loaded_meta = mlx_map_string_to_string_new();
    assert(mlxbridge_load_safetensors(&loaded_params, &loaded_meta, path) == 0);

    mlx_array loaded = mlx_array_new();
    assert(MLXB_CHECK(mlx_map_string_to_array_get(&loaded, loaded_params, "weight")));
    assert(MLXB_CHECK(mlx_array_eval(loaded)));
    assert(mlx_array_ndim(loaded) == 2);
    assert(mlx_array_shape(loaded)[0] == 2);
    assert(mlx_array_shape(loaded)[1] == 3);
    const float *ld = mlx_array_data_float32(loaded);
    for (int i = 0; i < 6; i++)
        assert(ld[i] == data[i]);

    unlink(path);
    mlx_array_free(loaded);
    mlx_map_string_to_string_free(loaded_meta);
    mlx_map_string_to_array_free(loaded_params);
    mlx_map_string_to_string_free(save_meta);
    mlx_map_string_to_array_free(save_params);
    mlx_array_free(arr);
    mlx_stream_free(cpu);
}

/* ---- Cycle 13: map helpers ---- */

static void test_map_helpers(void) {
    float data[] = {10.0f, 20.0f};
    mlx_array a = mlx_array_new_data(data, (int[]){2}, 1, MLX_FLOAT32);
    mlx_array b = mlx_array_new_data((float[]){30.0f}, (int[]){1}, 1, MLX_FLOAT32);

    mlx_map_string_to_array params = mlx_map_string_to_array_new();
    assert(MLXB_CHECK(mlx_map_string_to_array_insert(params, "x", a)));
    assert(MLXB_CHECK(mlx_map_string_to_array_insert(params, "y", b)));
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();

    assert(mlxbridge_map_count(params) == 2);

    mlx_array got = mlx_array_new();
    assert(mlxbridge_map_get(&got, params, "x") == 0);
    assert(MLXB_CHECK(mlx_array_eval(got)));
    assert(mlx_array_size(got) == 2);

    mlx_array miss = mlx_array_new();
    assert(mlxbridge_map_get(&miss, params, "nonexistent") == -1);

    mlxbridge_map_free(params, meta);
    mlx_array_free(miss);
    mlx_array_free(got);
    mlx_array_free(b);
    mlx_array_free(a);
}

/* ---- Cycle 14: eval helpers ---- */

static void test_eval_helpers(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    mlx_array r = mlx_array_new();
    assert(MLXB_CHECK(mlx_arange(&r, 0, 10, 1, MLX_INT32, s)));
    mlx_array sum = mlx_array_new();
    assert(MLXB_CHECK(mlx_sum(&sum, r, false, s)));

    assert(mlxbridge_async_eval(sum) == 0);
    assert(mlxbridge_synchronize(s) == 0);

    int32_t val = 0;
    assert(mlxbridge_item_int32(&val, sum) == 0);
    assert(val == 45);

    mlx_array_free(sum);
    mlx_array_free(r);
    mlx_stream_free(s);
}

static void test_async_eval_n_pair(void) {
    mlx_stream s = mlxbridge_gpu_stream();

    mlx_array ra = mlx_array_new();
    assert(MLXB_CHECK(mlx_arange(&ra, 0, 5, 1, MLX_INT32, s)));
    mlx_array sa = mlx_array_new();
    assert(MLXB_CHECK(mlx_sum(&sa, ra, false, s)));

    mlx_array rb = mlx_array_new();
    assert(MLXB_CHECK(mlx_arange(&rb, 0, 4, 1, MLX_INT32, s)));
    mlx_array sb = mlx_array_new();
    assert(MLXB_CHECK(mlx_sum(&sb, rb, false, s)));

    mlx_array pair[2] = {sa, sb};
    assert(mlxbridge_async_eval_n(pair, 2) == 0);
    assert(mlxbridge_synchronize(s) == 0);

    int32_t va = 0, vb = 0;
    assert(mlxbridge_item_int32(&va, sa) == 0);
    assert(mlxbridge_item_int32(&vb, sb) == 0);
    assert(va == 10); /* 0+1+2+3+4 */
    assert(vb == 6);  /* 0+1+2+3 */

    mlx_array_free(sb);
    mlx_array_free(rb);
    mlx_array_free(sa);
    mlx_array_free(ra);
    mlx_stream_free(s);
}

/* ---- Cycle 15: memory helpers ---- */

static void test_memory_helpers(void) {
    size_t old_wired = 0;
    assert(mlxbridge_set_wired_limit(&old_wired, 0) == 0);
    size_t restore_wired = 0;
    assert(mlxbridge_set_wired_limit(&restore_wired, old_wired) == 0);

    size_t old_cache = 0;
    assert(mlxbridge_set_cache_limit(&old_cache, 1024 * 1024) == 0);
    size_t restore_cache = 0;
    assert(mlxbridge_set_cache_limit(&restore_cache, old_cache) == 0);

    size_t active = 0, peak = 0;
    assert(mlxbridge_get_active_memory(&active) == 0);
    assert(mlxbridge_get_peak_memory(&peak) == 0);
    assert(peak >= active);

    assert(mlxbridge_clear_cache() == 0);
}

/* ---- Cycle 8 (review): mlxbridge_item_int32 null-check (L2) ---- */

static void test_item_int32_null_out(void) {
    mlx_array a = mlx_array_new_int(42);
    int rc = mlxbridge_item_int32(NULL, a);
    assert(rc == -1);
    mlx_array_free(a);
}

/* ---- main ---- */

int main(void) {
    test_gpu_stream();
    printf("  test_gpu_stream: passed\n");

    test_get_shape();
    printf("  test_get_shape: passed\n");

    test_print_array();
    printf("  test_print_array: passed\n");

    test_version();
    printf("  test_version: passed\n");

    test_device();
    printf("  test_device: passed\n");

    test_stream();
    printf("  test_stream: passed\n");

    test_metal_available();
    printf("  test_metal_available: passed\n");

    test_array_basics();
    printf("  test_array_basics: passed\n");

    test_ops();
    printf("  test_ops: passed\n");

    test_shape_ops();
    printf("  test_shape_ops: passed\n");

    test_vector_array();
    printf("  test_vector_array: passed\n");

    test_closure_compile();
    printf("  test_closure_compile: passed\n");

    test_maps();
    printf("  test_maps: passed\n");

    test_safetensors_io();
    printf("  test_safetensors_io: passed\n");

    test_fast_ops();
    printf("  test_fast_ops: passed\n");

    test_quantization();
    printf("  test_quantization: passed\n");

    test_gather_qmm();
    printf("  test_gather_qmm: passed\n");

    test_custom_metal_kernel();
    printf("  test_custom_metal_kernel: passed\n");

    test_conv_fft();
    printf("  test_conv_fft: passed\n");

    test_sampler_reduction_ops();
    printf("  test_sampler_reduction_ops: passed\n");

    test_random();
    printf("  test_random: passed\n");

    test_batch_eval_memory();
    printf("  test_batch_eval_memory: passed\n");

    test_device_info();
    printf("  test_device_info: passed\n");

    test_error_handler();
    printf("  test_error_handler: passed\n");

    test_cpu_stream_and_load_safetensors();
    printf("  test_cpu_stream_and_load_safetensors: passed\n");

    test_map_helpers();
    printf("  test_map_helpers: passed\n");

    test_eval_helpers();
    printf("  test_eval_helpers: passed\n");

    test_async_eval_n_pair();
    printf("  test_async_eval_n_pair: passed\n");

    test_memory_helpers();
    printf("  test_memory_helpers: passed\n");

    test_item_int32_null_out();
    printf("  test_item_int32_null_out: passed\n");

    printf("test_mlxbridge_gpu: all passed\n");
    return 0;
}
