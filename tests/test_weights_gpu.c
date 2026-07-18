#include "model/weights.h"
#include "mlxbridge/mlxbridge.h"
#include "registry/registry.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

extern char *mkdtemp(char *);

#define FIXTURES MLXD_FIXTURES_DIR

/* ---- Cycle 19: load tiny_qwen3 (dense bf16) ---- */

static void test_load_dense(void) {
    model_config_t cfg = {0};
    int rc = model_config_load(&cfg, FIXTURES "/tiny_qwen3");
    assert(rc == 0);

    weights_t w;
    char err[256] = {0};
    rc = weights_load(&w, FIXTURES "/tiny_qwen3", &cfg, err, sizeof(err));
    if (rc != 0) {
        fprintf(stderr, "weights_load failed: %s\n", err);
        assert(0);
    }

    assert(weights_count(&w) == 25);
    assert(weights_total_bytes(&w) > 0);

    mlx_array embed = mlx_array_new();
    assert(weights_get(&embed, &w, "model.embed_tokens.weight") == 0);
    assert(MLXB_CHECK(mlx_array_eval(embed)));
    assert(mlx_array_ndim(embed) == 2);
    assert(mlx_array_shape(embed)[0] == 256);
    assert(mlx_array_shape(embed)[1] == 64);
    assert(mlx_array_dtype(embed) == MLX_BFLOAT16);
    mlx_array_free(embed);

    mlx_array q0 = mlx_array_new();
    assert(weights_get(&q0, &w, "model.layers.0.self_attn.q_proj.weight") == 0);
    assert(MLXB_CHECK(mlx_array_eval(q0)));
    assert(mlx_array_shape(q0)[0] == 64);
    assert(mlx_array_shape(q0)[1] == 64);
    assert(mlx_array_dtype(q0) == MLX_BFLOAT16);
    mlx_array_free(q0);

    mlx_array missing = mlx_array_new();
    assert(weights_get(&missing, &w, "nonexistent") == -1);
    mlx_array_free(missing);

    weights_free(&w);
    model_config_free(&cfg);
}

/* ---- Cycle 20: load tiny_qwen3_sharded (quantized) ---- */

static void test_load_sharded(void) {
    model_config_t cfg = {0};
    int rc = model_config_load(&cfg, FIXTURES "/tiny_qwen3_sharded");
    assert(rc == 0);
    assert(cfg.quant_bits == 4);
    assert(cfg.quant_group_size == 32);

    weights_t w;
    char err[256] = {0};
    rc = weights_load(&w, FIXTURES "/tiny_qwen3_sharded", &cfg, err, sizeof(err));
    if (rc != 0) {
        fprintf(stderr, "weights_load failed: %s\n", err);
        assert(0);
    }

    assert(weights_count(&w) > 25);

    weight_triplet_t tri;
    rc = weights_get_triplet(&tri, &w, "model.layers.0.self_attn.q_proj");
    assert(rc == 0);
    assert(tri.quantized == true);

    assert(MLXB_CHECK(mlx_array_eval(tri.weight)));
    assert(mlx_array_dtype(tri.weight) == MLX_UINT32);

    assert(MLXB_CHECK(mlx_array_eval(tri.scales)));
    mlx_dtype sdtype = mlx_array_dtype(tri.scales);
    assert(sdtype == MLX_FLOAT16 || sdtype == MLX_BFLOAT16 ||
           sdtype == MLX_FLOAT32);

    assert(MLXB_CHECK(mlx_array_eval(tri.biases)));
    mlx_dtype bdtype = mlx_array_dtype(tri.biases);
    assert(bdtype == MLX_FLOAT16 || bdtype == MLX_BFLOAT16 ||
           bdtype == MLX_FLOAT32);

    weights_triplet_free(&tri);

    weight_triplet_t norm_tri;
    rc = weights_get_triplet(&norm_tri, &w,
                             "model.layers.0.input_layernorm");
    assert(rc == 0);
    assert(norm_tri.quantized == false);
    weights_triplet_free(&norm_tri);

    weights_free(&w);
    model_config_free(&cfg);
}

/* ---- Cycle 21: validation error paths ---- */

static void test_validation_missing_shard(void) {
    char tmpdir[] = "/tmp/mlxd_weights_val_XXXXXX";
    assert(mkdtemp(tmpdir) != NULL);

    char **paths = NULL;
    size_t count = 0;
    int rc = weights_enumerate_shards(tmpdir, &paths, &count, NULL);
    assert(rc == -1);

    rmdir(tmpdir);
}

static void test_validation_deepseek_v4_rejected(void) {
    model_config_t cfg = {0};
    cfg.family = MODEL_DEEPSEEK_V4;

    weights_t w;
    char err[256] = {0};
    int rc = weights_load(&w, FIXTURES "/tiny_qwen3", &cfg, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "GGUF") != NULL);
}

/* ---- Cycle 4 (review): duplicate keys across shards (M7) ---- */

static char *make_tmpdir(const char *tag) {
    static char buf[256];
    snprintf(buf, sizeof(buf), "/tmp/mlxd_%s_XXXXXX", tag);
    assert(mkdtemp(buf) != NULL);
    return buf;
}

static void write_test_safetensors(const char *dir, const char *fname,
                                   const char *tensor_name) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, fname);

    mlx_array a = mlx_array_new_float(1.0f);
    assert(MLXB_CHECK(mlx_array_eval(a)));
    mlx_map_string_to_array m = mlx_map_string_to_array_new();
    assert(MLXB_CHECK(mlx_map_string_to_array_insert(m, tensor_name, a)));
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();
    assert(MLXB_CHECK(mlx_save_safetensors(path, m, meta)));
    mlx_map_string_to_string_free(meta);
    mlx_map_string_to_array_free(m);
    mlx_array_free(a);
}

static void test_duplicate_key_across_shards(void) {
    char *tmpdir = make_tmpdir("dupkey");

    write_test_safetensors(tmpdir, "a.safetensors", "shared_tensor");
    write_test_safetensors(tmpdir, "b.safetensors", "shared_tensor");

    model_config_t cfg = {0};
    cfg.family = MODEL_QWEN3;
    cfg.weight_prefix = "model";
    cfg.num_hidden_layers = 0;

    weights_t w;
    char err[512] = {0};
    int rc = weights_load(&w, tmpdir, &cfg, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "shared_tensor") != NULL);

    char pa[512], pb[512];
    snprintf(pa, sizeof(pa), "%s/a.safetensors", tmpdir);
    snprintf(pb, sizeof(pb), "%s/b.safetensors", tmpdir);
    unlink(pa);
    unlink(pb);
    rmdir(tmpdir);
}

/* ---- Cycle 5 (review): generic load-time validation (B1+M8) ---- */

static void write_index_json(const char *dir, const char *const *keys,
                              const char *const *shard_fnames, size_t n) {
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir);
    FILE *f = fopen(path, "w");
    assert(f);
    fprintf(f, "{\"weight_map\": {");
    for (size_t i = 0; i < n; i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "\"%s\": \"%s\"", keys[i], shard_fnames[i]);
    }
    fprintf(f, "}}\n");
    fclose(f);
}

static void test_validate_index_missing_tensor(void) {
    char *tmpdir = make_tmpdir("val_idx");

    write_test_safetensors(tmpdir, "s.safetensors", "real_tensor");

    const char *keys[] = {"real_tensor", "phantom_tensor"};
    const char *fnames[] = {"s.safetensors", "s.safetensors"};
    write_index_json(tmpdir, keys, fnames, 2);

    model_config_t cfg = {0};
    cfg.family = MODEL_QWEN3;
    cfg.weight_prefix = "model";
    cfg.num_hidden_layers = 0;

    weights_t w;
    char err[512] = {0};
    int rc = weights_load(&w, tmpdir, &cfg, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "phantom_tensor") != NULL);

    char p1[512], p2[512];
    snprintf(p1, sizeof(p1), "%s/s.safetensors", tmpdir);
    snprintf(p2, sizeof(p2), "%s/model.safetensors.index.json", tmpdir);
    unlink(p1); unlink(p2); rmdir(tmpdir);
}

static void write_config_json_layers(const char *dir, int layers, bool quant) {
    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    FILE *f = fopen(path, "w");
    assert(f);
    fprintf(f, "{\n");
    fprintf(f, "  \"model_type\": \"qwen3\",\n");
    fprintf(f, "  \"vocab_size\": 256,\n");
    fprintf(f, "  \"hidden_size\": 64,\n");
    fprintf(f, "  \"num_hidden_layers\": %d,\n", layers);
    fprintf(f, "  \"num_attention_heads\": 4,\n");
    fprintf(f, "  \"num_key_value_heads\": 2,\n");
    fprintf(f, "  \"head_dim\": 16,\n");
    fprintf(f, "  \"intermediate_size\": 128,\n");
    fprintf(f, "  \"max_position_embeddings\": 512,\n");
    fprintf(f, "  \"rms_norm_eps\": 1e-6,\n");
    fprintf(f, "  \"rope_theta\": 1000000.0,\n");
    fprintf(f, "  \"tie_word_embeddings\": false");
    if (quant) {
        fprintf(f, ",\n  \"quantization\": {\"bits\": 4, \"group_size\": 32, \"mode\": \"affine\"}");
    }
    fprintf(f, "\n}\n");
    fclose(f);
}

static void test_validate_layer_coverage(void) {
    char *tmpdir = make_tmpdir("val_lay");

    char cmd[600];
    snprintf(cmd, sizeof(cmd), "cp %s/tiny_qwen3/model.safetensors %s/",
             FIXTURES, tmpdir);
    assert(system(cmd) == 0);

    write_config_json_layers(tmpdir, 3, false);

    model_config_t cfg = {0};
    int rc = model_config_load(&cfg, tmpdir);
    assert(rc == 0);
    assert(cfg.num_hidden_layers == 3);

    weights_t w;
    char err[512] = {0};
    rc = weights_load(&w, tmpdir, &cfg, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "layer") != NULL || strstr(err, "missing") != NULL);

    char p[512];
    snprintf(p, sizeof(p), "%s/model.safetensors", tmpdir);
    unlink(p);
    snprintf(p, sizeof(p), "%s/config.json", tmpdir);
    unlink(p);
    model_config_free(&cfg);
    rmdir(tmpdir);
}

static void test_validate_quant_dtype(void) {
    char *tmpdir = make_tmpdir("val_qdt");

    char cmd[600];
    snprintf(cmd, sizeof(cmd), "cp %s/tiny_qwen3/model.safetensors %s/",
             FIXTURES, tmpdir);
    assert(system(cmd) == 0);

    write_config_json_layers(tmpdir, 2, true);

    model_config_t cfg = {0};
    int rc = model_config_load(&cfg, tmpdir);
    assert(rc == 0);
    assert(cfg.quant_bits == 4);

    weights_t w;
    char err[512] = {0};
    rc = weights_load(&w, tmpdir, &cfg, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "dtype") != NULL || strstr(err, "quant") != NULL);

    char p[512];
    snprintf(p, sizeof(p), "%s/model.safetensors", tmpdir);
    unlink(p);
    snprintf(p, sizeof(p), "%s/config.json", tmpdir);
    unlink(p);
    model_config_free(&cfg);
    rmdir(tmpdir);
}

/* ---- Cycle 6 (review): tighten triplet contract (B3) ---- */

static void test_triplet_partial_quant_rejected(void) {
    mlx_map_string_to_array m = mlx_map_string_to_array_new();

    mlx_array w = mlx_array_new_float(1.0f);
    assert(MLXB_CHECK(mlx_map_string_to_array_insert(m, "layer.weight", w)));
    mlx_array s = mlx_array_new_float(0.5f);
    assert(MLXB_CHECK(mlx_map_string_to_array_insert(m, "layer.scales", s)));

    weights_t wt = {0};
    wt.params = m;
    wt.count = 2;

    weight_triplet_t tri;
    int rc = weights_get_triplet(&tri, &wt, "layer");
    assert(rc == -1);

    mlx_array_free(s);
    mlx_array_free(w);
    mlx_map_string_to_array_free(m);
}

static void test_triplet_overflow_rejected(void) {
    mlx_map_string_to_array m = mlx_map_string_to_array_new();
    mlx_array w = mlx_array_new_float(1.0f);
    assert(MLXB_CHECK(mlx_map_string_to_array_insert(m, "x.weight", w)));

    weights_t wt = {0};
    wt.params = m;
    wt.count = 1;

    char longbase[300];
    memset(longbase, 'a', 299);
    longbase[299] = '\0';

    weight_triplet_t tri;
    int rc = weights_get_triplet(&tri, &wt, longbase);
    assert(rc == -1);

    mlx_array_free(w);
    mlx_map_string_to_array_free(m);
}

static void test_triplet_free_helper(void) {
    weights_t w;
    char err[256] = {0};
    model_config_t cfg = {0};
    int rc = model_config_load(&cfg, FIXTURES "/tiny_qwen3_sharded");
    assert(rc == 0);
    rc = weights_load(&w, FIXTURES "/tiny_qwen3_sharded", &cfg, err, sizeof(err));
    assert(rc == 0);

    weight_triplet_t tri;
    rc = weights_get_triplet(&tri, &w, "model.layers.0.self_attn.q_proj");
    assert(rc == 0);
    assert(tri.quantized == true);

    weights_triplet_free(&tri);

    weight_triplet_t norm_tri;
    rc = weights_get_triplet(&norm_tri, &w, "model.layers.0.input_layernorm");
    assert(rc == 0);
    assert(norm_tri.quantized == false);

    weights_triplet_free(&norm_tri);

    weights_free(&w);
    model_config_free(&cfg);
}

/* ---- H2 regression: error paths must zero *out ---- */

static void test_triplet_error_zeroes_out(void) {
    mlx_map_string_to_array m = mlx_map_string_to_array_new();
    mlx_array dummy = mlx_array_new_float(1.0f);
    mlx_map_string_to_array_insert(m, "unrelated.weight", dummy);
    mlx_array_free(dummy);

    weights_t wt = {0};
    wt.params = m;
    wt.count = 1;

    weight_triplet_t tri;
    memset(&tri, 0xAA, sizeof(tri));

    int rc = weights_get_triplet(&tri, &wt, "no_such_key");
    assert(rc == -1);
    assert(tri.weight.ctx == NULL);
    assert(tri.scales.ctx == NULL);
    assert(tri.biases.ctx == NULL);

    weights_triplet_free(&tri);

    mlx_map_string_to_array_free(m);
}

/* ---- Cycle 7 (review): index lists absent shard file (L1) ---- */

static void test_index_absent_shard_file(void) {
    char *tmpdir = make_tmpdir("val_abs");

    write_test_safetensors(tmpdir, "present.safetensors", "some_tensor");

    const char *keys[] = {"some_tensor", "other_tensor"};
    const char *fnames[] = {"present.safetensors", "missing.safetensors"};
    write_index_json(tmpdir, keys, fnames, 2);

    model_config_t cfg = {0};
    cfg.family = MODEL_QWEN3;
    cfg.weight_prefix = "model";
    cfg.num_hidden_layers = 0;

    weights_t w;
    char err[512] = {0};
    int rc = weights_load(&w, tmpdir, &cfg, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "missing.safetensors") != NULL);

    char p1[512], p2[512];
    snprintf(p1, sizeof(p1), "%s/present.safetensors", tmpdir);
    snprintf(p2, sizeof(p2), "%s/model.safetensors.index.json", tmpdir);
    unlink(p1); unlink(p2); rmdir(tmpdir);
}

/* ---- Helper: copy safetensors map excluding specific keys ---- */

static void save_map_excluding(const char *src_dir, const char *dst_dir,
                                const char *dst_fname,
                                const char *const *exclude_keys,
                                size_t n_exclude) {
    char src_path[512];
    snprintf(src_path, sizeof(src_path), "%s/model.safetensors", src_dir);

    mlx_map_string_to_array params = mlx_map_string_to_array_new();
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();
    assert(mlxbridge_load_safetensors(&params, &meta, src_path) == 0);

    mlx_map_string_to_array filtered = mlx_map_string_to_array_new();
    mlx_map_string_to_array_iterator it =
        mlx_map_string_to_array_iterator_new(params);
    const char *key = NULL;
    mlx_array val = mlx_array_new();
    while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0 &&
           key != NULL) {
        bool skip = false;
        for (size_t i = 0; i < n_exclude; i++) {
            if (strcmp(key, exclude_keys[i]) == 0) { skip = true; break; }
        }
        if (!skip)
            assert(MLXB_CHECK(mlx_map_string_to_array_insert(filtered, key, val)));
        key = NULL;
    }
    mlx_array_free(val);
    mlx_map_string_to_array_iterator_free(it);

    char dst_path[512];
    snprintf(dst_path, sizeof(dst_path), "%s/%s", dst_dir, dst_fname);
    mlx_map_string_to_string empty_meta = mlx_map_string_to_string_new();
    assert(MLXB_CHECK(mlx_save_safetensors(dst_path, filtered, empty_meta)));

    mlx_map_string_to_string_free(empty_meta);
    mlx_map_string_to_array_free(filtered);
    mlxbridge_map_free(params, meta);
}

/* ---- Helper: copy safetensors with additional tensors ---- */

static void save_map_with_extra(const char *src_dir, const char *dst_dir,
                                const char *dst_fname,
                                const char *const *extra_keys,
                                const mlx_array *extra_vals,
                                size_t n_extra) {
    char src_path[512];
    snprintf(src_path, sizeof(src_path), "%s/model.safetensors", src_dir);

    mlx_map_string_to_array params = mlx_map_string_to_array_new();
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();
    assert(mlxbridge_load_safetensors(&params, &meta, src_path) == 0);

    for (size_t i = 0; i < n_extra; i++)
        assert(MLXB_CHECK(mlx_map_string_to_array_insert(
            params, extra_keys[i], extra_vals[i])));

    char dst_path[512];
    snprintf(dst_path, sizeof(dst_path), "%s/%s", dst_dir, dst_fname);
    mlx_map_string_to_string empty_meta = mlx_map_string_to_string_new();
    assert(MLXB_CHECK(mlx_save_safetensors(dst_path, params, empty_meta)));

    mlx_map_string_to_string_free(empty_meta);
    mlxbridge_map_free(params, meta);
}

/* ---- Helper: copy safetensors, replacing a key's value ---- */

static void save_map_replacing(const char *src_dir, const char *dst_dir,
                                const char *dst_fname,
                                const char *replace_key,
                                mlx_array replace_val) {
    char src_path[512];
    snprintf(src_path, sizeof(src_path), "%s/model.safetensors", src_dir);

    mlx_map_string_to_array params = mlx_map_string_to_array_new();
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();
    assert(mlxbridge_load_safetensors(&params, &meta, src_path) == 0);

    mlx_map_string_to_array filtered = mlx_map_string_to_array_new();
    mlx_map_string_to_array_iterator it =
        mlx_map_string_to_array_iterator_new(params);
    const char *key = NULL;
    mlx_array val = mlx_array_new();
    while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0 &&
           key != NULL) {
        if (strcmp(key, replace_key) == 0)
            assert(MLXB_CHECK(mlx_map_string_to_array_insert(
                filtered, key, replace_val)));
        else
            assert(MLXB_CHECK(mlx_map_string_to_array_insert(
                filtered, key, val)));
        key = NULL;
    }
    mlx_array_free(val);
    mlx_map_string_to_array_iterator_free(it);

    char dst_path[512];
    snprintf(dst_path, sizeof(dst_path), "%s/%s", dst_dir, dst_fname);
    mlx_map_string_to_string empty_meta = mlx_map_string_to_string_new();
    assert(MLXB_CHECK(mlx_save_safetensors(dst_path, filtered, empty_meta)));

    mlx_map_string_to_string_free(empty_meta);
    mlx_map_string_to_array_free(filtered);
    mlxbridge_map_free(params, meta);
}

/* ---- Cycle 3 (plan): strict expected-tensor validation ---- */

static void test_validate_missing_expected_tensor(void) {
    char *tmpdir = make_tmpdir("val_exp");

    const char *exclude[] = {"model.layers.0.self_attn.q_proj.weight"};
    save_map_excluding(FIXTURES "/tiny_qwen3", tmpdir,
                       "model.safetensors", exclude, 1);

    /* Copy config.json from fixture */
    char cmd[600];
    snprintf(cmd, sizeof(cmd), "cp %s/tiny_qwen3/config.json %s/",
             FIXTURES, tmpdir);
    assert(system(cmd) == 0);

    model_config_t cfg = {0};
    assert(model_config_load(&cfg, tmpdir) == 0);

    weights_t w;
    char err[512] = {0};
    int rc = weights_load(&w, tmpdir, &cfg, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "q_proj") != NULL);

    char p[512];
    snprintf(p, sizeof(p), "%s/model.safetensors", tmpdir);
    unlink(p);
    snprintf(p, sizeof(p), "%s/config.json", tmpdir);
    unlink(p);
    model_config_free(&cfg);
    rmdir(tmpdir);
}

/* ---- Cycle 4 (plan): orphan scales detection ---- */

static void test_validate_orphan_scales(void) {
    char *tmpdir = make_tmpdir("val_orph");

    /* Start with dense tiny_qwen3, add orphan .scales (no .biases, no uint32 base) */
    mlx_array orphan = mlx_array_new_float(0.5f);
    assert(MLXB_CHECK(mlx_array_eval(orphan)));
    const char *extra_keys[] = {"model.layers.0.self_attn.q_proj.scales"};
    const mlx_array extra_vals[] = {orphan};
    save_map_with_extra(FIXTURES "/tiny_qwen3", tmpdir,
                        "model.safetensors", extra_keys, extra_vals, 1);
    mlx_array_free(orphan);

    char cmd[600];
    snprintf(cmd, sizeof(cmd), "cp %s/tiny_qwen3/config.json %s/",
             FIXTURES, tmpdir);
    assert(system(cmd) == 0);

    model_config_t cfg = {0};
    assert(model_config_load(&cfg, tmpdir) == 0);

    weights_t w;
    char err[512] = {0};
    int rc = weights_load(&w, tmpdir, &cfg, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "q_proj") != NULL);

    char p[512];
    snprintf(p, sizeof(p), "%s/model.safetensors", tmpdir);
    unlink(p);
    snprintf(p, sizeof(p), "%s/config.json", tmpdir);
    unlink(p);
    model_config_free(&cfg);
    rmdir(tmpdir);
}

/* ---- Cycle 5 (plan): norm dtype validation ---- */

static void test_validate_norm_dtype(void) {
    char *tmpdir = make_tmpdir("val_ndt");

    /* Replace input_layernorm.weight with uint32 array */
    uint32_t ones[64];
    for (int i = 0; i < 64; i++) ones[i] = 1;
    int shape[] = {64};
    mlx_array uint32_arr = mlx_array_new_data(ones, shape, 1, MLX_UINT32);
    assert(MLXB_CHECK(mlx_array_eval(uint32_arr)));

    save_map_replacing(FIXTURES "/tiny_qwen3", tmpdir,
                       "model.safetensors",
                       "model.layers.0.input_layernorm.weight",
                       uint32_arr);
    mlx_array_free(uint32_arr);

    char cmd[600];
    snprintf(cmd, sizeof(cmd), "cp %s/tiny_qwen3/config.json %s/",
             FIXTURES, tmpdir);
    assert(system(cmd) == 0);

    model_config_t cfg = {0};
    assert(model_config_load(&cfg, tmpdir) == 0);

    weights_t w;
    char err[512] = {0};
    int rc = weights_load(&w, tmpdir, &cfg, err, sizeof(err));
    assert(rc == -1);
    assert(strstr(err, "input_layernorm") != NULL);

    char p[512];
    snprintf(p, sizeof(p), "%s/model.safetensors", tmpdir);
    unlink(p);
    snprintf(p, sizeof(p), "%s/config.json", tmpdir);
    unlink(p);
    model_config_free(&cfg);
    rmdir(tmpdir);
}

/* ---- Cycle 22: optional real model test ---- */

static void test_real_model_optional(void) {
    char *dir = registry_resolve("mlx-community/Qwen3-0.6B-4bit");
    if (!dir) {
        printf("  test_real_model: skipped (model not cached)\n");
        return;
    }

    model_config_t cfg = {0};
    int rc = model_config_load(&cfg, dir);
    if (rc != 0) {
        printf("  test_real_model: skipped (config load failed)\n");
        free(dir);
        return;
    }

    weights_t w;
    char err[512] = {0};
    rc = weights_load(&w, dir, &cfg, err, sizeof(err));
    if (rc != 0) {
        printf("  test_real_model: LOAD FAILED: %s\n", err);
        model_config_free(&cfg);
        free(dir);
        assert(0);
    }

    printf("  test_real_model: %zu tensors, %zu bytes\n",
           weights_count(&w), weights_total_bytes(&w));

    weights_free(&w);
    model_config_free(&cfg);
    free(dir);
}

/* ---- main ---- */

int main(void) {
    test_load_dense();
    printf("  test_load_dense: passed\n");

    test_load_sharded();
    printf("  test_load_sharded: passed\n");

    test_validation_missing_shard();
    printf("  test_validation_missing_shard: passed\n");

    test_validation_deepseek_v4_rejected();
    printf("  test_validation_deepseek_v4_rejected: passed\n");

    test_duplicate_key_across_shards();
    printf("  test_duplicate_key_across_shards: passed\n");

    test_validate_index_missing_tensor();
    printf("  test_validate_index_missing_tensor: passed\n");

    test_validate_layer_coverage();
    printf("  test_validate_layer_coverage: passed\n");

    test_validate_quant_dtype();
    printf("  test_validate_quant_dtype: passed\n");

    test_triplet_partial_quant_rejected();
    printf("  test_triplet_partial_quant_rejected: passed\n");

    test_triplet_overflow_rejected();
    printf("  test_triplet_overflow_rejected: passed\n");

    test_triplet_free_helper();
    printf("  test_triplet_free_helper: passed\n");

    test_triplet_error_zeroes_out();
    printf("  test_triplet_error_zeroes_out: passed\n");

    test_index_absent_shard_file();
    printf("  test_index_absent_shard_file: passed\n");

    test_validate_missing_expected_tensor();
    printf("  test_validate_missing_expected_tensor: passed\n");

    test_validate_orphan_scales();
    printf("  test_validate_orphan_scales: passed\n");

    test_validate_norm_dtype();
    printf("  test_validate_norm_dtype: passed\n");

    test_real_model_optional();

    printf("test_weights_gpu: all passed\n");
    return 0;
}
