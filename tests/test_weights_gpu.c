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

    mlx_array_free(tri.biases);
    mlx_array_free(tri.scales);
    mlx_array_free(tri.weight);

    weight_triplet_t norm_tri;
    rc = weights_get_triplet(&norm_tri, &w,
                             "model.layers.0.input_layernorm");
    assert(rc == 0);
    assert(norm_tri.quantized == false);
    mlx_array_free(norm_tri.biases);
    mlx_array_free(norm_tri.scales);
    mlx_array_free(norm_tri.weight);

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

    test_real_model_optional();

    printf("test_weights_gpu: all passed\n");
    return 0;
}
