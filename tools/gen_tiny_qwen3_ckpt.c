/*
 * Generates deterministic tiny Qwen3 fixtures for weight-loader tests.
 *
 * Emits:
 *   tests/fixtures/tiny_qwen3/          - dense bf16, single file, 25 tensors
 *   tests/fixtures/tiny_qwen3_sharded/  - 4-bit affine quantized, 2 shards + index.json
 *
 * Build: see Makefile target `tools/gen_tiny_qwen3_ckpt`
 * Run:   ./tools/gen_tiny_qwen3_ckpt   (from repo root, or MLXD_FIXTURES_DIR set)
 */

#include <mlx/c/mlx.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include <yyjson/yyjson.h>

#define CHECK(call) do { \
    int _rc = (call); \
    if (_rc != 0) { \
        fprintf(stderr, "FAIL: %s (rc=%d) at %s:%d\n", #call, _rc, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

/* Dims */
enum {
    VOCAB      = 256,
    HIDDEN     = 64,
    HEADS      = 4,
    KV_HEADS   = 2,
    HEAD_DIM   = 16,
    INTER      = 128,
    LAYERS     = 2,
    GROUP_SIZE = 32,
    QUANT_BITS = 4,
};

static mlx_array rand_bf16(mlx_array *key, const int *shape, int ndim, mlx_stream s) {
    mlx_array k1 = mlx_array_new();
    mlx_array k2 = mlx_array_new();
    CHECK(mlx_random_split(&k1, &k2, *key, s));
    CHECK(mlx_array_set(key, k2));
    mlx_array_free(k2);

    mlx_array f32 = mlx_array_new();
    CHECK(mlx_random_normal(&f32, shape, ndim, MLX_FLOAT32, 0.0f, 0.02f, k1, s));
    mlx_array_free(k1);

    mlx_array bf = mlx_array_new();
    CHECK(mlx_astype(&bf, f32, MLX_BFLOAT16, s));
    mlx_array_free(f32);
    return bf;
}

static mlx_array ones_bf16(const int *shape, int ndim, mlx_stream s) {
    mlx_array f32 = mlx_array_new();
    CHECK(mlx_ones(&f32, shape, ndim, MLX_FLOAT32, s));
    mlx_array bf = mlx_array_new();
    CHECK(mlx_astype(&bf, f32, MLX_BFLOAT16, s));
    mlx_array_free(f32);
    return bf;
}

static void insert(mlx_map_string_to_array m, const char *name, mlx_array a) {
    CHECK(mlx_map_string_to_array_insert(m, name, a));
    mlx_array_free(a);
}

static void insert_layer(mlx_map_string_to_array m, mlx_array *key,
                          int layer, const char *suffix,
                          const int *shape, int ndim, mlx_stream s) {
    char name[256];
    snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);
    insert(m, name, rand_bf16(key, shape, ndim, s));
}

static void insert_layer_norm(mlx_map_string_to_array m,
                               int layer, const char *suffix,
                               int size, mlx_stream s) {
    char name[256];
    snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);
    insert(m, name, ones_bf16((int[]){size}, 1, s));
}

static int cmp_str_ptr(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

static void append_newline(const char *path) {
    FILE *f = fopen(path, "ab");
    if (f) { fputc('\n', f); fclose(f); }
}

static void write_config_json_ex(const char *dir, bool quantized, bool tied) {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_obj_add_str(doc, root, "model_type", "qwen3");
    yyjson_mut_obj_add_int(doc, root, "vocab_size", VOCAB);
    yyjson_mut_obj_add_int(doc, root, "hidden_size", HIDDEN);
    yyjson_mut_obj_add_int(doc, root, "num_hidden_layers", LAYERS);
    yyjson_mut_obj_add_int(doc, root, "num_attention_heads", HEADS);
    yyjson_mut_obj_add_int(doc, root, "num_key_value_heads", KV_HEADS);
    yyjson_mut_obj_add_int(doc, root, "head_dim", HEAD_DIM);
    yyjson_mut_obj_add_int(doc, root, "intermediate_size", INTER);
    yyjson_mut_obj_add_int(doc, root, "max_position_embeddings", 512);
    yyjson_mut_obj_add_real(doc, root, "rms_norm_eps", 1e-6);
    yyjson_mut_obj_add_real(doc, root, "rope_theta", 1000000.0);
    yyjson_mut_obj_add_bool(doc, root, "tie_word_embeddings", tied);

    if (quantized) {
        yyjson_mut_val *qcfg = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_int(doc, qcfg, "bits", QUANT_BITS);
        yyjson_mut_obj_add_int(doc, qcfg, "group_size", GROUP_SIZE);
        yyjson_mut_obj_add_str(doc, qcfg, "mode", "affine");
        yyjson_mut_obj_add_val(doc, root, "quantization", qcfg);
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    yyjson_write_err werr;
    if (!yyjson_mut_write_file(path, doc, YYJSON_WRITE_PRETTY, NULL, &werr)) {
        fprintf(stderr, "failed to write %s: %s\n", path, werr.msg);
        exit(1);
    }
    append_newline(path);
    yyjson_mut_doc_free(doc);
}

static void write_config_json(const char *dir, bool quantized) {
    write_config_json_ex(dir, quantized, false);
}

static void quantize_and_replace(mlx_map_string_to_array m,
                                  const char *name, mlx_stream s) {
    mlx_array orig = mlx_array_new();
    CHECK(mlx_map_string_to_array_get(&orig, m, name));

    mlx_array f32 = mlx_array_new();
    CHECK(mlx_astype(&f32, orig, MLX_FLOAT32, s));

    mlx_vector_array qr = mlx_vector_array_new();
    CHECK(mlx_quantize(&qr, f32,
                       (mlx_optional_int){.value = GROUP_SIZE, .has_value = true},
                       (mlx_optional_int){.value = QUANT_BITS, .has_value = true},
                       "affine", mlx_array_empty, s));

    mlx_array qw = mlx_array_new();
    mlx_array qs = mlx_array_new();
    mlx_array qb = mlx_array_new();
    CHECK(mlx_vector_array_get(&qw, qr, 0));
    CHECK(mlx_vector_array_get(&qs, qr, 1));
    CHECK(mlx_vector_array_get(&qb, qr, 2));

    CHECK(mlx_map_string_to_array_insert(m, name, qw));

    const char *dot = strrchr(name, '.');
    assert(dot);
    int base_len = (int)(dot - name);

    char scales_name[512];
    snprintf(scales_name, sizeof(scales_name), "%.*s.scales", base_len, name);
    CHECK(mlx_map_string_to_array_insert(m, scales_name, qs));

    char biases_name[512];
    snprintf(biases_name, sizeof(biases_name), "%.*s.biases", base_len, name);
    CHECK(mlx_map_string_to_array_insert(m, biases_name, qb));

    mlx_array_free(qb);
    mlx_array_free(qs);
    mlx_array_free(qw);
    mlx_vector_array_free(qr);
    mlx_array_free(f32);
    mlx_array_free(orig);
}

static const char *matmul_suffixes[] = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    NULL,
};

static mlx_map_string_to_array build_tensors(mlx_stream s) {
    mlx_array key = mlx_array_new();
    CHECK(mlx_random_key(&key, 42));

    mlx_map_string_to_array m = mlx_map_string_to_array_new();

    insert(m, "model.embed_tokens.weight",
           rand_bf16(&key, (int[]){VOCAB, HIDDEN}, 2, s));
    insert(m, "model.norm.weight",
           ones_bf16((int[]){HIDDEN}, 1, s));
    insert(m, "lm_head.weight",
           rand_bf16(&key, (int[]){VOCAB, HIDDEN}, 2, s));

    for (int L = 0; L < LAYERS; L++) {
        insert_layer(m, &key, L, "self_attn.q_proj.weight",
                     (int[]){HEADS * HEAD_DIM, HIDDEN}, 2, s);
        insert_layer(m, &key, L, "self_attn.k_proj.weight",
                     (int[]){KV_HEADS * HEAD_DIM, HIDDEN}, 2, s);
        insert_layer(m, &key, L, "self_attn.v_proj.weight",
                     (int[]){KV_HEADS * HEAD_DIM, HIDDEN}, 2, s);
        insert_layer(m, &key, L, "self_attn.o_proj.weight",
                     (int[]){HIDDEN, HEADS * HEAD_DIM}, 2, s);

        insert_layer_norm(m, L, "self_attn.q_norm.weight", HEAD_DIM, s);
        insert_layer_norm(m, L, "self_attn.k_norm.weight", HEAD_DIM, s);

        insert_layer(m, &key, L, "mlp.gate_proj.weight",
                     (int[]){INTER, HIDDEN}, 2, s);
        insert_layer(m, &key, L, "mlp.up_proj.weight",
                     (int[]){INTER, HIDDEN}, 2, s);
        insert_layer(m, &key, L, "mlp.down_proj.weight",
                     (int[]){HIDDEN, INTER}, 2, s);

        insert_layer_norm(m, L, "input_layernorm.weight", HIDDEN, s);
        insert_layer_norm(m, L, "post_attention_layernorm.weight", HIDDEN, s);
    }

    mlx_array_free(key);
    return m;
}

static void save_dense(const char *dir, mlx_map_string_to_array m) {
    mkdir(dir, 0755);
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();
    CHECK(mlx_save_safetensors(path, m, meta));
    mlx_map_string_to_string_free(meta);
    write_config_json(dir, false);
    printf("  wrote %s\n", dir);
}

static void save_sharded(const char *dir, mlx_map_string_to_array m,
                          mlx_stream s) {
    mkdir(dir, 0755);

    quantize_and_replace(m, "model.embed_tokens.weight", s);
    for (int L = 0; L < LAYERS; L++) {
        for (const char **sp = matmul_suffixes; *sp; sp++) {
            char name[256];
            snprintf(name, sizeof(name), "model.layers.%d.%s", L, *sp);
            quantize_and_replace(m, name, s);
        }
    }
    quantize_and_replace(m, "lm_head.weight", s);

    mlx_map_string_to_array shard1 = mlx_map_string_to_array_new();
    mlx_map_string_to_array shard2 = mlx_map_string_to_array_new();

    yyjson_mut_doc *idx_doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *idx_root = yyjson_mut_obj(idx_doc);
    yyjson_mut_doc_set_root(idx_doc, idx_root);
    yyjson_mut_val *wm = yyjson_mut_obj(idx_doc);
    yyjson_mut_obj_add_val(idx_doc, idx_root, "weight_map", wm);

    /* Collect all keys, sort for deterministic shard assignment */
    mlx_map_string_to_array_iterator it =
        mlx_map_string_to_array_iterator_new(m);
    const char *key = NULL;
    mlx_array val = mlx_array_new();
    size_t key_cap = 64, key_count = 0;
    char **keys = malloc(key_cap * sizeof(char *));
    while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0 &&
           key != NULL) {
        if (key_count >= key_cap) {
            key_cap *= 2;
            keys = realloc(keys, key_cap * sizeof(char *));
        }
        keys[key_count++] = strdup(key);
        key = NULL;
    }
    mlx_array_free(val);
    mlx_map_string_to_array_iterator_free(it);

    qsort(keys, key_count, sizeof(char *), cmp_str_ptr);

    for (size_t ki = 0; ki < key_count; ki++) {
        bool first_shard = (ki % 2 == 0);
        const char *shard_name = first_shard
            ? "model-00001-of-00002.safetensors"
            : "model-00002-of-00002.safetensors";

        mlx_array tensor_val = mlx_array_new();
        CHECK(mlx_map_string_to_array_get(&tensor_val, m, keys[ki]));

        mlx_map_string_to_array target = first_shard ? shard1 : shard2;
        CHECK(mlx_map_string_to_array_insert(target, keys[ki], tensor_val));
        yyjson_mut_obj_add_str(idx_doc, wm, keys[ki], shard_name);

        mlx_array_free(tensor_val);
    }

    char path[512];
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();

    snprintf(path, sizeof(path), "%s/model-00001-of-00002.safetensors", dir);
    CHECK(mlx_save_safetensors(path, shard1, meta));

    snprintf(path, sizeof(path), "%s/model-00002-of-00002.safetensors", dir);
    CHECK(mlx_save_safetensors(path, shard2, meta));

    mlx_map_string_to_string_free(meta);
    mlx_map_string_to_array_free(shard2);
    mlx_map_string_to_array_free(shard1);

    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir);
    yyjson_write_err werr;
    if (!yyjson_mut_write_file(path, idx_doc, YYJSON_WRITE_PRETTY, NULL, &werr)) {
        fprintf(stderr, "failed to write %s: %s\n", path, werr.msg);
        exit(1);
    }
    append_newline(path);
    yyjson_mut_doc_free(idx_doc);

    for (size_t ki = 0; ki < key_count; ki++) free(keys[ki]);
    free(keys);

    write_config_json(dir, true);
    printf("  wrote %s\n", dir);
}

static void save_tied(const char *dir, mlx_map_string_to_array src,
                      mlx_stream s) {
    mkdir(dir, 0755);

    mlx_map_string_to_array m = mlx_map_string_to_array_new();

    mlx_map_string_to_array_iterator it =
        mlx_map_string_to_array_iterator_new(src);
    const char *key = NULL;
    mlx_array val = mlx_array_new();
    while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0 &&
           key != NULL) {
        if (strncmp(key, "lm_head", 7) == 0) { key = NULL; continue; }
        CHECK(mlx_map_string_to_array_insert(m, key, val));
        key = NULL;
    }
    mlx_array_free(val);
    mlx_map_string_to_array_iterator_free(it);

    quantize_and_replace(m, "model.embed_tokens.weight", s);
    for (int L = 0; L < LAYERS; L++) {
        for (const char **sp = matmul_suffixes; *sp; sp++) {
            char name[256];
            snprintf(name, sizeof(name), "model.layers.%d.%s", L, *sp);
            quantize_and_replace(m, name, s);
        }
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();
    CHECK(mlx_save_safetensors(path, m, meta));
    mlx_map_string_to_string_free(meta);
    mlx_map_string_to_array_free(m);

    write_config_json_ex(dir, true, true);
    printf("  wrote %s\n", dir);
}

int main(void) {
    const char *fixtures_dir = getenv("MLXD_FIXTURES_DIR");
    if (!fixtures_dir) fixtures_dir = MLXD_FIXTURES_DIR;

    mlx_stream cpu = mlx_default_cpu_stream_new();
    mlx_map_string_to_array tensors = build_tensors(cpu);

    char dense_dir[512], sharded_dir[512], tied_dir[512];
    snprintf(dense_dir, sizeof(dense_dir), "%s/tiny_qwen3", fixtures_dir);
    snprintf(sharded_dir, sizeof(sharded_dir), "%s/tiny_qwen3_sharded", fixtures_dir);
    snprintf(tied_dir, sizeof(tied_dir), "%s/tiny_qwen3_tied", fixtures_dir);

    printf("gen_tiny_qwen3_ckpt:\n");
    save_dense(dense_dir, tensors);

    save_sharded(sharded_dir, tensors, cpu);
    mlx_map_string_to_array_free(tensors);

    mlx_map_string_to_array tensors2 = build_tensors(cpu);
    save_tied(tied_dir, tensors2, cpu);
    mlx_map_string_to_array_free(tensors2);

    mlx_stream_free(cpu);

    printf("  done.\n");
    return 0;
}
