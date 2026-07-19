/*
 * Generates deterministic tiny model fixtures for weight-loader and forward tests.
 *
 * Recipe-driven: each recipe emits a family-specific fixture directory under
 * tests/fixtures/.  Adding a new family = adding a recipe + tensor tables.
 *
 * Build: see Makefile target `tools/gen_tiny_ckpt`
 * Run:   ./tools/gen_tiny_ckpt   (from repo root, or MLXD_FIXTURES_DIR set)
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

/* --- Recipe types -------------------------------------------------------- */

typedef enum { T_RAND, T_ONES } tensor_kind_t;

typedef struct {
    const char *suffix;
    tensor_kind_t kind;
} tensor_spec_t;

typedef struct {
    const char *model_type;
    int vocab, hidden, heads, kv_heads, head_dim, inter, layers;
    int group_size, quant_bits;
    int max_position_embeddings;
    double rms_norm_eps, rope_theta;

    int sliding_window;
    int sliding_window_pattern;
    int query_pre_attn_scalar;
    double rope_local_base_freq;
    bool has_lm_head;
    bool tie_word_embeddings;

    const char *rope_scaling_type;
    double rope_scaling_factor;
    double rope_low_freq_factor;
    double rope_high_freq_factor;
    int rope_original_max_position_embeddings;

    const tensor_spec_t *top;   int n_top;
    const tensor_spec_t *layer; int n_layer;

    bool emit_dense, emit_sharded, emit_tied;
    const char *fixture_base;
} recipe_t;

/* --- Shape resolution ---------------------------------------------------- */

static void resolve_shape(const recipe_t *r, const char *suffix,
                          int *shape, int *ndim) {
    if (strcmp(suffix, "embed_tokens.weight") == 0) {
        shape[0] = r->vocab; shape[1] = r->hidden; *ndim = 2;
    } else if (strcmp(suffix, "norm.weight") == 0) {
        shape[0] = r->hidden; *ndim = 1;
    } else if (strcmp(suffix, "lm_head.weight") == 0) {
        shape[0] = r->vocab; shape[1] = r->hidden; *ndim = 2;
    } else if (strcmp(suffix, "self_attn.q_proj.weight") == 0) {
        shape[0] = r->heads * r->head_dim; shape[1] = r->hidden; *ndim = 2;
    } else if (strcmp(suffix, "self_attn.k_proj.weight") == 0) {
        shape[0] = r->kv_heads * r->head_dim; shape[1] = r->hidden; *ndim = 2;
    } else if (strcmp(suffix, "self_attn.v_proj.weight") == 0) {
        shape[0] = r->kv_heads * r->head_dim; shape[1] = r->hidden; *ndim = 2;
    } else if (strcmp(suffix, "self_attn.o_proj.weight") == 0) {
        shape[0] = r->hidden; shape[1] = r->heads * r->head_dim; *ndim = 2;
    } else if (strcmp(suffix, "self_attn.q_norm.weight") == 0 ||
               strcmp(suffix, "self_attn.k_norm.weight") == 0) {
        shape[0] = r->head_dim; *ndim = 1;
    } else if (strcmp(suffix, "mlp.gate_proj.weight") == 0 ||
               strcmp(suffix, "mlp.up_proj.weight") == 0) {
        shape[0] = r->inter; shape[1] = r->hidden; *ndim = 2;
    } else if (strcmp(suffix, "mlp.down_proj.weight") == 0) {
        shape[0] = r->hidden; shape[1] = r->inter; *ndim = 2;
    } else if (strcmp(suffix, "input_layernorm.weight") == 0 ||
               strcmp(suffix, "post_attention_layernorm.weight") == 0 ||
               strcmp(suffix, "pre_feedforward_layernorm.weight") == 0 ||
               strcmp(suffix, "post_feedforward_layernorm.weight") == 0) {
        shape[0] = r->hidden; *ndim = 1;
    } else {
        fprintf(stderr, "FATAL: unknown suffix '%s'\n", suffix);
        exit(1);
    }
}

/* --- MLX helpers --------------------------------------------------------- */

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

static int cmp_str_ptr(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

static void append_newline(const char *path) {
    FILE *f = fopen(path, "ab");
    if (f) { fputc('\n', f); fclose(f); }
}

#define LEN(a) ((int)(sizeof(a) / sizeof((a)[0])))

/* --- Tensor tables: qwen3 ------------------------------------------------ */

static const tensor_spec_t QWEN3_TOP[] = {
    {"embed_tokens.weight", T_RAND},
    {"norm.weight",         T_ONES},
    {"lm_head.weight",      T_RAND},
};

static const tensor_spec_t QWEN3_LAYER[] = {
    {"self_attn.q_proj.weight",            T_RAND},
    {"self_attn.k_proj.weight",            T_RAND},
    {"self_attn.v_proj.weight",            T_RAND},
    {"self_attn.o_proj.weight",            T_RAND},
    {"self_attn.q_norm.weight",            T_ONES},
    {"self_attn.k_norm.weight",            T_ONES},
    {"mlp.gate_proj.weight",               T_RAND},
    {"mlp.up_proj.weight",                 T_RAND},
    {"mlp.down_proj.weight",               T_RAND},
    {"input_layernorm.weight",             T_ONES},
    {"post_attention_layernorm.weight",    T_ONES},
};

/* --- Tensor tables: gemma3 ----------------------------------------------- */

static const tensor_spec_t GEMMA3_TOP[] = {
    {"embed_tokens.weight", T_RAND},
    {"norm.weight",         T_ONES},
};

static const tensor_spec_t GEMMA3_LAYER[] = {
    {"self_attn.q_proj.weight",            T_RAND},
    {"self_attn.k_proj.weight",            T_RAND},
    {"self_attn.v_proj.weight",            T_RAND},
    {"self_attn.o_proj.weight",            T_RAND},
    {"self_attn.q_norm.weight",            T_ONES},
    {"self_attn.k_norm.weight",            T_ONES},
    {"mlp.gate_proj.weight",               T_RAND},
    {"mlp.up_proj.weight",                 T_RAND},
    {"mlp.down_proj.weight",               T_RAND},
    {"input_layernorm.weight",             T_ONES},
    {"post_attention_layernorm.weight",    T_ONES},
    {"pre_feedforward_layernorm.weight",   T_ONES},
    {"post_feedforward_layernorm.weight",  T_ONES},
};

/* --- Tensor tables: llama3 ----------------------------------------------- */

static const tensor_spec_t LLAMA3_TOP[] = {
    {"embed_tokens.weight", T_RAND},
    {"norm.weight",         T_ONES},
    {"lm_head.weight",      T_RAND},
};

static const tensor_spec_t LLAMA3_LAYER[] = {
    {"self_attn.q_proj.weight",            T_RAND},
    {"self_attn.k_proj.weight",            T_RAND},
    {"self_attn.v_proj.weight",            T_RAND},
    {"self_attn.o_proj.weight",            T_RAND},
    {"mlp.gate_proj.weight",               T_RAND},
    {"mlp.up_proj.weight",                 T_RAND},
    {"mlp.down_proj.weight",               T_RAND},
    {"input_layernorm.weight",             T_ONES},
    {"post_attention_layernorm.weight",    T_ONES},
};

/* --- Recipes ------------------------------------------------------------- */

static const recipe_t QWEN3 = {
    .model_type = "qwen3",
    .vocab = 256, .hidden = 64, .heads = 4, .kv_heads = 2,
    .head_dim = 16, .inter = 128, .layers = 2,
    .group_size = 32, .quant_bits = 4,
    .max_position_embeddings = 512,
    .rms_norm_eps = 1e-6, .rope_theta = 1000000.0,
    .sliding_window = 0, .sliding_window_pattern = 0,
    .query_pre_attn_scalar = 0, .rope_local_base_freq = 0,
    .has_lm_head = true, .tie_word_embeddings = false,
    .top = QWEN3_TOP, .n_top = LEN(QWEN3_TOP),
    .layer = QWEN3_LAYER, .n_layer = LEN(QWEN3_LAYER),
    .emit_dense = true, .emit_sharded = true, .emit_tied = true,
    .fixture_base = "tiny_qwen3",
};

static const recipe_t GEMMA3 = {
    .model_type = "gemma3_text",
    .vocab = 256, .hidden = 64, .heads = 4, .kv_heads = 2,
    .head_dim = 16, .inter = 128, .layers = 2,
    .group_size = 32, .quant_bits = 4,
    .max_position_embeddings = 512,
    .rms_norm_eps = 1e-6, .rope_theta = 1000000.0,
    .sliding_window = 4, .sliding_window_pattern = 2,
    .query_pre_attn_scalar = 64, .rope_local_base_freq = 10000.0,
    .has_lm_head = false, .tie_word_embeddings = true,
    .top = GEMMA3_TOP, .n_top = LEN(GEMMA3_TOP),
    .layer = GEMMA3_LAYER, .n_layer = LEN(GEMMA3_LAYER),
    /* quantized (sharded/tied-quant) gemma3 deferred to D1 gate; dense fixture is already tied */
    .emit_dense = true, .emit_sharded = false, .emit_tied = false,
    .fixture_base = "tiny_gemma3",
};

static const recipe_t LLAMA3 = {
    .model_type = "llama",
    .vocab = 256, .hidden = 64, .heads = 4, .kv_heads = 2,
    .head_dim = 16, .inter = 128, .layers = 2,
    .group_size = 32, .quant_bits = 4,
    .max_position_embeddings = 512,
    .rms_norm_eps = 1e-6, .rope_theta = 500000.0,
    .sliding_window = 0, .sliding_window_pattern = 0,
    .query_pre_attn_scalar = 0, .rope_local_base_freq = 0,
    .has_lm_head = true, .tie_word_embeddings = false,
    .rope_scaling_type = "llama3",
    .rope_scaling_factor = 8.0,
    .rope_low_freq_factor = 1.0,
    .rope_high_freq_factor = 4.0,
    .rope_original_max_position_embeddings = 8192,
    .top = LLAMA3_TOP, .n_top = LEN(LLAMA3_TOP),
    .layer = LLAMA3_LAYER, .n_layer = LEN(LLAMA3_LAYER),
    .emit_dense = true, .emit_sharded = false, .emit_tied = true,
    .fixture_base = "tiny_llama3",
};

static const recipe_t *RECIPES[] = { &QWEN3, &GEMMA3, &LLAMA3 };
static const int N_RECIPES = 3;

/* --- Build tensors (table-driven) ---------------------------------------- */

static mlx_map_string_to_array build_tensors(const recipe_t *r, mlx_stream s) {
    mlx_array key = mlx_array_new();
    CHECK(mlx_random_key(&key, 42));

    mlx_map_string_to_array m = mlx_map_string_to_array_new();

    for (int i = 0; i < r->n_top; i++) {
        int shape[4], ndim;
        resolve_shape(r, r->top[i].suffix, shape, &ndim);

        char name[256];
        if (strcmp(r->top[i].suffix, "lm_head.weight") == 0)
            snprintf(name, sizeof(name), "lm_head.weight");
        else
            snprintf(name, sizeof(name), "model.%s", r->top[i].suffix);

        if (r->top[i].kind == T_RAND)
            insert(m, name, rand_bf16(&key, shape, ndim, s));
        else
            insert(m, name, ones_bf16(shape, ndim, s));
    }

    for (int L = 0; L < r->layers; L++) {
        for (int i = 0; i < r->n_layer; i++) {
            int shape[4], ndim;
            resolve_shape(r, r->layer[i].suffix, shape, &ndim);

            char name[256];
            snprintf(name, sizeof(name), "model.layers.%d.%s", L, r->layer[i].suffix);

            if (r->layer[i].kind == T_RAND)
                insert(m, name, rand_bf16(&key, shape, ndim, s));
            else
                insert(m, name, ones_bf16(shape, ndim, s));
        }
    }

    mlx_array_free(key);
    return m;
}

/* --- Config writer ------------------------------------------------------- */

static void write_config_json(const recipe_t *r, const char *dir,
                              bool quantized, bool tied) {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_obj_add_str(doc, root, "model_type", r->model_type);
    yyjson_mut_obj_add_int(doc, root, "vocab_size", r->vocab);
    yyjson_mut_obj_add_int(doc, root, "hidden_size", r->hidden);
    yyjson_mut_obj_add_int(doc, root, "num_hidden_layers", r->layers);
    yyjson_mut_obj_add_int(doc, root, "num_attention_heads", r->heads);
    yyjson_mut_obj_add_int(doc, root, "num_key_value_heads", r->kv_heads);
    yyjson_mut_obj_add_int(doc, root, "head_dim", r->head_dim);
    yyjson_mut_obj_add_int(doc, root, "intermediate_size", r->inter);
    yyjson_mut_obj_add_int(doc, root, "max_position_embeddings",
                           r->max_position_embeddings);
    yyjson_mut_obj_add_real(doc, root, "rms_norm_eps", r->rms_norm_eps);
    yyjson_mut_obj_add_real(doc, root, "rope_theta", r->rope_theta);
    yyjson_mut_obj_add_bool(doc, root, "tie_word_embeddings", tied);

    if (r->sliding_window > 0) {
        yyjson_mut_obj_add_int(doc, root, "sliding_window", r->sliding_window);
        yyjson_mut_obj_add_int(doc, root, "sliding_window_pattern",
                               r->sliding_window_pattern);
    }
    if (r->query_pre_attn_scalar > 0)
        yyjson_mut_obj_add_int(doc, root, "query_pre_attn_scalar",
                               r->query_pre_attn_scalar);
    if (r->rope_local_base_freq > 0)
        yyjson_mut_obj_add_real(doc, root, "rope_local_base_freq",
                                r->rope_local_base_freq);

    if (r->rope_scaling_type) {
        yyjson_mut_val *rs = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_str(doc, rs, "rope_type", r->rope_scaling_type);
        yyjson_mut_obj_add_real(doc, rs, "factor", r->rope_scaling_factor);
        yyjson_mut_obj_add_real(doc, rs, "low_freq_factor",
                                r->rope_low_freq_factor);
        yyjson_mut_obj_add_real(doc, rs, "high_freq_factor",
                                r->rope_high_freq_factor);
        yyjson_mut_obj_add_int(doc, rs, "original_max_position_embeddings",
                               r->rope_original_max_position_embeddings);
        yyjson_mut_obj_add_val(doc, root, "rope_scaling", rs);
    }

    if (quantized) {
        yyjson_mut_val *qcfg = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_int(doc, qcfg, "bits", r->quant_bits);
        yyjson_mut_obj_add_int(doc, qcfg, "group_size", r->group_size);
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

/* --- Quantize helper ----------------------------------------------------- */

static void quantize_and_replace(mlx_map_string_to_array m,
                                  const char *name, int group_size,
                                  int quant_bits, mlx_stream s) {
    mlx_array orig = mlx_array_new();
    CHECK(mlx_map_string_to_array_get(&orig, m, name));

    mlx_array f32 = mlx_array_new();
    CHECK(mlx_astype(&f32, orig, MLX_FLOAT32, s));

    mlx_vector_array qr = mlx_vector_array_new();
    CHECK(mlx_quantize(&qr, f32,
                       (mlx_optional_int){.value = group_size, .has_value = true},
                       (mlx_optional_int){.value = quant_bits, .has_value = true},
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

/* Suffixes that get quantized in sharded/tied variants */
static const char *const matmul_suffixes[] = {
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    NULL,
};

/* --- Save: dense --------------------------------------------------------- */

static void save_dense(const recipe_t *r, const char *dir,
                       mlx_map_string_to_array m) {
    mkdir(dir, 0755);
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();
    CHECK(mlx_save_safetensors(path, m, meta));
    mlx_map_string_to_string_free(meta);
    write_config_json(r, dir, false, r->tie_word_embeddings);
    printf("  wrote %s\n", dir);
}

/* --- Save: sharded ------------------------------------------------------- */

static void save_sharded(const recipe_t *r, const char *dir,
                         mlx_map_string_to_array m, mlx_stream s) {
    mkdir(dir, 0755);

    quantize_and_replace(m, "model.embed_tokens.weight",
                         r->group_size, r->quant_bits, s);
    for (int L = 0; L < r->layers; L++) {
        for (const char *const *sp = matmul_suffixes; *sp; sp++) {
            char name[256];
            snprintf(name, sizeof(name), "model.layers.%d.%s", L, *sp);
            quantize_and_replace(m, name, r->group_size, r->quant_bits, s);
        }
    }
    if (r->has_lm_head)
        quantize_and_replace(m, "lm_head.weight",
                             r->group_size, r->quant_bits, s);

    mlx_map_string_to_array shard1 = mlx_map_string_to_array_new();
    mlx_map_string_to_array shard2 = mlx_map_string_to_array_new();

    yyjson_mut_doc *idx_doc = yyjson_mut_doc_new(NULL);
    yyjson_mut_val *idx_root = yyjson_mut_obj(idx_doc);
    yyjson_mut_doc_set_root(idx_doc, idx_root);
    yyjson_mut_val *wm = yyjson_mut_obj(idx_doc);
    yyjson_mut_obj_add_val(idx_doc, idx_root, "weight_map", wm);

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

    write_config_json(r, dir, true, r->tie_word_embeddings);
    printf("  wrote %s\n", dir);
}

/* --- Save: tied ---------------------------------------------------------- */

static void save_tied(const recipe_t *r, const char *dir,
                      mlx_map_string_to_array src, mlx_stream s) {
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

    quantize_and_replace(m, "model.embed_tokens.weight",
                         r->group_size, r->quant_bits, s);
    for (int L = 0; L < r->layers; L++) {
        for (const char *const *sp = matmul_suffixes; *sp; sp++) {
            char name[256];
            snprintf(name, sizeof(name), "model.layers.%d.%s", L, *sp);
            quantize_and_replace(m, name, r->group_size, r->quant_bits, s);
        }
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    mlx_map_string_to_string meta = mlx_map_string_to_string_new();
    CHECK(mlx_save_safetensors(path, m, meta));
    mlx_map_string_to_string_free(meta);
    mlx_map_string_to_array_free(m);

    write_config_json(r, dir, true, true);
    printf("  wrote %s\n", dir);
}

/* --- Main ---------------------------------------------------------------- */

int main(void) {
    const char *fixtures_dir = getenv("MLXD_FIXTURES_DIR");
    if (!fixtures_dir) fixtures_dir = MLXD_FIXTURES_DIR;

    mlx_stream cpu = mlx_default_cpu_stream_new();

    printf("gen_tiny_ckpt:\n");

    for (int ri = 0; ri < N_RECIPES; ri++) {
        const recipe_t *r = RECIPES[ri];

        char dense_dir[512], sharded_dir[512], tied_dir[512];
        snprintf(dense_dir, sizeof(dense_dir), "%s/%s", fixtures_dir, r->fixture_base);
        snprintf(sharded_dir, sizeof(sharded_dir), "%s/%s_sharded",
                 fixtures_dir, r->fixture_base);
        snprintf(tied_dir, sizeof(tied_dir), "%s/%s_tied",
                 fixtures_dir, r->fixture_base);

        if (r->emit_dense) {
            mlx_map_string_to_array tensors = build_tensors(r, cpu);
            save_dense(r, dense_dir, tensors);

            if (r->emit_sharded) {
                save_sharded(r, sharded_dir, tensors, cpu);
            }
            mlx_map_string_to_array_free(tensors);
        }

        if (r->emit_tied) {
            mlx_map_string_to_array tensors2 = build_tensors(r, cpu);
            save_tied(r, tied_dir, tensors2, cpu);
            mlx_map_string_to_array_free(tensors2);
        }
    }

    mlx_stream_free(cpu);

    printf("  done.\n");
    return 0;
}
