#include "model/model.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MLXD_FIXTURES_DIR
#error "MLXD_FIXTURES_DIR must be defined"
#endif

/* --- Cycle I1: happy path ------------------------------------------------- */

static void test_happy_path(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config");
    assert(rc == 0);

    assert(cfg.model_type != NULL);
    assert(strcmp(cfg.model_type, "llama") == 0);

    assert(cfg.architectures != NULL);
    assert(strcmp(cfg.architectures, "LlamaForCausalLM") == 0);

    assert(cfg.vocab_size == 32000);
    assert(cfg.hidden_size == 4096);
    assert(cfg.num_hidden_layers == 32);
    assert(cfg.num_attention_heads == 32);
    assert(cfg.max_position_embeddings == 2048);

    model_config_free(&cfg);
}

/* --- Cycle I2: kv-heads defaults to num_attention_heads ------------------ */

static void test_kv_heads_default(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config");
    assert(rc == 0);
    assert(cfg.num_key_value_heads == 32);
    model_config_free(&cfg);
}

/* --- Cycle I3: missing config / NULL args -------------------------------- */

static void test_missing_config(void) {
    model_config_t cfg;
    assert(model_config_load(&cfg, MLXD_FIXTURES_DIR "/no_such_dir") == -1);
    assert(model_config_load(NULL, MLXD_FIXTURES_DIR "/model_config") == -1);

    /* NULL model_dir must zero cfg so free-after-failure is safe */
    memset(&cfg, 0xAB, sizeof(cfg));
    assert(model_config_load(&cfg, NULL) == -1);
    assert(cfg.model_type == NULL);
    assert(cfg.architectures == NULL);
    model_config_free(&cfg);
}

/* --- Cycle I4: missing model_type ---------------------------------------- */

static void test_missing_model_type(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_no_type");
    assert(rc == -1);
    /* cfg must be safe to free after a failed load */
    model_config_free(&cfg);
}

/* --- Cycle I6: corrupt JSON ---------------------------------------------- */

static void test_bad_json(void) {
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_bad_json");
    assert(rc == -1);
    model_config_free(&cfg);
}

/* --- Cycle I7: root is not an object ------------------------------------- */

static void test_root_not_object(void) {
    model_config_t cfg;
    int rc =
        model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config_root_array");
    assert(rc == -1);
    model_config_free(&cfg);
}

/* --- Cycle I8: model_type present but not a string ----------------------- */

static void test_model_type_not_string(void) {
    model_config_t cfg;
    int rc = model_config_load(
        &cfg, MLXD_FIXTURES_DIR "/model_config_type_not_string");
    assert(rc == -1);
    model_config_free(&cfg);
}

/* --- Cycle I5: free semantics -------------------------------------------- */

static void test_free_semantics(void) {
    /* double free after successful load */
    model_config_t cfg;
    int rc = model_config_load(&cfg, MLXD_FIXTURES_DIR "/model_config");
    assert(rc == 0);
    model_config_free(&cfg);
    model_config_free(&cfg);

    /* NULL argument is a no-op */
    model_config_free(NULL);

    /* free after failed load is safe (poison to catch missing zeroing) */
    model_config_t cfg2;
    memset(&cfg2, 0xAB, sizeof(cfg2));
    assert(model_config_load(&cfg2, MLXD_FIXTURES_DIR "/no_such_dir") == -1);
    model_config_free(&cfg2);
}

int main(void) {
    test_happy_path();
    test_kv_heads_default();
    test_missing_config();
    test_missing_model_type();
    test_bad_json();
    test_root_not_object();
    test_model_type_not_string();
    test_free_semantics();
    printf("test_model_config: all passed\n");
    return 0;
}
