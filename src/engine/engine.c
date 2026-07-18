#include "engine/engine.h"
#include "engine/engine_internal.h"
#include "engine/emodel.h"
#include "engine/kvcache.h"
#include "engine/sampler.h"
#include "mlxbridge/mlxbridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ---- Stream -------------------------------------------------------------- */

stream_t *stream_create(int capacity) {
    if (capacity <= 0) return NULL;
    stream_t *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->buf = calloc((size_t)capacity, sizeof(chunk_t));
    if (!s->buf) { free(s); return NULL; }
    s->cap = capacity;
    s->len = 0;
    s->read_pos = 0;
    pthread_mutex_init(&s->mtx, NULL);
    pthread_cond_init(&s->cond, NULL);
    atomic_store(&s->cancelled, false);
    atomic_store(&s->refcount, 1);
    s->on_push = NULL;
    s->on_push_ctx = NULL;
    return s;
}

void stream_retain(stream_t *s) {
    atomic_fetch_add(&s->refcount, 1);
}

void stream_release(stream_t *s) {
    if (!s) return;
    if (atomic_fetch_sub(&s->refcount, 1) == 1) {
        for (int i = 0; i < s->len; i++) {
            int idx = (s->read_pos + i) % s->cap;
            if (s->buf[idx].tag == CHUNK_ERROR)
                free(s->buf[idx].error);
        }
        pthread_mutex_destroy(&s->mtx);
        pthread_cond_destroy(&s->cond);
        free(s->buf);
        free(s);
    }
}

bool stream_push(stream_t *s, chunk_t chunk) {
    pthread_mutex_lock(&s->mtx);
    while (s->len == s->cap && !atomic_load(&s->cancelled)) {
        pthread_cond_wait(&s->cond, &s->mtx);
    }
    if (atomic_load(&s->cancelled)) {
        pthread_mutex_unlock(&s->mtx);
        if (chunk.tag == CHUNK_ERROR)
            free(chunk.error);
        return false;
    }
    int write_pos = (s->read_pos + s->len) % s->cap;
    s->buf[write_pos] = chunk;
    s->len++;
    void (*cb)(void *) = s->on_push;
    void *ctx = s->on_push_ctx;
    pthread_cond_broadcast(&s->cond);
    pthread_mutex_unlock(&s->mtx);
    if (cb) cb(ctx);
    return true;
}

bool stream_next(stream_t *s, chunk_t *out, int timeout_ms) {
    pthread_mutex_lock(&s->mtx);

    if (timeout_ms < 0) {
        while (s->len == 0 && !atomic_load(&s->cancelled)) {
            pthread_cond_wait(&s->cond, &s->mtx);
        }
    } else if (timeout_ms > 0) {
        struct timespec deadline;
        clock_gettime(CLOCK_REALTIME, &deadline);
        long total_ns = deadline.tv_nsec + (long)timeout_ms * 1000000L;
        deadline.tv_sec += total_ns / 1000000000L;
        deadline.tv_nsec = total_ns % 1000000000L;
        while (s->len == 0 && !atomic_load(&s->cancelled)) {
            int rc = pthread_cond_timedwait(&s->cond, &s->mtx, &deadline);
            if (rc != 0) break;
        }
    }

    if (s->len == 0) {
        pthread_mutex_unlock(&s->mtx);
        return false;
    }
    *out = s->buf[s->read_pos];
    s->read_pos = (s->read_pos + 1) % s->cap;
    s->len--;
    pthread_cond_broadcast(&s->cond);
    pthread_mutex_unlock(&s->mtx);
    return true;
}

void stream_cancel(stream_t *s) {
    pthread_mutex_lock(&s->mtx);
    atomic_store(&s->cancelled, true);
    pthread_cond_broadcast(&s->cond);
    pthread_mutex_unlock(&s->mtx);
}

void stream_set_notify(stream_t *s, void (*cb)(void *), void *ctx) {
    pthread_mutex_lock(&s->mtx);
    s->on_push = cb;
    s->on_push_ctx = ctx;
    pthread_mutex_unlock(&s->mtx);
}

bool stream_sole_owner(const stream_t *s) {
    return atomic_load(&s->refcount) == 1;
}

/* ---- Stream helpers (static) --------------------------------------------- */

static bool stream_push_error_kind(stream_t *s, const char *msg,
                                   gen_err_kind_t kind) {
    /* strdup ownership transfers to stream_push (stored in ring buffer,
     * or freed on cancel). Analyzer cannot track through computed index. */
    char *dup = strdup(msg);
    if (!dup) return false;
    __attribute__((suppress))
    return stream_push(s, (chunk_t){
        .tag = CHUNK_ERROR,
        .error_kind = kind,
        .error = dup,
    });
}

static bool stream_push_error(stream_t *s, const char *msg) {
    return stream_push_error_kind(s, msg, GEN_ERR_INTERNAL);
}

static void stream_finish_cancelled(stream_t *s) {
    pthread_mutex_lock(&s->mtx);
    if (s->len < s->cap) {
        int write_pos = (s->read_pos + s->len) % s->cap;
        s->buf[write_pos] = (chunk_t){.tag = CHUNK_DONE, .done = FINISH_CANCELLED};
        s->len++;
    } else {
        int last = (s->read_pos + s->len - 1) % s->cap;
        if (s->buf[last].tag == CHUNK_ERROR)
            free(s->buf[last].error);
        s->buf[last] = (chunk_t){.tag = CHUNK_DONE, .done = FINISH_CANCELLED};
    }
    atomic_store(&s->cancelled, true);
    void (*cb)(void *) = s->on_push;
    void *ctx = s->on_push_ctx;
    pthread_cond_broadcast(&s->cond);
    pthread_mutex_unlock(&s->mtx);
    if (cb) cb(ctx);
}

/* Push ERROR(+kind) + DONE/STOP, or FINISH_CANCELLED if push fails under shutdown. */
static void generate_fail(engine_t *eng, stream_t *s,
                          const char *msg, gen_err_kind_t kind) {
    bool ok = stream_push_error_kind(s, msg, kind);
    if (ok)
        ok = stream_push(s, (chunk_t){.tag = CHUNK_DONE, .done = FINISH_STOP});
    if (!ok && atomic_load(&eng->shutdown))
        stream_finish_cancelled(s);
}

/* ---- Load-state helpers -------------------------------------------------- */

load_state_t engine_load_state(const engine_t *eng) {
    return (load_state_t)atomic_load(&eng->load_state);
}

int engine_load_error(engine_t *eng, char *buf, size_t n) {
    if (!buf || n == 0) return -1;
    pthread_mutex_lock(&eng->load_mtx);
    size_t len = strlen(eng->load_error);
    if (len + 1 > n) {
        pthread_mutex_unlock(&eng->load_mtx);
        return -1;
    }
    memcpy(buf, eng->load_error, len + 1);
    pthread_mutex_unlock(&eng->load_mtx);
    return 0;
}

static void set_load_state(engine_t *eng, load_state_t st) {
    atomic_store(&eng->load_state, (int)st);
}

static void clear_load_error(engine_t *eng) {
    pthread_mutex_lock(&eng->load_mtx);
    eng->load_error[0] = '\0';
    pthread_mutex_unlock(&eng->load_mtx);
}

static void set_load_error(engine_t *eng, const char *msg) {
    pthread_mutex_lock(&eng->load_mtx);
    if (msg)
        snprintf(eng->load_error, sizeof(eng->load_error), "%s", msg);
    else
        eng->load_error[0] = '\0';
    pthread_mutex_unlock(&eng->load_mtx);
}

/* Free a resident real model (no-op for stub / empty). Keeps the shell. */
static void free_resident_model(engine_t *eng) {
    if (!eng->model) return;
    if (eng->model->stub) {
        memset(eng->model, 0, sizeof(*eng->model));
        return;
    }
    engine_model_free(eng->model);
}

int engine_wait_load_until(engine_t *eng, int timeout_ms,
                           bool (*cancel)(void *), void *ud) {
    /* timeout_ms < 0 => wait forever (until OK / FAILED / cancel). */
    for (int i = 0; timeout_ms < 0 || i < timeout_ms; i++) {
        if (cancel && cancel(ud))
            return -1;
        load_state_t st = engine_load_state(eng);
        if (st == LOAD_OK)
            return 0;
        if (st == LOAD_FAILED)
            return -1;
        usleep(1000);
    }
    load_state_t st = engine_load_state(eng);
    if (st == LOAD_OK)
        return 0;
    return -1;
}

int engine_wait_load(engine_t *eng, int timeout_ms) {
    return engine_wait_load_until(eng, timeout_ms, NULL, NULL);
}

/* ---- Command helpers ----------------------------------------------------- */

static void cmd_cancel(engine_cmd_t *cmd) {
    switch (cmd->tag) {
    case CMD_GENERATE:
        stream_finish_cancelled(cmd->generate.stream);
        stream_release(cmd->generate.stream);
        free(cmd->generate.token_ids);
        break;
    case CMD_RECLAIM:
        stream_release(cmd->reclaim.stream);
        break;
    case CMD_LOAD:
        free(cmd->load.model_path);
        break;
    default:
        break;
    }
    free(cmd);
}

/* ---- Mailbox ------------------------------------------------------------- */

static engine_cmd_t *mailbox_dequeue(engine_t *eng) {
    pthread_mutex_lock(&eng->mailbox_mtx);
    while (!eng->mailbox_head && !atomic_load(&eng->shutdown)) {
        pthread_cond_wait(&eng->mailbox_cond, &eng->mailbox_mtx);
    }
    engine_cmd_t *cmd = eng->mailbox_head;
    if (cmd) {
        eng->mailbox_head = cmd->next;
        if (!eng->mailbox_head) eng->mailbox_tail = NULL;
        cmd->next = NULL;
    }
    pthread_mutex_unlock(&eng->mailbox_mtx);
    return cmd;
}

/* ---- Mailbox drain ------------------------------------------------------- */

static void mailbox_drain_cancel(engine_t *eng) {
    pthread_mutex_lock(&eng->mailbox_mtx);
    engine_cmd_t *head = eng->mailbox_head;
    eng->mailbox_head = NULL;
    eng->mailbox_tail = NULL;
    pthread_mutex_unlock(&eng->mailbox_mtx);

    while (head) {
        engine_cmd_t *next = head->next;
        cmd_cancel(head);
        head = next;
    }
}

/* ---- Generate helpers ---------------------------------------------------- */

static bool token_is_eos(const model_config_t *cfg, int32_t id) {
    for (int i = 0; i < cfg->num_eos_tokens; i++) {
        if ((int32_t)cfg->eos_token_ids[i] == id)
            return true;
    }
    return false;
}

/* Lazy form: returns argmax array without materializing. Caller owns *tok_out. */
static int greedy_next_token_lazy(mlx_array logits, mlx_stream s,
                                  mlx_array *tok_out) {
    mlx_array tok = mlx_array_new();
    if (!MLXB_CHECK(mlx_argmax_axis(&tok, logits, -1, false, s))) {
        mlx_array_free(tok);
        return -1;
    }
    *tok_out = tok;
    return 0;
}

/* Greedy argmax over last-axis of logits. logits: [1, vocab] (or [1,1,vocab]).
 * Exposed (non-static) so GPU tests can call it directly. */
int greedy_next_id(mlx_array logits, mlx_stream s, int32_t *id_out) {
    mlx_array tok = mlx_array_new();
    if (greedy_next_token_lazy(logits, s, &tok) != 0) {
        mlx_array_free(tok);
        return -1;
    }
    if (mlxbridge_item_int32(id_out, tok) != 0) {
        mlx_array_free(tok);
        return -1;
    }
    mlx_array_free(tok);
    return 0;
}

static mlx_array make_ids_array(const int32_t *ids, int n) {
    int shape[] = {1, n};
    return mlx_array_new_data(ids, shape, 2, MLX_INT32);
}

/* Reshape a lazy token (shape [1] or scalar-ish) to [1, 1] for embed/forward. */
static int reshape_token_ids(mlx_array lazy_tok, mlx_array *ids_out, mlx_stream s) {
    int shape[] = {1, 1};
    mlx_array reshaped = mlx_array_new();
    if (!MLXB_CHECK(mlx_reshape(&reshaped, lazy_tok, shape, 2, s))) {
        mlx_array_free(reshaped);
        return -1;
    }
    *ids_out = reshaped;
    return 0;
}

static void handle_generate_stub(engine_t *eng, engine_cmd_t *cmd) {
    stream_t *s = cmd->generate.stream;
    int count = cmd->generate.token_count;
    int max = cmd->generate.params.max_tokens;
    int limit = count < max ? count : max;
    bool truncated = limit < count;
    bool aborted = false;

    for (int i = 0; i < limit; i++) {
        if (atomic_load(&eng->shutdown)) {
            stream_finish_cancelled(s);
            aborted = true;
            break;
        }
        chunk_t tok = {
            .tag = CHUNK_TOKEN,
            .token = {.id = cmd->generate.token_ids[i], .logprob = 0},
        };
        if (!stream_push(s, tok)) {
            stream_finish_cancelled(s);
            aborted = true;
            break;
        }
    }

    if (!aborted) {
        finish_reason_t reason = truncated ? FINISH_LENGTH : FINISH_STOP;
        if (!stream_push(s, (chunk_t){.tag = CHUNK_DONE, .done = reason}))
            stream_finish_cancelled(s);
    }
}

/* Real prefill + pipelined greedy decode (mlx-lm cadence). */
static void handle_generate_real(engine_t *eng, engine_cmd_t *cmd) {
    stream_t *s = cmd->generate.stream;
    engine_model_t *em = eng->model;
    int n = cmd->generate.token_count;
    int max_new = cmd->generate.params.max_tokens;
    const int32_t *token_ids = cmd->generate.token_ids;
    const sampling_params_t *sp = &cmd->generate.params.sampling;
    bool want_lp = cmd->generate.params.logprobs;

    kvcache_t kv;
    memset(&kv, 0, sizeof(kv));
    bool kv_inited = false;
    mlx_array pending_logits = mlx_array_new();
    sampler_key_t skey;
    sampler_key_init(&skey, sp->seed);

    if (n > em->cfg.max_position_embeddings) {
        char msg[160];
        snprintf(msg, sizeof(msg),
                 "prompt token count %d exceeds max_position_embeddings %d",
                 n, em->cfg.max_position_embeddings);
        generate_fail(eng, s, msg, GEN_ERR_CONTEXT_LENGTH);
        goto out_free;
    }

    if (n == 0 || max_new <= 0) {
        if (!stream_push(s, (chunk_t){.tag = CHUNK_DONE, .done = FINISH_STOP}))
            stream_finish_cancelled(s);
        goto out_free;
    }

    if (kvcache_init(&kv, em->cfg.num_hidden_layers,
                     em->cfg.num_key_value_heads, em->cfg.head_dim) != 0) {
        generate_fail(eng, s, "kvcache init failed", GEN_ERR_INTERNAL);
        goto out_free;
    }
    kv_inited = true;

    /* ---- chunked prefill of prompt[:-1] ---- */
    int pos = 0;
    int prefix_end = n > 0 ? n - 1 : 0;
    while (pos < prefix_end) {
        if (atomic_load(&eng->shutdown) || atomic_load(&s->cancelled)) {
            stream_finish_cancelled(s);
            goto out_free;
        }
        int chunk_end = pos + MLXD_PREFILL_CHUNK;
        if (chunk_end > prefix_end)
            chunk_end = prefix_end;
        int chunk_len = chunk_end - pos;
        mlx_array ids = make_ids_array(token_ids + pos, chunk_len);
        if (model_forward(em, ids, &kv, false, NULL) != 0) {
            mlx_array_free(ids);
            generate_fail(eng, s, "prefill forward failed", GEN_ERR_INTERNAL);
            goto out_free;
        }
        mlx_array_free(ids);
        mlxbridge_clear_cache();
        pos = chunk_end;
    }

    /* ---- seed logits from last prompt token ---- */
    {
        mlx_array last = make_ids_array(&token_ids[n - 1], 1);
        if (model_forward(em, last, &kv, true, &pending_logits) != 0) {
            mlx_array_free(last);
            generate_fail(eng, s, "seed forward failed", GEN_ERR_INTERNAL);
            goto out_free;
        }
        mlx_array_free(last);
    }

    /* ---- pipelined greedy decode ---- */
    int emitted = 0;
    while (emitted < max_new) {
        if (atomic_load(&eng->shutdown) || atomic_load(&s->cancelled)) {
            stream_finish_cancelled(s);
            goto out_free;
        }

        /* 1) sample CURRENT token lazily */
        mlx_array subkey = mlx_array_new();
        if (sampler_key_next(&skey, &subkey, em->stream) != 0) {
            mlx_array_free(subkey);
            generate_fail(eng, s, "sampler key_next failed", GEN_ERR_INTERNAL);
            goto out_free;
        }
        mlx_array lazy_token = mlx_array_new();
        mlx_array lazy_lp = mlx_array_new();
        if (sampler_sample_lazy(pending_logits, sp, subkey, want_lp, em->stream,
                                &lazy_token, &lazy_lp) != 0) {
            mlx_array_free(subkey);
            mlx_array_free(lazy_token);
            mlx_array_free(lazy_lp);
            generate_fail(eng, s, "sampler sample failed", GEN_ERR_INTERNAL);
            goto out_free;
        }
        mlx_array_free(subkey);
        mlx_array_free(pending_logits);
        pending_logits = mlx_array_new();

        /* 2) if more steps remain, build NEXT forward and async_eval both.
           next_logits / step_ids are scoped to the pipelined branch only
           (final-step path never constructs them). */
        mlx_array next_logits;
        bool have_next = false;

        if (emitted + 1 < max_new) {
            next_logits = mlx_array_new();
            mlx_array step_ids = mlx_array_new();
            if (reshape_token_ids(lazy_token, &step_ids, em->stream) != 0) {
                mlx_array_free(step_ids);
                mlx_array_free(next_logits);
                mlx_array_free(lazy_token);
                mlx_array_free(lazy_lp);
                generate_fail(eng, s, "token reshape failed", GEN_ERR_INTERNAL);
                goto out_free;
            }
            if (model_forward(em, step_ids, &kv, true, &next_logits) != 0) {
                mlx_array_free(step_ids);
                mlx_array_free(next_logits);
                mlx_array_free(lazy_token);
                mlx_array_free(lazy_lp);
                generate_fail(eng, s, "decode forward failed", GEN_ERR_INTERNAL);
                goto out_free;
            }
            have_next = true;

            mlx_array batch[3] = {lazy_token, next_logits, lazy_lp};
            size_t batch_n = want_lp ? 3 : 2;
            if (mlxbridge_async_eval_n(batch, batch_n) != 0) {
                mlx_array_free(step_ids);
                mlx_array_free(next_logits);
                mlx_array_free(lazy_token);
                mlx_array_free(lazy_lp);
                generate_fail(eng, s, "async_eval failed", GEN_ERR_INTERNAL);
                goto out_free;
            }
            mlx_array_free(step_ids);
        } else {
            /* last allowed token: force-eval the token alone */
            mlx_array batch[2] = {lazy_token, lazy_lp};
            size_t batch_n = want_lp ? 2 : 1;
            if (mlxbridge_async_eval_n(batch, batch_n) != 0) {
                mlx_array_free(lazy_token);
                mlx_array_free(lazy_lp);
                generate_fail(eng, s, "async_eval failed", GEN_ERR_INTERNAL);
                goto out_free;
            }
        }

        /* 3) NOW materialize CURRENT token */
        int32_t id = 0;
        if (mlxbridge_item_int32(&id, lazy_token) != 0) {
            if (have_next) mlx_array_free(next_logits);
            mlx_array_free(lazy_token);
            mlx_array_free(lazy_lp);
            generate_fail(eng, s, "token materialize failed", GEN_ERR_INTERNAL);
            goto out_free;
        }
        mlx_array_free(lazy_token);

        float lp_val = 0.0f;
        if (want_lp) {
            if (mlxbridge_item_float32(&lp_val, lazy_lp) != 0) {
                if (have_next) mlx_array_free(next_logits);
                mlx_array_free(lazy_lp);
                generate_fail(eng, s, "logprob materialize failed", GEN_ERR_INTERNAL);
                goto out_free;
            }
        }
        mlx_array_free(lazy_lp);

        if (token_is_eos(&em->cfg, id)) {
            if (have_next) mlx_array_free(next_logits);
            if (!stream_push(s, (chunk_t){.tag = CHUNK_DONE, .done = FINISH_STOP}))
                stream_finish_cancelled(s);
            goto out_free;
        }

        chunk_t tok = {.tag = CHUNK_TOKEN, .token = {.id = id, .logprob = lp_val}};
        if (!stream_push(s, tok)) {
            if (have_next) mlx_array_free(next_logits);
            stream_finish_cancelled(s);
            goto out_free;
        }
        emitted++;

        if (emitted == max_new) {
            /* Final step never sets have_next; guard is defensive. */
            if (have_next) mlx_array_free(next_logits);
            if (!stream_push(s, (chunk_t){.tag = CHUNK_DONE, .done = FINISH_LENGTH}))
                stream_finish_cancelled(s);
            goto out_free;
        }

        /* promote next_logits to pending (ownership transfer) */
        mlx_array_free(pending_logits);
        pending_logits = next_logits;

        if (emitted % 256 == 0)
            mlxbridge_clear_cache();
    }

out_free:
    sampler_key_free(&skey);
    mlx_array_free(pending_logits);
    if (kv_inited)
        kvcache_free(&kv);
    mlxbridge_clear_cache();
}

static void handle_generate(engine_t *eng, engine_cmd_t *cmd) {
    stream_t *s = cmd->generate.stream;

    pthread_mutex_lock(&eng->mailbox_mtx);
    eng->inflight = s;
    pthread_mutex_unlock(&eng->mailbox_mtx);

    if (engine_load_state(eng) != LOAD_OK) {
        bool ok = stream_push_error(s, "model not loaded");
        if (ok)
            ok = stream_push(s, (chunk_t){.tag = CHUNK_DONE, .done = FINISH_STOP});
        if (!ok && atomic_load(&eng->shutdown))
            stream_finish_cancelled(s);
        goto done;
    }

    if (eng->model->stub)
        handle_generate_stub(eng, cmd);
    else
        handle_generate_real(eng, cmd);

done:
    pthread_mutex_lock(&eng->mailbox_mtx);
    eng->inflight = NULL;
    pthread_mutex_unlock(&eng->mailbox_mtx);

    stream_release(s);
    free(cmd->generate.token_ids);
    free(cmd);
}

/* ---- CMD_LOAD / CMD_UNLOAD ----------------------------------------------- */

static void handle_load(engine_t *eng, engine_cmd_t *cmd) {
    char *path = cmd->load.model_path;
    free(cmd);

    set_load_state(eng, LOAD_IN_PROGRESS);
    clear_load_error(eng);

    if (path && strcmp(path, MLXD_STUB_MODEL_PATH) == 0) {
        free_resident_model(eng);
        free(eng->loaded_model);
        eng->loaded_model = path;
        memset(eng->model, 0, sizeof(*eng->model));
        eng->model->stub = true;
        set_load_state(eng, LOAD_OK);
        return;
    }

    /* Real load path. free_resident_model runs before the new attempt, so a
       bad reload destroys a previously working resident model (bricks until
       another successful load). Matches "free prior model on reload". */
    free_resident_model(eng);

    char errbuf[256] = {0};
    if (!path || engine_model_load(eng->model, path, errbuf, sizeof(errbuf)) != 0) {
        if (errbuf[0] == '\0')
            snprintf(errbuf, sizeof(errbuf), "failed to load model from %s",
                     path ? path : "(null)");
        set_load_error(eng, errbuf);
        free(path);
        free(eng->loaded_model);
        eng->loaded_model = NULL;
        memset(eng->model, 0, sizeof(*eng->model));
        set_load_state(eng, LOAD_FAILED);
        return;
    }

    eng->model->stub = false;
    free(eng->loaded_model);
    eng->loaded_model = path;
    set_load_state(eng, LOAD_OK);
}

static void handle_unload(engine_t *eng) {
    free_resident_model(eng);
    free(eng->loaded_model);
    eng->loaded_model = NULL;
    memset(eng->model, 0, sizeof(*eng->model));
    clear_load_error(eng);
    set_load_state(eng, LOAD_IDLE);
}

/* ---- Engine thread ------------------------------------------------------- */

static void *engine_thread_main(void *arg) {
    engine_t *eng = arg;

    while (!atomic_load(&eng->shutdown)) {
        engine_cmd_t *cmd = mailbox_dequeue(eng);
        if (!cmd) continue;

        switch (cmd->tag) {
        case CMD_STOP:
            free(cmd);
            mailbox_drain_cancel(eng);
            return NULL;
        case CMD_LOAD:
            handle_load(eng, cmd);
            break;
        case CMD_UNLOAD:
            free(cmd);
            handle_unload(eng);
            break;
        case CMD_GENERATE:
            handle_generate(eng, cmd);
            break;
        case CMD_RECLAIM:
            stream_release(cmd->reclaim.stream);
            free(cmd);
            break;
        case CMD_EMBED:
            free(cmd);
            break;
        }
    }

    mailbox_drain_cancel(eng);
    return NULL;
}

/* ---- Engine public API --------------------------------------------------- */

int engine_init(engine_t *eng) {
    memset(eng, 0, sizeof(*eng));
    eng->model = calloc(1, sizeof(*eng->model));
    if (!eng->model)
        return -1;
    pthread_mutex_init(&eng->mailbox_mtx, NULL);
    pthread_cond_init(&eng->mailbox_cond, NULL);
    pthread_mutex_init(&eng->load_mtx, NULL);
    eng->mailbox_head = NULL;
    eng->mailbox_tail = NULL;
    atomic_store(&eng->shutdown, false);
    atomic_store(&eng->load_state, (int)LOAD_IDLE);
    eng->load_error[0] = '\0';
    eng->loaded_model = NULL;
    eng->inflight = NULL;

    if (pthread_create(&eng->thread, NULL, engine_thread_main, eng) != 0) {
        free(eng->model);
        eng->model = NULL;
        pthread_mutex_destroy(&eng->mailbox_mtx);
        pthread_cond_destroy(&eng->mailbox_cond);
        pthread_mutex_destroy(&eng->load_mtx);
        return -1;
    }
    return 0;
}

void engine_signal_shutdown(engine_t *eng) {
    atomic_store(&eng->shutdown, true);
    pthread_mutex_lock(&eng->mailbox_mtx);
    if (eng->inflight)
        stream_cancel(eng->inflight);
    pthread_cond_signal(&eng->mailbox_cond);
    pthread_mutex_unlock(&eng->mailbox_mtx);
}

void engine_destroy(engine_t *eng) {
    engine_signal_shutdown(eng);
    pthread_join(eng->thread, NULL);

    mailbox_drain_cancel(eng);

    free_resident_model(eng);
    free(eng->model);
    eng->model = NULL;
    free(eng->loaded_model);
    eng->loaded_model = NULL;
    clear_load_error(eng);
    set_load_state(eng, LOAD_IDLE);

    pthread_mutex_destroy(&eng->mailbox_mtx);
    pthread_cond_destroy(&eng->mailbox_cond);
    pthread_mutex_destroy(&eng->load_mtx);
}

/* Enqueue under mailbox_mtx. Caller must hold the lock and have already
   checked shutdown. */
static void mailbox_enqueue_locked(engine_t *eng, engine_cmd_t *cmd) {
    cmd->next = NULL;
    if (eng->mailbox_tail) {
        eng->mailbox_tail->next = cmd;
    } else {
        eng->mailbox_head = cmd;
    }
    eng->mailbox_tail = cmd;
    pthread_cond_signal(&eng->mailbox_cond);
}

void engine_post(engine_t *eng, engine_cmd_t *cmd) {
    if (atomic_load(&eng->shutdown)) {
        cmd_cancel(cmd);
        return;
    }
    pthread_mutex_lock(&eng->mailbox_mtx);
    if (atomic_load(&eng->shutdown)) {
        pthread_mutex_unlock(&eng->mailbox_mtx);
        cmd_cancel(cmd);
        return;
    }
    mailbox_enqueue_locked(eng, cmd);
    pthread_mutex_unlock(&eng->mailbox_mtx);
}

int engine_post_load(engine_t *eng, char *model_path) {
    engine_cmd_t *cmd = calloc(1, sizeof(*cmd));
    if (!cmd) {
        free(model_path);
        return -1;
    }
    cmd->tag = CMD_LOAD;
    cmd->load.model_path = model_path;

    if (atomic_load(&eng->shutdown)) {
        cmd_cancel(cmd);
        return -1;
    }
    pthread_mutex_lock(&eng->mailbox_mtx);
    if (atomic_load(&eng->shutdown)) {
        pthread_mutex_unlock(&eng->mailbox_mtx);
        cmd_cancel(cmd);
        return -1;
    }
    /* Synchronize the waiter with "a load is outstanding". */
    clear_load_error(eng);
    set_load_state(eng, LOAD_IN_PROGRESS);
    mailbox_enqueue_locked(eng, cmd);
    pthread_mutex_unlock(&eng->mailbox_mtx);
    return 0;
}
