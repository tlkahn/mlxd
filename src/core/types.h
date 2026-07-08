#ifndef MLXD_CORE_TYPES_H
#define MLXD_CORE_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* --- Finish reason -------------------------------------------------------- */

typedef enum {
    FINISH_STOP,
    FINISH_LENGTH,
    FINISH_TOOL_CALLS,
    FINISH_CONTENT_FILTER,
    FINISH_CANCELLED,
} finish_reason_t;

const char *finish_reason_str(finish_reason_t r);
const char *finish_reason_wire_str(finish_reason_t r);

/* --- Sampling parameters -------------------------------------------------- */

typedef struct {
    float temperature;
    float top_p;
    int   top_k;
    float min_p;
    int   seed;
} sampling_params_t;

#define SAMPLING_PARAMS_DEFAULT \
    ((sampling_params_t){.temperature = 1.0f, .top_p = 1.0f, .top_k = -1, .min_p = 0.0f, .seed = -1})

bool sampling_params_validate(const sampling_params_t *p, const char **err);

/* --- Generation parameters ------------------------------------------------ */

typedef struct {
    sampling_params_t sampling;
    int      max_tokens;
    int      n;
    bool     stream;
    bool     logprobs;
    int      top_logprobs;
    char   **stop;
    int      stop_count;
} gen_params_t;

/* --- Role ----------------------------------------------------------------- */

typedef enum {
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_TOOL,
} role_t;

const char *role_str(role_t r);

/* --- Tool call ------------------------------------------------------------ */

typedef struct {
    char *id;
    char *function_name;
    char *arguments;
} tool_call_t;

void tool_call_free(tool_call_t *tc);

/* --- Message -------------------------------------------------------------- */

typedef struct {
    role_t       role;
    char        *content;
    char        *name;
    tool_call_t *tool_calls;
    int          tool_call_count;
    char        *tool_call_id;
} message_t;

void message_free(message_t *msg);

/* --- Chunk (per-request stream item) -------------------------------------- */

typedef enum {
    CHUNK_TOKEN,
    CHUNK_DONE,
    CHUNK_ERROR,
} chunk_tag_t;

typedef struct {
    chunk_tag_t tag;
    union {
        struct {
            int32_t id;
            float   logprob;
        } token;
        finish_reason_t done;
        char           *error;
    };
} chunk_t;

/* --- Usage ---------------------------------------------------------------- */

typedef struct {
    int prompt_tokens;
    int completion_tokens;
    int total_tokens;
} usage_t;

#endif
