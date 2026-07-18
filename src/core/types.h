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

/* --- Sampling set-mask bits (which fields were explicitly set on the wire) -- */

#define SAMPLING_SET_TEMPERATURE (1u << 0)
#define SAMPLING_SET_TOP_P       (1u << 1)
#define SAMPLING_SET_TOP_K       (1u << 2)
#define SAMPLING_SET_MIN_P       (1u << 3)
#define SAMPLING_SET_SEED        (1u << 4)

/* --- Generation parameters ------------------------------------------------ */

typedef struct {
    sampling_params_t sampling;
    unsigned sampling_set;
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

/* --- Message content (string | multimodal parts) -------------------------- */

/* A content part's image URL. Only the url is retained; detail is deferred. */
typedef struct {
    char *url;
} image_url_t;

/* One element of a multimodal content array. `type` is "text" or "image_url";
 * `text` is set for text parts, `image_url` for image parts. */
typedef struct {
    char        *type;
    char        *text;
    bool         has_image_url;
    image_url_t  image_url;
} content_part_t;

typedef enum {
    CONTENT_NONE,   /* absent or JSON null */
    CONTENT_STRING, /* a plain string */
    CONTENT_PARTS,  /* a multimodal array */
} content_kind_t;

/* Mirrors the OpenAI StringOr(ContentPart) shape; v2 vision needs the parts. */
typedef struct {
    content_kind_t  kind;
    char           *string;
    content_part_t *parts;
    int             part_count;
} message_content_t;

void content_part_free(content_part_t *part);
void message_content_free(message_content_t *content);

/* --- Message -------------------------------------------------------------- */

typedef struct {
    role_t            role;
    message_content_t content;
    char             *name;
    tool_call_t      *tool_calls;
    int               tool_call_count;
    char             *tool_call_id;
} message_t;

void message_free(message_t *msg);

/* --- Tools (request-side function definitions) ---------------------------- */

typedef struct {
    char *name;
    char *description;
    char *parameters_json; /* raw JSON object text, via yyjson_val_write */
} function_def_t;

typedef struct {
    char          *type; /* always "function" today */
    function_def_t function;
} tool_t;

void tool_free(tool_t *tool);

typedef enum {
    TOOL_CHOICE_UNSET,
    TOOL_CHOICE_AUTO,
    TOOL_CHOICE_NONE,
    TOOL_CHOICE_REQUIRED,
    TOOL_CHOICE_FUNCTION,
} tool_choice_kind_t;

typedef struct {
    tool_choice_kind_t kind;
    char              *function_name; /* set only for TOOL_CHOICE_FUNCTION */
} tool_choice_t;

void tool_choice_free(tool_choice_t *choice);

/* --- Chunk (per-request stream item) -------------------------------------- */

typedef enum {
    CHUNK_TOKEN,
    CHUNK_DONE,
    CHUNK_ERROR,
} chunk_tag_t;

typedef enum {
    GEN_ERR_INTERNAL = 0,
    GEN_ERR_CONTEXT_LENGTH,
} gen_err_kind_t;

typedef struct {
    chunk_tag_t    tag;
    gen_err_kind_t error_kind; /* meaningful only when tag == CHUNK_ERROR; zero-init = INTERNAL */
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
