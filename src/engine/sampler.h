#ifndef MLXD_ENGINE_SAMPLER_H
#define MLXD_ENGINE_SAMPLER_H

#include "core/types.h"
#include "mlxbridge/mlxbridge.h"

/* Stage C sampler pipeline. All functions are lazy (no eval) and run on the
 * engine thread's stream; masking filters write -inf, never scatter. */

/* top_k filter: mask everything below the k-th largest logit to -inf.
 * k <= 0 (disabled) or k >= vocab returns the logits unchanged (lazy copy).
 * Caller owns *out. */
int sampler_apply_top_k(mlx_array logits, int k, mlx_stream s, mlx_array *out);

/* top_p (nucleus) filter: mask tokens outside the smallest set whose
 * probability mass reaches p. p >= 1 returns the logits unchanged (lazy copy).
 * Caller owns *out. */
int sampler_apply_top_p(mlx_array logits, float p, mlx_stream s, mlx_array *out);

/* Full pipeline: temperature scale, top_k, top_p, min_p, categorical.
 * temperature < 0.01 short-circuits to argmax (greedy limit).
 * subkey: pass an empty mlx_array for the nondeterministic default.
 * Caller owns *tok_out. logprob_out is untouched unless want_logprob. */
int sampler_sample_lazy(mlx_array logits, const sampling_params_t *sp, mlx_array subkey,
                        bool want_logprob, mlx_stream s, mlx_array *tok_out,
                        mlx_array *logprob_out);

#endif
