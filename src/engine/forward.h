#ifndef MLXD_ENGINE_FORWARD_H
#define MLXD_ENGINE_FORWARD_H

#include "mlxbridge/mlxbridge.h"
#include "model/model.h"
#include "model/weights.h"
#include "engine/kvcache.h"

#include <stdbool.h>

/* *out must be a live mlx_array (e.g. from mlx_array_new()). On success the
   previous value is freed and replaced; unchanged on failure. */

int fwd_linear(mlx_array *out, mlx_array x, const weight_triplet_t *tri,
               const model_config_t *cfg, mlx_stream s);

int fwd_embed(mlx_array *out, mlx_array ids, const weights_t *w,
              const model_config_t *cfg, mlx_stream s);

/* add_unit_offset applies the gemma-style (1 + weight) variant: bf16 1.0 is
   added to the raw weight before the norm. */
int fwd_rmsnorm(mlx_array *out, mlx_array x, mlx_array weight, float eps,
                bool add_unit_offset, mlx_stream s);

int fwd_attention(mlx_array *out, mlx_array x, int layer,
                  const weights_t *w, const model_config_t *cfg,
                  kvcache_t *kv, mlx_array rope_freqs, mlx_stream s);

/* Sliding-window additive masks (bf16, 0 / -inf) for SDPA mask mode "array".
   Prefill mask is [1,1,q_len,kv_len]: causal + window over absolute
   positions, where query row i sits at absolute position kv_len - q_len + i.
   Decode mask is [1,1,1,kv_len]: positions before kv_len - window masked. */
int fwd_sliding_window_mask(mlx_array *out, int q_len, int kv_len, int window,
                            mlx_stream s);

int fwd_sliding_window_decode_mask(mlx_array *out, int kv_len, int window,
                                   mlx_stream s);

int fwd_softcap(mlx_array *out, mlx_array logits, float cap, mlx_stream s);

int fwd_gelu_approx(mlx_array *out, mlx_array x, mlx_stream s);

int fwd_swiglu(mlx_array *out, mlx_array x, int layer,
               const weights_t *w, const model_config_t *cfg, mlx_stream s);

int fwd_ple_inputs(mlx_array *out, mlx_array ids, mlx_array h,
                   const weights_t *w, const model_config_t *cfg,
                   mlx_stream s);

int fwd_ple_apply(mlx_array *io_h, int layer, mlx_array ple_inputs,
                  const weights_t *w, const model_config_t *cfg,
                  mlx_stream s);

int fwd_layer_scalar_apply(mlx_array *io, int layer, const weights_t *w,
                           const model_config_t *cfg, mlx_stream s);

int fwd_decoder_layer(mlx_array *out, mlx_array x, int layer,
                      const weights_t *w, const model_config_t *cfg,
                      kvcache_t *kv, mlx_array rope_freqs,
                      mlx_array ple_inputs, mlx_stream s);

int fwd_rope_llama3_freqs(const model_config_t *cfg, float *out, int n);
int fwd_rope_proportional_freqs(const model_config_t *cfg, float *out, int n);

int fwd_rope_apply(mlx_array *res, mlx_array x, int dims,
                   const model_config_t *cfg, int layer, int offset,
                   mlx_array rope_freqs, mlx_stream s);

/* Build precomputed freqs array for the model's rope scaling type.
   rc 0 + out->ctx == NULL: family needs no custom freqs.
   rc 0 + live array: [head_dim/2] f32.
   rc -1: validation or alloc failure. */
int fwd_rope_freqs_build(mlx_array *out, const model_config_t *cfg);

/* ---- MoE block (family-agnostic; DeepSeek group gate is #108) ---- */

/* Expert weight layout contract (caller-supplied; no internal transpose):
   - Quantized triplets: weight [E, out, in] (+ matching scales/biases),
     dispatched via mlx_gather_qmm with transpose=true.
   - bf16 triplets: weight pre-shaped [E, in, out], dispatched via
     mlx_gather_mm (sorted_indices always false on mlx 0.31.2).
   - Expert stacks are bias-free today (no SwitchLinear bias[indices] path);
     Qwen/DeepSeek MoE experts do not ship expert bias.
   Presence is flag-driven on this struct:
   - has_shared / has_shared_expert_gate select shared residual modes.
   - Absent optional weights are zeroed triplets (weight.ctx == NULL), not
     NULL pointers (fields are by value).
   For fwd_switch_glu / fwd_gather_expert_linear args, pass a NULL
   weight_triplet_t* or a zeroed triplet to mean absent. */
typedef struct {
    weight_triplet_t router;            /* [E, H] or quant equiv via fwd_linear */
    weight_triplet_t switch_gate;       /* expert stack */
    weight_triplet_t switch_up;
    weight_triplet_t switch_down;
    /* shared expert dense SwiGLU (non-stacked); used when has_shared */
    weight_triplet_t shared_gate;
    weight_triplet_t shared_up;
    weight_triplet_t shared_down;
    /* optional scalar gate for qwen-style shared; always-active when
       has_shared is set and has_shared_expert_gate is false */
    weight_triplet_t shared_expert_gate;
    bool has_shared;
    bool has_shared_expert_gate;
} fwd_moe_weights_t;

typedef struct {
    int  num_experts;   /* E */
    int  top_k;         /* K */
    bool norm_topk_prob;
} fwd_moe_params_t;

/* Softmax top-k route. router_logits [..., E] -> owned inds [..., K] (int)
   and scores [..., K]. Caller frees both on success. Index order within the
   top-k set is partition-unstable (set equality only). */
int fwd_moe_route_softmax(mlx_array *inds, mlx_array *scores,
                          mlx_array router_logits, int top_k,
                          bool norm_topk_prob, mlx_stream s);

/* Gathered expert linear: x @ expert[inds]. Quant uses gather_qmm
   (transpose=true); bf16 uses gather_mm (sorted_indices forced false).
   sorted is honored only on the quantized path (R4). */
int fwd_gather_expert_linear(mlx_array *out, mlx_array x, mlx_array inds,
                             const weight_triplet_t *tri,
                             mlx_array lhs_indices, /* may be null ctx */
                             bool sorted,
                             const model_config_t *cfg, mlx_stream s);

/* Expert SwiGLU gather. x [B,S,H], inds [B,S,K] -> out [B,S,K,H].
   Sort threshold (R2): S > 1 || B*S*K >= 64. */
int fwd_switch_glu(mlx_array *out, mlx_array x, mlx_array inds,
                   const weight_triplet_t *gate,
                   const weight_triplet_t *up,
                   const weight_triplet_t *down,
                   const model_config_t *cfg, mlx_stream s);

/* Score-combine + optional shared residual.
   y_exp [B,S,K,H], scores [B,S,K] -> out [B,S,H].
   Weighted sum over K, then optional shared residual from mw flags.
   x is required for shared expert(s). *out replaced only on success. */
int fwd_moe_combine(mlx_array *out,
                    mlx_array y_exp,
                    mlx_array scores,
                    mlx_array x,
                    const fwd_moe_weights_t *mw,
                    const model_config_t *cfg,
                    mlx_stream s);

/* Full MoE block. x [B,S,H] -> out [B,S,H]. Softmax route + switch +
   score-combine + optional shared expert residual. Convenience wrapper
   over route_softmax + switch_glu + combine. */
int fwd_moe(mlx_array *out, mlx_array x,
            const fwd_moe_weights_t *mw,
            const fwd_moe_params_t *p,
            const model_config_t *cfg,
            mlx_stream s);

/* Materialize a row-contiguous copy. Required after any lazy slice of
   expert weight stacks before gather_qmm (silent-zeros gotcha). */
int fwd_array_contiguous(mlx_array *out, mlx_array in, mlx_stream s);

#endif /* MLXD_ENGINE_FORWARD_H */
