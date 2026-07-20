# Engine Inference Core

Roadmap for [issue #51](https://github.com/tlkahn/mlxd/issues/51): replace the engine's stub echo loop (`handle_generate` in `src/engine/engine.c`) with real MLX text generation. All actor infrastructure from #5 (MPSC mailbox, `stream_t` chunk buffer, shutdown/drain) and all HTTP plumbing from #7 stay as-is; this work fills in model loading, per-family forward passes, KV cache, sampling, and the prefill/decode loop.

Stage issues (one PR per stage, close with `Closes #NN (Stage X of #51)`):

| Issue | Stage | Depends on |
|---|---|---|
| [#52](https://github.com/tlkahn/mlxd/issues/52) | A: loading foundations | - |
| [#53](https://github.com/tlkahn/mlxd/issues/53) | B: qwen3 dense vertical slice | A |
| [#54](https://github.com/tlkahn/mlxd/issues/54) | C: sampling + generation params | B |
| [#55](https://github.com/tlkahn/mlxd/issues/55) | D: dense family breadth | C |
| [#56](https://github.com/tlkahn/mlxd/issues/56) | E: MoE + hybrid families | D |
| [#57](https://github.com/tlkahn/mlxd/issues/57) | F: lifecycle + memory hardening (closes #46) | B (parallel with D/E) |

---

## Current baseline (pre-Stage A)

What already exists, and what is still a stub:

| Surface | State |
|---|---|
| `src/engine/` | Actor only: MPSC mailbox, `stream_t`, shutdown/drain. `handle_generate` echoes prompt token IDs. `CMD_LOAD` stores the path string and flips `engine_loaded`; no weights. `CMD_EMBED` accepted and dropped. |
| `src/model/` | Tokenizer (BPE / WordPiece / SentencePiece), detokenizer, chat-template load, minimal `model_config_t` (`model_type`, `architectures`, dims/heads). No safetensors, no family enum, no generation defaults. |
| `src/mlxbridge/` | Four helpers: GPU stream, print, shape query, `MLXB_CHECK`. No safetensors, eval, sampling, or memory-limit wrappers. |
| `src/registry/` | HF download/resolve; can fetch safetensors files, nothing consumes them. |
| `src/http/`, `src/cli/` | Full serve/run path wired to the echo engine. |
| `tests/test_mlxbridge_gpu.c` | Metal smoke tests for the ops Stage B will call (RMSNorm, RoPE, SDPA, quantized matmul, categorical sampling, safetensors load, etc.). Call-pattern reference for new wrappers. |
| `tests/test_engine.c`, `gen_server_harness.h` | CPU-only suites that assert echo-stub behavior; Stage B must keep them green via an explicit stub mode. |

## Migration source

The full Zig implementation is `tlkahn/mlx-serve` (local checkout `../../mlx-serve` relative to the repo root). Text-gen core:

- `src/transformer.zig` (~9.6k lines) — config-driven forward: dense `forwardStandardWith`, MoE `forwardMoeWith`, hybrid `forwardHybridWith`
- `src/generate.zig` (~5k lines) — chunked prefill + GPU sampling
- `src/model.zig` — config parse + safetensors load
- `src/arch/llama.zig`, `src/arch/ds4.zig` — family specifics
- deferred follow-ups live in `src/kv_quant.zig`, `src/prefix_cache.zig`

For DeepSeek V3/V3.2/V4: mlx-serve wraps a separate `ds4` engine (GGUF, custom Metal kernels) - **do not port** that path. Instead port from mlx-lm's Python implementation (`deepseek_v3.py`, `deepseek_v32.py`, `mla.py`) which runs natively on MLX safetensors. mlx-lm keeps **separate modules** for V3 vs V3.2 (DSA indexer); mlxd mirrors that with separate `model_family_t` values (decision 5 / 11). Do not port from `ds4.zig`.

`tlkahn/mlxd-zig` never implemented its engine (7-line stub). Its only reusable inference artifact is `src/mlxbridge/root.zig` (~190 mlx-c externs). On the C side, `tests/test_mlxbridge_gpu.c` is the reference for call patterns.

## Intentional divergences from mlx-serve

mlx-serve is the port source, not a ceiling. mlxd aims for a feature-complete text engine under YOGNI (Unix-philosophy modules, no speculative bloat): cover the HF/MLX checkpoint surface real users hit, even where the Zig parent cut corners. These three are **required**, not optional niceties - do not "simplify" them away to match mlx-serve.

| Area | mlx-serve today | mlxd (this plan) | Why |
|---|---|---|---|
| **Shard loading** | Globs every `*.safetensors` in the model dir; never reads `model.safetensors.index.json` / `weight_map` | Enumerate single `model.safetensors` **or** sharded `model-*-of-*.safetensors` guided by `model.safetensors.index.json` (yyjson `weight_map`); fall back to glob only when no index is present | Correct multi-shard HF layouts, avoids loading stray/non-model safetensors, enables missing-tensor diagnostics against the declared weight set |
| **Llama3 rope scaling** | Only Gemma3 `rope_scaling.factor` and Gemma4 `rope_type:"proportional"`; no `low_freq_factor` / `high_freq_factor` / `original_max_position_embeddings` / `rope_type:"llama3"` | Parse full rope-scaling block on `model_config_t` (type/factor/low-high freq/original max position) and apply llama3-style factor scheduling in the dense forward (Stage D) | Llama-3/3.1-class checkpoints are first-class families; without this their long-context positions are wrong |
| **EOS + sampling defaults source** | `eos_token_id` from `config.json` only; `generation_config.json` contributes temperature/top_p/top_k (nullable) and is silent on EOS | Parse `generation_config.json` for **both** default sampling values **and** the eos token id list (scalar or array); merge with `config.json` eos (union, de-duped, fixed cap). Defaults precedence: request > `generation_config.json` > `SAMPLING_PARAMS_DEFAULT`. Multi-eos termination in the decode loop (Stage C) | Matches HuggingFace generation semantics; many chat models put the real stop set in `generation_config.json` |

Everything else ports mlx-serve's behavior unless a stage task says otherwise.

## Resolved design decisions (2026-07-17)

Decisions from the pre-Stage-A/B review ([#51 comment](https://github.com/tlkahn/mlxd/issues/51#issuecomment-5001232699)); each is folded into the stage tasks below.

| # | Decision | Lands in |
|---|---|---|
| 1 | **Load failure contract**: replace `atomic_bool loaded` with a load-state enum (`LOAD_IDLE` / `LOAD_IN_PROGRESS` / `LOAD_OK` / `LOAD_FAILED`) plus a mutex-guarded last-error string; thread-safe getters `engine_load_state()` / `engine_load_error()`. Posting stays fire-and-forget; CLI/serve pollers distinguish in-progress from failed. | B3 (API change), A1 names the contract |
| 2 | **Quantization modes**: parse `quantization.mode`; support **affine only**; hard-error with a clear message on `nvfp4` / `mxfp4` / `mxfp8` (never silent-ignore). Non-affine modes are a named follow-up. | A1 (parse), A3/B1 (enforce) |
| 3 | **Config surface**: A1 owns the **full** `model_config_t` surface, including `weight_prefix` and behavioral flags (`has_qk_norm`, `norm_has_offset`, `scale_embeddings`, `has_pre_ff_norm`, `hidden_act`, `tie_word_embeddings`, dual rope bases), with per-family defaults, even though only qwen3 dense is exercised until D/E. No per-stage `model_config_t` rewrites. | A1 |
| 4 | **qwen3_5 family split**: `MODEL_QWEN3_5` (dense) and `MODEL_QWEN3_5_MOE` are **separate** `model_family_t` values, discriminated at config-parse time via MoE fields. Do not copy mlx-serve's collapsed tag. | A1, D/E boundary |
| 5 | **~~deepseek_v4 = GGUF~~** (superseded 2026-07-20): mlx-serve wraps a separate `ds4` engine via FFI for this family, but mlx-lm (Apple's Python reference) implements DeepSeek V3/V3.2/V4 natively on MLX using **safetensors** with standard mlx ops. The architecture needs MLA (Multi-head Latent Attention) and MoE, both implementable via mlx-c. No GGUF loader needed - the Stage A safetensors path works. Stage E ports from mlx-lm's `deepseek_v3.py` / `deepseek_v32.py` + `mla.py`, not from `ds4.zig`. Target checkpoint: `mlx-community/DeepSeek-V4-Flash-4bit` (safetensors). | E |
| 11 | **DeepSeek family enum grain** (2026-07-20): do **not** collapse `deepseek_v3` / `deepseek_v3_2`/`deepseek_v32` / `deepseek_v4` into a single `MODEL_DEEPSEEK_V4`. mlx-lm keeps separate modules; V3.2 adds a DSA **Indexer** plus dual cache (`CacheList`) - a real forward/cache fork, not a config alias (same grain as decision 4 for qwen3_5 dense vs MoE). Mapping: `deepseek_v3` -> `MODEL_DEEPSEEK_V3` (MLA+MoE); `deepseek_v3_2`/`deepseek_v32` -> `MODEL_DEEPSEEK_V32` (MLA+MoE+DSA); `deepseek_v4` -> `MODEL_DEEPSEEK_V4` (own value from day one - only alias onto V3 or V32 after the Flash `config.json`/weights are characterized). Shared code is primitives (`fwd_mla_*`, MoE gate, sanitize helpers), not one family tag. HF aliases `v3_2`/`v32` map together. Reject/error strings must name the actual family (never report "deepseek_v4" for a V3 load). **A1 interim debt:** current code still maps all four strings to `MODEL_DEEPSEEK_V4`; correct the enum + `model_family_from_type()` + tests before or at the start of Stage E2. | A1 correction, E2 |
| 6 | **Prefill chunk**: named constant `MLXD_PREFILL_CHUNK = 512` (cancel grain vs kernel-launch overhead); no env override in v1. | B3 |
| 7 | **Load vs tokenizer ownership**: engine `CMD_LOAD` = config + weights + eos set only; tokenizer stays caller-side (HTTP/CLI), never on the engine thread. | B3 |
| 8 | **Oversized prompts**: prompt token count exceeding `max_position_embeddings` / KV capacity **errors the stream** with token count and limit in the message (HTTP 400, `context_length_exceeded` style). No truncation, silent or otherwise. | B3 |
| 9 | **Compute dtype**: activations default to **bf16** (mlx-serve-like) unless the checkpoint dtype forces otherwise. No ad-hoc f32 "for simplicity". | B1 |
| 10 | **Stub seam**: single sentinel `MLXD_STUB_MODEL_PATH = "stub"` (named constant in `engine.h`), compared exactly in `CMD_LOAD`. CPU suites must load the stub; real load never matches it. No other sentinel conventions in tests. | B3, test strategy |

Named deferrals confirmed (see Out of scope): string `stop` sequences, `n > 1`, `top_logprobs` list (Stage C = single logprob only), repeat/presence penalties, non-affine quant modes, Core AI (#50, orthogonal).

## Architecture invariants

Restated from `CLAUDE.md` — every stage must respect them:

- **Single engine thread** is the sole mlx-c caller, including all frees. Forward pass, KV cache, and sampler code live in `src/engine/`; weight/config loading in `src/model/` is invoked only from the engine thread.
- **Engine emits token IDs, not text.** Detokenization stays on the event-loop side (`detok_t`).
- **No function-pointer dispatch:** `model_family_t` enum + switch.
- **Bounded drain (#46):** the decode loop polls shutdown/cancel at eval-step grain, and prefill is chunked with a poll between chunks, so a single mlx op cannot hold the engine thread past the drain deadline.

## Parity oracle

The local mlx-serve Zig binary. Each family's acceptance gate is a **temp-0 first-50-token match** on the same 4-bit checkpoint (mlx-serve is itself cross-checked against mlx-lm).

Parity/CI hygiene: the parity script takes the binary via `MLXD_MLX_SERVE_BIN` plus a checkpoint path, and **skips if either is absent - it never fails CI** when the Zig binary or checkpoint is missing. Real-checkpoint GPU tests follow the same skip-if-absent pattern. Canonical checkpoint ids are recorded in the parity script itself (Stage D requires this).

## Test strategy

- CPU-only tests (`make test`) never execute mlx graph code.
- Anything that runs ops goes in `tests/test_*_gpu.c` (`make test-gpu`).
- A tiny random-weight qwen3-shaped checkpoint fixture (generated by a small tool using `mlx_save_safetensors`, committed under `tests/fixtures/`) makes GPU tests hermetic.
- Parity gates against real checkpoints are skip-if-absent (e.g. registry-cache Qwen3-0.6B-4bit).
- Stage B introduces the explicit stub/echo engine mode behind the single sentinel `MLXD_STUB_MODEL_PATH = "stub"` (decision 10) so `test_engine.c` and the HTTP gen harness keep running CPU-only; real-inference assertions move to `_gpu.c` tests.

## Known gotchas (carry over as tests)

- `mlx_load_safetensors` must run on a **CPU stream** (GPU stream aborts).
- Inputs to `mlx_gather_qmm` must be **materialized contiguous**.
- Call `mlx_clear_cache` periodically during long decodes (and after each request).
- DeepSeek `kv_b_proj` weight must be split into per-head `embed_q` + `unembed_out` at load time (reshape + slice + requantize); mlx-lm's `sanitize()` does this in Python.

---

## Stages

### Stage A: loading foundations (config, mlxbridge safetensors surface, weight loader)

**Issue:** [#52](https://github.com/tlkahn/mlxd/issues/52)

Everything needed to get checkpoint bytes into named mlx arrays, before any forward pass exists. Green + demoable: a GPU test loads the Qwen3-0.6B-4bit fixture-cache checkpoint and reports tensor count and bytes.

#### Phase A1: model config extension + family dispatch

- [ ] Extend `model_config_t` (`src/model/model.h`) with the **full** transformer config surface (decision 3 - owned by A1 once, no per-stage rewrites): `head_dim`, `intermediate_size`, `rope_theta`, **full rope scaling block** (type / factor / `low_freq_factor` / `high_freq_factor` / `original_max_position_embeddings` - intentional superset of mlx-serve; see divergences), `rms_norm_eps`, `sliding_window` (+ layer pattern where applicable), `attention_bias`, `max_position_embeddings`, plus `weight_prefix` and the per-family behavioral flags the forward pass branches on: `has_qk_norm`, `norm_has_offset`, `scale_embeddings`, `has_pre_ff_norm`, `hidden_act`, `tie_word_embeddings`, dual rope bases. Per-family defaults; fields unused until D/E sit with defaults, covered by parse tests.
- [ ] Quantization block: `group_size`, `bits`, **and `mode`** (decision 2). Support `affine` only; hard-error with a clear message on `nvfp4` / `mxfp4` / `mxfp8` - never silent-ignore (loads then crashes mid-matmul). Non-affine modes are a named follow-up.
- [ ] Add `model_family_t` enum + `model_family_from_type()` keyed off `model_type` (gemma3, gemma4, qwen2, qwen3, qwen3_5, qwen3_5_moe, llama, mistral, lfm2, nemotron_h, deepseek_v3, deepseek_v32, deepseek_v4; bert recognized but rejected for generate). `MODEL_QWEN3_5` and `MODEL_QWEN3_5_MOE` are **separate values** discriminated at config-parse time via MoE fields (decision 4 - do not copy mlx-serve's collapsed tag); this keeps the Stage D dense / Stage E MoE boundary an enum case, not a runtime branch. DeepSeek likewise uses **separate** values `MODEL_DEEPSEEK_V3` / `MODEL_DEEPSEEK_V32` / `MODEL_DEEPSEEK_V4` (decision 11) - do not collapse V3/V3.2 into V4. (A1 shipped an interim collapse to `MODEL_DEEPSEEK_V4`; correct before E2.)
- [ ] Handle nested text-config shapes (e.g. multimodal-wrapped `text_config`) the way `model_discovery.zig` / `model.zig` does (`model_type` from root; dimensional fields from `text_config` when present)
- [ ] Parse `generation_config.json` alongside `config.json`: default sampling values **and** eos token id list (scalar or array). Merge eos with any `config.json` `eos_token_id` (union, de-duped, fixed cap). Missing/malformed `generation_config.json` is non-fatal (leave defaults unset). Intentional: mlx-serve ignores eos here.
- [ ] CPU tests: per-family fixture configs under `tests/fixtures/`, including malformed/missing-field cases following the existing `model_config_*` fixture pattern; cover rope-scaling full block, generation_config eos list, config+generation eos merge, `weight_prefix` + behavioral-flag defaults per family, quant `mode` parse (affine accepted, nvfp4/mxfp* rejected), and qwen3_5 dense-vs-moe discrimination

#### Phase A2: mlxbridge safetensors + eval/memory helpers

- [ ] `mlxbridge` safetensors load helper pinned to the CPU stream, returning `mlx_map_string_to_array`; document + test the CPU-stream-only gotcha
- [ ] Map lookup/iteration helpers and a free-whole-map helper (all frees on the calling engine thread)
- [ ] Eval helpers: `mlx_async_eval` wrapper, int32 item extraction, synchronize
- [ ] Memory helpers: set wired/cache limits, query active/peak memory, clear cache
- [ ] Extend `tests/test_mlxbridge_gpu.c` with smoke tests for each new helper

#### Phase A3: weight loader

- [ ] Shard enumeration in `src/model/`: prefer `model.safetensors.index.json` (`weight_map` via yyjson) to decide which `model-*-of-*.safetensors` shards to open; accept a single unsharded `model.safetensors`; fall back to `*.safetensors` glob only when no index exists. Intentional: mlx-serve always globs and never reads the index.
- [ ] Name-to-tensor lookup table with quantized triplet grouping (`weight`/`scales`/`biases`), keyed under the family's `weight_prefix` from A1 (`model` vs `language_model.model` vs `backbone`, etc.); affine-only enforcement from the parsed quant `mode` (decision 2)
- [ ] Validation against config: expected tensor set per family and layer count, clear error on missing tensor or dtype mismatch (and, when an index is present, against the declared `weight_map` keys)
- [ ] Fixture generator tool (using `mlx_save_safetensors`) emitting a tiny random-weight qwen3-shaped checkpoint (2 layers, small dims) into `tests/fixtures/`; commit the generated fixture. Include both a single-file fixture and a tiny sharded+index fixture so the index path is tested.
- [ ] GPU test: load the tiny fixture end-to-end (single + sharded); skip-if-absent load of a real Qwen3-0.6B-4bit checkpoint from the registry cache

### Stage B: qwen3 dense vertical slice (forward pass, KV cache, greedy decode)

**Issue:** [#53](https://github.com/tlkahn/mlxd/issues/53) · **Depends on:** Stage A

One model end-to-end before any breadth: qwen3 dense (standard GQA attention + QK-norm, SwiGLU, no MoE/SSM), greedy decode only. Green + demoable: `mlxd run` produces real text from Qwen3-0.6B-4bit, temp-0 parity gate passes against mlx-serve.

#### Phase B1: forward building blocks (`src/engine/`)

- [ ] Compute dtype: activations default to **bf16** unless the checkpoint dtype forces otherwise (decision 9); no ad-hoc f32 paths
- [ ] Linear helper honoring quantization: `mlx_quantized_matmul` for quantized weights (affine mode only, guaranteed by A1/A3 rejection), `mlx_matmul` (+ bias add) otherwise, with contiguous-input guarantee
- [ ] Embedding lookup (take), including quantized-embedding dequant path
- [ ] Attention block: GQA q/k/v projections, QK-norm (`mlx_fast_rms_norm`), `mlx_fast_rope`, `mlx_fast_scaled_dot_product_attention` with causal mask mode, output projection
- [ ] SwiGLU MLP: gate/up/down with SiLU via `mlx_sigmoid`/`mlx_multiply`
- [ ] Decoder layer: pre/post RMSNorm residual wiring; final norm + lm_head (tied embeddings supported)
- [ ] GPU test: single-layer forward shape/dtype checks on the tiny fixture

#### Phase B2: KV cache

- [ ] `kvcache_t`: per-layer K/V buffers `[B, H, T, D]`, logical offset, capacity-doubling growth via `mlx_slice_update`, dense view handed to SDPA (port of `KVCacheEntry`/`updateDense`/`denseView` from `transformer.zig`)
- [ ] Bulk update for prefill chunks + single-step update for decode
- [ ] GPU test: incremental decode equals full-context forward on the tiny fixture (logit equivalence within tolerance)

#### Phase B3: engine wiring (prefill + decode loop)

- [ ] **Load failure contract** (decision 1): replace `atomic_bool loaded` with a load-state enum (`LOAD_IDLE` / `LOAD_IN_PROGRESS` / `LOAD_OK` / `LOAD_FAILED`) + mutex-guarded last-error string on `engine_t`; add thread-safe `engine_load_state()` / `engine_load_error()`; keep `engine_loaded()` as a compat shim (`== LOAD_OK`). Posting stays fire-and-forget; update CLI/serve pollers to fail fast with the error message on `LOAD_FAILED` instead of timing out.
- [ ] `CMD_LOAD`: real load on the engine thread - config + weights + eos ids into engine-owned model state, **nothing else**: tokenizer stays caller-side on HTTP/CLI (decision 7). Any load failure sets `LOAD_FAILED` + error string; success sets `LOAD_OK`.
- [ ] Chunked prefill: named constant `MLXD_PREFILL_CHUNK = 512` (cancel grain vs kernel-launch overhead; no env override in v1 - decision 6) with shutdown/cancel poll between chunks
- [ ] **Oversized prompt policy** (decision 8): prompt token count > `max_position_embeddings` (or KV capacity) errors the stream with a `CHUNK_ERR` naming the token count and the limit; HTTP maps it to 400 (`context_length_exceeded` style). No truncation.
- [ ] Decode loop with `mlx_async_eval` pipelining (mlx-lm cadence: build next step, async_eval, extract current item), emitting `CHUNK_TOKEN` ids into `stream_t`
- [ ] Termination: eos id -> `FINISH_STOP`, `max_tokens` -> `FINISH_LENGTH`, shutdown/cancel -> `stream_finish_cancelled`, polled at eval-step grain (#46)
- [ ] Greedy sampling path (`mlx_argmax_axis`) as the temp-0 baseline
- [ ] **Stub seam** (decision 10): single sentinel `MLXD_STUB_MODEL_PATH = "stub"` (named constant in `engine.h`), compared exactly in `CMD_LOAD`; CPU suites (`test_engine.c`, HTTP gen harness) must load the stub, real load never matches it, and no other sentinel convention appears in tests. Real-inference assertions move to `_gpu.c` tests.
- [ ] All mlx frees on the engine thread; ASan/UBSan clean on the GPU smoke test

#### Phase B4: end-to-end demo + parity gate

- [ ] `mlxd run` one-shot generation works against Qwen3-0.6B-4bit
- [ ] `mlxd serve` + `/v1/chat/completions` streaming works end-to-end with real tokens
- [ ] Parity script (`scripts/`): temp-0 first-50-token match vs the mlx-serve binary on the same checkpoint
- [ ] `tests/test_engine_gpu.c`: generate on the tiny fixture (deterministic), cancel mid-generate, shutdown mid-generate

### Stage C: sampling + generation params

**Issue:** [#54](https://github.com/tlkahn/mlxd/issues/54) · **Depends on:** Stage B

Honor `sampling_params_t` (temperature, top_p, top_k, min_p, seed) and logprobs on the GPU, mirroring `generate.zig`'s pipeline. Green + demoable: sampled generations respect params; same-seed runs are reproducible.

#### Phase C1: sampler pipeline

- [ ] Temperature scaling, top-k filter (`mlx_topk`/`mlx_argpartition_axis`), top-p filter (`mlx_argsort_axis` + `mlx_cumsum`), min-p filter, then softmax + `mlx_random_categorical`
- [ ] Seed handling: `seed >= 0` gives a deterministic per-request key (`mlx_random_key`); default remains nondeterministic
- [ ] GPU tests: greedy equals temp-0 limit; top-k=1 equals greedy; same-seed reproducibility; param edge cases (top_p=1, top_k=-1 disabled)

#### Phase C2: logprobs + defaults precedence

- [ ] Populate `chunk_t.logprob` with the sampled token's log-probability - **single logprob only** for Stage C; a `top_logprobs` list payload is a named follow-up, not in #51 (resolved deferral)
- [ ] Sampling defaults precedence: request > `generation_config.json` > `SAMPLING_PARAMS_DEFAULT`
- [ ] Multi-eos support: any id from the merged eos set (config.json union generation_config.json, see Stage A1 / divergences) terminates with `FINISH_STOP`

### Stage D: dense family breadth (llama, mistral, qwen2, gemma3, gemma4, qwen3_5)

**Issue:** [#55](https://github.com/tlkahn/mlxd/issues/55) · **Depends on:** Stage C

Extend the dense forward path with per-family quirks, config parsing, and weight mappings; one temp-0 parity gate per family on a small 4-bit checkpoint. Port quirks from `transformer.zig`'s config-driven branches. Green + demoable: all six families generate and pass their parity gates.

- [ ] llama: **llama3-style rope factor scheduling** using the full rope-scaling block from Stage A1 (`rope_type:"llama3"` / factor / low-high freq / original max position) - intentional superset of mlx-serve, which lacks this path; plus untied/tied head handling
- [ ] mistral: sliding-window attention
- [ ] qwen2: attention bias, tied embeddings on small variants
- [ ] gemma3: embedding scaling, extra pre/post feedforward norms, interleaved local/global sliding-window layer pattern with dual rope bases, family-specific norm offsets
- [ ] gemma4: port from the reference's gemma4/gemma4_unified handling
- [ ] qwen3_5 (dense text variant): config shape differences vs qwen3
- [ ] Per-family weight-name mapping tables + config fixtures + CPU mapping tests
- [ ] Parity gate script run per family; record checkpoint ids used in the script

### Stage E: MoE + hybrid families (deepseek_v3/v32/v4, qwen3_5_moe, lfm2, nemotron_h)

**Issue:** [#56](https://github.com/tlkahn/mlxd/issues/56) · **Depends on:** Stage D

The two non-dense forward paths from the reference: `forwardMoeWith` (expert dispatch via `mlx_gather_qmm`/`mlx_gather_mm`) and `forwardHybridWith` (conv/SSM layers with their own cache entries). Green + demoable: all remaining generate families pass parity gates.

#### E1: MoE infrastructure (shared by deepseek + qwen3_5_moe)

- [ ] MoE block (`fwd_moe`): router top-k + softmax, quantized expert dispatch via `mlx_gather_qmm` (contiguous-materialization gotcha test), bf16 path via `mlx_gather_mm`, shared experts
- [ ] DeepSeek-style MoE gate: sigmoid scoring (not softmax), group expert selection with `n_group`/`topk_group`, `e_score_correction_bias`, `routed_scaling_factor`, `norm_topk_prob`

#### E2: DeepSeek V3 / V3.2 / V4 (MLA + MoE; separate families)

> **Decision 5 superseded (2026-07-20):** mlx-lm implements DeepSeek V3/V3.2/V4 natively on safetensors - no GGUF or ds4 engine needed. Port from mlx-lm's `deepseek_v3.py` / `deepseek_v32.py` + `mla.py`, not from `ds4.zig`.
>
> **Decision 11 (2026-07-20):** do **not** track all DeepSeek variants as `MODEL_DEEPSEEK_V4`. Enum grain matches mlx-lm's module split and decision 4's "separate enum when paths fork" rule:
> - `deepseek_v3` -> `MODEL_DEEPSEEK_V3` (MLA + MoE, full-sequence attention)
> - `deepseek_v3_2` / `deepseek_v32` -> `MODEL_DEEPSEEK_V32` (MLA + MoE + DSA Indexer, dual cache)
> - `deepseek_v4` -> `MODEL_DEEPSEEK_V4` (own value; characterize Flash before any alias onto V3/V32)
>
> Shared primitives (not one family tag): `fwd_mla_*`, MultiLinear, MoE gate, weight sanitize. V3.2's indexer is a real attention/cache fork - not a boolean papered over `model_type` string checks inside a collapsed family. Correct A1's interim collapse before implementing forward.

- [ ] **Family split correction** (decision 11): replace interim `MODEL_DEEPSEEK_V4`-only mapping with `MODEL_DEEPSEEK_V3` / `MODEL_DEEPSEEK_V32` / `MODEL_DEEPSEEK_V4` in `model.h`, `model_family_from_type()`, config defaults, weight-load reject paths, and `tests/test_model_config.c`. Reject messages must name the actual family.
- [ ] **MLA attention** (`fwd_mla_attention`) for `MODEL_DEEPSEEK_V3` (and shared by V3.2/V4 as the base path): a new attention path, not a variant of `fwd_attention`. Key differences from standard GQA:
  - Q LoRA compression: `q_a_proj` (hidden -> `q_lora_rank`) -> `q_a_layernorm` (RMSNorm) -> `q_b_proj` (`q_lora_rank` -> `n_heads * q_head_dim`). First layer has no LoRA (direct `q_proj`)
  - Compressed KV: `kv_a_proj_with_mqa` (hidden -> `kv_lora_rank + qk_rope_head_dim`) -> split into `kv_latent` (normed) + `k_pe` (positional)
  - Split Q: `q_nope` (`qk_nope_head_dim`) + `q_pe` (`qk_rope_head_dim`); RoPE only on pe parts
  - Per-head `MultiLinear` transforms: `embed_q` (kv_latent -> nope-key space) and `unembed_out` (kv_latent -> v space), weight shape `[n_heads, out_dim, in_dim]`
  - Attention = SDPA(q_nope, k, v) with `pe_scores` as mask (positional component pre-computed as `q_pe @ k_pe^T`)
  - Decode-path optimization: `embed_q` applied to q_nope (not kv_latent), then unembed_out post-SDPA
  - Attention scale: `q_head_dim^(-0.5)`, optionally multiplied by `mscale_all_dim` correction from `rope_scaling`
- [ ] **MultiLinear** (`fwd_multi_linear`): per-head batched matmul, weight `[H, O, I]`; quantized variant via `mlx_quantized_matmul` with per-head scales/biases (port `mla.py`'s `MultiLinear`/`QuantizedMultiLinear`)
- [ ] **MLA KV cache**: stores compressed `kv_latent` (`[B, 1, T, kv_lora_rank]`) + `k_pe` (`[B, 1, T, qk_rope_head_dim]`), NOT full K/V. Substantially smaller per-token footprint than GQA
- [ ] **V3.2 DSA indexer** (`MODEL_DEEPSEEK_V32`, port `deepseek_v32.py` `Indexer`): separate top-k index heads (`index_n_heads` / `index_head_dim` / `index_topk`), index-side RoPE + K cache, sparse gather (`take_along_axis`) on decode and sparse mask on prefill. Dual cache entry (MLA latent cache + indexer K cache), matching mlx-lm `CacheList`. Config fields: `index_head_dim`, `index_n_heads`, `index_topk` plus indexer weight names (`wq_b`, `wk`, `weights_proj`, ...)
- [ ] **Weight sanitization** in `weights.c`: fp8 dequant (block-128 `weight_scale_inv`), `kv_b_proj` -> `embed_q` + `unembed_out` split (reshape `[n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank]`, split, requantize), expert stacking (`experts.N.gate/up/down_proj` -> `switch_mlp.gate/up/down_proj` stacked), MTP layer removal; V3.2 also keeps indexer tensors
- [ ] **Config fields** on `model_config_t` (add in this stage or backfill into A1 if convenient): `q_lora_rank`, `kv_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `v_head_dim`, `n_routed_experts`, `n_shared_experts`, `routed_scaling_factor`, `moe_layer_freq`, `first_k_dense_replace`, `n_group`, `topk_group`, `moe_intermediate_size`; `rope_scaling` as nested dict (type/factor/mscale_all_dim); V3.2 indexer fields above
- [ ] **Dense/MoE layer mixing**: first `first_k_dense_replace` layers use standard SwiGLU MLP; remaining layers use MoE (every `moe_layer_freq`-th layer)
- [ ] **V4 characterization**: inspect `mlx-community/DeepSeek-V4-Flash-4bit` `config.json` + weight names before implementing `MODEL_DEEPSEEK_V4` forward; only alias onto V3 or V32 if the architecture truly matches - never assume by product name
- [ ] Parity gates (skip-if-absent; oracle = mlx-lm, not mlx-serve/ds4): at least one V3-class checkpoint, one V3.2-class checkpoint if available, and DeepSeek-V4-Flash-4bit once characterized

#### E3: qwen3_5_moe, lfm2, nemotron_h

- [ ] qwen3_5_moe: linear-attention layers (separate QKV/Z/A/B projections per `LinearAttnWeights`) + MoE FFN
- [ ] Hybrid cache: SSM/conv state entries alongside KV entries (port `SSMCacheEntry`), prefill chunk boundaries respecting SSM checkpointing (`nextChunkEnd`)
- [ ] lfm2: short-conv blocks + attention hybrid
- [ ] nemotron_h: mamba-style SSM layers interleaved with attention/MLP
- [ ] Parity gates per family

### Stage F: lifecycle + memory hardening

**Issue:** [#57](https://github.com/tlkahn/mlxd/issues/57) · **Depends on:** Stage B (can proceed in parallel with D/E)

Real resource lifecycle and closing [#46](https://github.com/tlkahn/mlxd/issues/46). Green + demoable: `make test`, `make test-gpu`, `make test-tsan`, `make analyze` all clean; #46 closed.

- [ ] `CMD_UNLOAD`: free weights and caches on the engine thread; loading model B over model A frees A first
- [ ] `CMD_RECLAIM`: release per-request cache state
- [ ] Wired-memory limit at load (`mlx_set_wired_limit`, sized from the device recommendation) + cache limit; `mlx_clear_cache` every N decode steps and after each request; active/peak memory logging
- [ ] #46 verification: tests asserting engine release within the drain deadline when cancelled mid-prefill and mid-decode; close #46
- [ ] `make test-tsan` pass with the real engine; ASan/UBSan pass on GPU tests; `make analyze` clean on all new files
- [ ] Update CLAUDE.md: engine module layout (forward/kvcache/sampler files), stub mode, parity script usage

---

## Out of scope (deferred follow-ups)

Tracked separately; not required to close #51:

| Issue | Topic |
|---|---|
| [#58](https://github.com/tlkahn/mlxd/issues/58) | bert embeddings path (`CMD_EMBED` + `/v1/embeddings`) |
| [#59](https://github.com/tlkahn/mlxd/issues/59) | KV cache quantization (port `kv_quant.zig`) |
| [#60](https://github.com/tlkahn/mlxd/issues/60) | Prefix-cache reuse across turns (port `prefix_cache.zig`) |

Named deferrals resolved in the 2026-07-17 design review (deferred deliberately, not by accident):

| Item | Current state | Disposition |
|---|---|---|
| String `stop` sequences | Parsed in OpenAI DTOs, then zeroed before `engine_post` | Deferred; stop matching is detok/event-loop side, not engine token-id side |
| `n > 1` | In `gen_params_t`, unused | Deferred multi-completion |
| `top_logprobs` list | `chunk_t` carries a single logprob | Stage C = single logprob only; top-k list is a follow-up if needed |
| Repeat/presence penalty | In mlx-serve sampler, absent from `sampling_params_t` | Stays out unless the public API expands |
| Non-affine quant modes (`nvfp4` / `mxfp4` / `mxfp8`) | Hard-rejected at config parse (Stage A) | Named follow-up |
| Core AI (#50) | Separate spike | Orthogonal; do not shape mlxbridge around it |

Also deferred per `CLAUDE.md` v2 list: speculative decode, constrained/grammar decoding, continuous batching, multi-model residency, vision, DiffusionGemma, text-model LoRA, WebSocket/Responses/Ollama/Anthropic dialects.

---

## Verification

- `make test` stays green and GPU-free at every stage (stub mode covers actor/HTTP suites)
- `make test-gpu` covers loading, forward equivalence, decode, sampling, and lifecycle on the tiny fixture; real-checkpoint tests skip when absent
- Parity: temp-0 first-50-token match vs the mlx-serve Zig binary per family, scripted and repeatable
- Live check per stage: `mlxd run` (Stage B+) and `mlxd serve` + streaming chat completion (Stage B4+)

## Suggested module layout (lands across stages)

```
src/model/
  model.h / model.c          # model_config_t + family enum (A1)
  weights.h / weights.c      # safetensors shard load + name table (A3)
src/mlxbridge/
  mlxbridge.h / mlxbridge.c  # safetensors/eval/memory helpers (A2)
src/engine/
  engine.h / engine.c        # actor + CMD_LOAD/GENERATE wiring (B3, F)
  forward.h / forward.c      # dense/MoE/hybrid forward (B1, D, E)
  kvcache.h / kvcache.c      # dense KV (+ hybrid SSM entries in E)
  sampler.h / sampler.c      # greedy (B) + full pipeline (C)
tests/
  test_model_config.c        # extended fixtures (A1)
  test_mlxbridge_gpu.c       # new helper smokes (A2)
  test_weights_gpu.c         # fixture + skip-if-absent real load (A3)
  test_engine_gpu.c          # generate/cancel/shutdown (B4, F)
scripts/
  parity_temp0.sh            # first-50-token match vs mlx-serve (B4+)
tools/
  gen_tiny_ckpt.*             # tiny random-weight fixture generator (A3, family-parameterized)
```

Exact file splits are flexible; the Stage F CLAUDE.md update records the final layout.
