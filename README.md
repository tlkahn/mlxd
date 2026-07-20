# mlxd

Fast modular text-generation server for Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx).

A ground-up C11 rewrite of the inference core from [mlxd-zig](https://github.com/tlkahn/mlxd-zig), delivering an OpenAI-compatible API with Unix-philosophy modularity.

## Features

- OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`)
- Server-Sent Events (SSE) streaming
- Single-binary with subcommands: `serve`, `run`, `pull`, `list`
- Single dedicated engine thread - all GPU work isolated from the event loop
- Real mlx-c inference: forward pass, KV cache, top-k/p sampling, temperature, repetition penalty
- Thinking mode toggle (`enable_thinking` / `--no-think`) for reasoning models
- Sliding-window attention (Mistral-style)
- BPE, WordPiece, and SentencePiece tokenizers with NFC/NFKC Unicode normalization
- Jinja2 chat template rendering (vendored jinja_cpp)
- HuggingFace model download and local cache management

## Supported models

Dense: Gemma 3/4, Qwen 2/3/3.5, LLaMA, Mistral, LFM-2, Nemotron-H, BERT.
MoE (in progress): DeepSeek-V3, DeepSeek-V3.2, DeepSeek-V4.

## Requirements

- macOS on Apple Silicon
- Homebrew dependencies:
  - [mlx-c](https://github.com/ml-explore/mlx-c) >= 0.6.0 (and mlx >= 0.31.2)
  - libuv
  - llhttp
  - libcurl
- Clang with C11 support (ships with Xcode Command Line Tools)

## Build

```bash
make              # debug-friendly build (no optimization)
make release      # -O2 -DNDEBUG
make debug        # -g -O0 -fsanitize=address,undefined
make tsan         # -g -O1 -fsanitize=thread
make test         # all CPU-only tests
make analyze      # Clang Static Analyzer via scan-build
make coverage     # source-based coverage -> build/cov/html/index.html
make install      # install to /usr/local/bin/mlxd
```

## Usage

```bash
# Start the server
mlxd serve --model ~/.cache/mlxd/Qwen3-0.6B-MLX

# One-shot generation
mlxd run ~/.cache/mlxd/Qwen3-0.6B-MLX "Explain quicksort in one paragraph"

# One-shot generation with thinking disabled
mlxd run ~/.cache/mlxd/Qwen3-0.6B-MLX --no-think "Explain quicksort"

# Download a model from HuggingFace
mlxd pull mlx-community/Qwen3-0.6B-MLX

# List locally cached models
mlxd list
```

Once the server is running, use any OpenAI-compatible client:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B-MLX",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

The chat completions endpoint accepts an optional `"enable_thinking": false` field (non-standard extension) to suppress thinking for models whose chat template checks the `enable_thinking` variable. Templates default thinking on when the variable is undefined, so there is no `--think` flag or `"enable_thinking": true` needed to enable it.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-0.6B-MLX",
    "messages": [{"role": "user", "content": "Hello!"}],
    "enable_thinking": false
  }'
```

## Architecture

```
cli (main.c)
 |
 +-- http        libuv event loop, llhttp parser, SSE writer
 |    |
 |    +-- engine     inference thread, MPSC mailbox, stream chunks
 |    |    |
 |    |    +-- model      tokenizer, config loading
 |    |    |    |
 |    |    |    +-- core       types, logging, sysinfo, OpenAI DTOs
 |    |    |    +-- mlxbridge  mlx-c FFI, Metal frameworks
 |    |    |
 |    |    +-- mlxbridge
 |    |
 |    +-- chat       Jinja template rendering (pure, no mlx)
 |    +-- registry   HuggingFace download/resolve
 |
 +-- (all modules)
```

Key design decisions:

- **Single engine thread** - one dedicated thread is the sole mlx-c caller. No GPU calls from the event loop or callbacks.
- **Engine emits token IDs, not text** - detokenization runs as a stream transform on the event-loop side.
- **Chat module is pure** - no tokenizer, no architecture awareness, no mlx includes.
- **JSON via yyjson only** - no manual string construction or escape helpers.
- **No vtable dispatch** - concrete engine struct; HTTP layer takes an engine pointer directly.

## Performance

Benchmarks pending ([#76](https://github.com/tlkahn/mlxd/issues/76)). The engine is functional for dense families; numbers will be published after Stage F (lifecycle hardening) is complete.

## Project status

The Zig-to-C11 migration is complete ([epic #9](https://github.com/tlkahn/mlxd/issues/9) - all 8 modules done). The server is functional end-to-end with real mlx inference for dense model families.

Current focus is the engine inference core ([#51](https://github.com/tlkahn/mlxd/issues/51)):

| Stage | Scope | Status |
|-------|-------|--------|
| A | Loading foundations (safetensors, weight loader) | Done ([#52](https://github.com/tlkahn/mlxd/issues/52)) |
| B | Qwen3 dense vertical slice (forward, KV cache, greedy decode) | Done ([#53](https://github.com/tlkahn/mlxd/issues/53)) |
| C | Sampling + generation params (top-k/p, temperature, repetition penalty) | Done ([#54](https://github.com/tlkahn/mlxd/issues/54)) |
| D | Dense family breadth (LLaMA, Mistral, Qwen2, Gemma 3/4, Qwen3.5) | Done ([#55](https://github.com/tlkahn/mlxd/issues/55)) |
| E | MoE + hybrid families (DeepSeek V3/V3.2/V4, Qwen3.5-MoE, LFM-2, Nemotron-H) | In progress ([#56](https://github.com/tlkahn/mlxd/issues/56)) |
| F | Lifecycle + memory hardening | Not started ([#57](https://github.com/tlkahn/mlxd/issues/57)) |

## License

MIT
