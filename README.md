# mlxd

Fast modular text-generation server for Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx).

A ground-up C11 rewrite of the inference core from [mlxd-zig](https://github.com/tlkahn/mlxd-zig), delivering an OpenAI-compatible API with Unix-philosophy modularity.

## Features

- OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`)
- Server-Sent Events (SSE) streaming
- Single-binary with subcommands: `serve`, `run`, `pull`, `list`
- Single dedicated engine thread - all GPU work isolated from the event loop
- BPE, WordPiece, and SentencePiece tokenizers (GPT-2, BERT, Gemma-style)
- HuggingFace model download and local cache management

## Supported models

Gemma 3/4, Qwen 2/3/3.5, LLaMA, Mistral, LFM-2, Nemotron-H, BERT, DeepSeek-V4.

## Requirements

- macOS on Apple Silicon
- Homebrew dependencies:
  - [mlx-c](https://github.com/ml-explore/mlx-c) >= 0.6.0 (and mlx >= 0.31.2)
  - libuv
  - llhttp
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

Benchmark on Apple M4 Max (128 GB), Qwen3-0.6B-MLX (4-bit), 256 prompt tokens / 128 generated tokens:

| Server | Prompt (tok/s) | Generation (tok/s) | Time to first token (ms) | Peak RSS (MB) |
|--------|---------------:|--------------------|--------------------------|---------------|
| mlxd | TBD | TBD | TBD | TBD |
| [ollama](https://github.com/ollama/ollama) | TBD | TBD | TBD | TBD |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | TBD | TBD | TBD | TBD |
| [mlx-serve](https://github.com/ddalcucu/mlx-serve) | TBD | TBD | TBD | TBD |

Methodology: each server warmed with one request, then measured over 10 runs. See `tools/bench.sh` for reproduction.

> Numbers will be filled once the engine module is complete.

## Project status

This project is an active rewrite from Zig to C11 ([tracking epic](https://github.com/tlkahn/mlxd/issues/9)). Current progress:

| Module | Status |
|--------|--------|
| core | Done ([#1](https://github.com/tlkahn/mlxd/issues/1)) |
| mlxbridge | Done ([#2](https://github.com/tlkahn/mlxd/issues/2)) |
| chat | Done ([#3](https://github.com/tlkahn/mlxd/issues/3)) |
| model/tokenizer | In progress ([#4](https://github.com/tlkahn/mlxd/issues/4)) - stages A-D complete, E-I remaining |
| engine | Not started ([#5](https://github.com/tlkahn/mlxd/issues/5)) |
| registry | Not started ([#6](https://github.com/tlkahn/mlxd/issues/6)) |
| http | Not started ([#7](https://github.com/tlkahn/mlxd/issues/7)) |
| cli | Not started ([#8](https://github.com/tlkahn/mlxd/issues/8)) |

## License

MIT
