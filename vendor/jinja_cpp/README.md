# jinja_cpp

Vendored Jinja2 template engine for chat-template rendering, built from source by `build.zig` and linked into the `chat` module.

## Provenance

- The engine (`caps`, `lexer`, `parser`, `runtime`, `value`, `jinja_string`, `utils.h`) is llama.cpp's in-tree Jinja engine at `common/jinja/`, introduced in [ggml-org/llama.cpp#18462](https://github.com/ggml-org/llama.cpp/pull/18462), the PR that replaced minja (merged January 2026). License: MIT, The ggml authors - see `LICENSE` in this directory.
- Local renames relative to upstream: `string.{h,cpp}` is renamed to `jinja_string.{h,cpp}`; upstream's `log.h` dependency is replaced by a minimal stub `log.h`.
- `nlohmann/json.hpp` is [nlohmann/json](https://github.com/nlohmann/json) (MIT, license text embedded in the header itself).
- `jinja_wrapper.{h,cpp}` is the C wrapper originating from mlx-serve, exposing the 3-function C API (`jinja_render_chat`, `jinja_str_free`, `jinja_last_error`) consumed by `src/chat/root.zig`.

To sync upstream fixes, diff against `common/jinja/` in llama.cpp master, accounting for the renames above.

The minja alternative (the engine llama.cpp used before #18462) was evaluated and rejected (2026-07-05) in favor of this engine.

## Vendoring recipe

Copied from mlx-serve `lib/jinja_cpp/`: the 7 translation units (`jinja_wrapper, caps, lexer, parser, runtime, jinja_string, value`) with their headers, plus `utils.h`, `log.h`, and `nlohmann/json.hpp`.

Excluded: prebuilt artifacts (`libjinja.a`, `obj/`) - this tree is compiled from source by `build.zig` (`-std=c++17 -O2 -DNDEBUG`, Zig's bundled clang++/libc++). Also excluded: mlx-serve's `LICENSE` file, which mislabeled the engine as Apache-2.0 "Wang Zhaode" (a leftover from an earlier engine); the correct license is llama.cpp's MIT, shipped here as `LICENSE`.
