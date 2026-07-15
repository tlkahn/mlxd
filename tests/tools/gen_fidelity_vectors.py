#!/usr/bin/env python3
"""Dev-time oracle for tests/test_tokenizer_fidelity.c (Stage H).

Generates known-good token-id vectors from the HuggingFace `tokenizers`
library for the bert and gemma4 fixtures, printed as ready-to-paste C
int32_t array literals plus expected decoded strings.

Run from the repo root:
    uv run --with tokenizers python tests/tools/gen_fidelity_vectors.py

Rules (see issue #24):
  - BERT encodes with add_special_tokens=True  (the C WordPiece path wraps
    [CLS]/[SEP] unconditionally).
  - Gemma encodes with add_special_tokens=False (the C SentencePiece path
    emits no <bos>).
  - Oracle output is ground truth: a C-side mismatch is a C bug; vectors
    are never edited to fit.
"""

import os
import sys

from tokenizers import Tokenizer

FIXTURES = os.path.join(os.path.dirname(__file__), "..", "fixtures")

BERT_INPUTS = [
    "the quick brown fox jumps over the lazy dog",
    "unaffable tokenization",
    "hello, world!",
    "qwertzuiopasd flimflam",
]

GEMMA_INPUTS = [
    "a  b   c",
    "byte fallback: ܏ end",  # U+070F SYRIAC ABBREVIATION MARK, absent from vocab
    "你好世界",  # CJK
    "The quick brown fox jumps over the lazy dog.",
]


def c_str(s: str) -> str:
    out = []
    for ch in s:
        b = ch.encode("utf-8")
        if ch in ('"', "\\"):
            out.append("\\" + ch)
        elif 0x20 <= ord(ch) < 0x7F:
            out.append(ch)
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        else:
            out.append("".join(f"\\x{x:02x}" for x in b) + '""')
    return '"' + "".join(out) + '"'


def emit(name: str, tok: Tokenizer, inputs, add_special: bool, decode_skip_special: bool):
    print(f"/* ---- {name} ---- */")
    for i, text in enumerate(inputs):
        enc = tok.encode(text, add_special_tokens=add_special)
        ids = enc.ids
        dec = tok.decode(ids, skip_special_tokens=decode_skip_special)
        print(f"/* input: {c_str(text)} */")
        arr = ", ".join(str(x) for x in ids)
        print(f"static const int32_t {name}_ids_{i}[] = {{{arr}}};")
        print(f"/* decoded: {c_str(dec)} */")
    print()


def main():
    bert = Tokenizer.from_file(os.path.join(FIXTURES, "bert", "tokenizer.json"))
    emit("bert", bert, BERT_INPUTS, add_special=True, decode_skip_special=True)

    gemma_path = os.path.join(FIXTURES, "gemma4", "tokenizer.json")
    if not os.path.exists(gemma_path):
        print("/* gemma4 fixture absent - skipped (see test header for setup) */")
        return
    gemma = Tokenizer.from_file(gemma_path)
    emit("gemma", gemma, GEMMA_INPUTS, add_special=False, decode_skip_special=False)


if __name__ == "__main__":
    sys.exit(main())
