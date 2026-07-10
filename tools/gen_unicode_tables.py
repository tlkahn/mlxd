#!/usr/bin/env python3
"""Generate src/model/tok_unicode_tables.h from Python's bundled Unicode
Character Database (stdlib unicodedata; no network, no UCD download).

Each table is a sorted list of inclusive {lo, hi} codepoint ranges for a
Unicode general-category class, consumed by binary search in tok_unicode.c.

Regenerate with: make unicode-tables
"""

import os
import sys
import unicodedata

# (array_name, general-category prefixes, human label)
TABLES = [
    ("uc_letter_ranges", ("Lu", "Ll", "Lt", "Lm", "Lo"), "\\p{L}"),
    ("uc_mark_ranges", ("Mn", "Mc", "Me"), "\\p{M}"),
    ("uc_number_ranges", ("Nd", "Nl", "No"), "\\p{N}"),
]

MAX_CP = 0x10FFFF


def collect_ranges(categories):
    """Coalesce codepoints whose category is in `categories` into sorted
    inclusive (lo, hi) ranges."""
    ranges = []
    lo = None
    for cp in range(MAX_CP + 1):
        if unicodedata.category(chr(cp)) in categories:
            if lo is None:
                lo = cp
        elif lo is not None:
            ranges.append((lo, cp - 1))
            lo = None
    if lo is not None:
        ranges.append((lo, MAX_CP))
    return ranges


def emit(out):
    out.write("/* GENERATED FILE - DO NOT EDIT.\n")
    out.write(" * Regenerate with `make unicode-tables` "
              "(tools/gen_unicode_tables.py).\n")
    out.write(" * Unicode Character Database version: %s "
              "(Python stdlib unicodedata).\n"
              % unicodedata.unidata_version)
    for name, cats, label in TABLES:
        ranges = collect_ranges(cats)
        n_cps = sum(hi - lo + 1 for lo, hi in ranges)
        out.write(" * %s: %s -> %d ranges, %d codepoints.\n"
                  % (label, "+".join(cats), len(ranges), n_cps))
    out.write(" */\n")
    out.write("#ifndef MLXD_TOK_UNICODE_TABLES_H\n")
    out.write("#define MLXD_TOK_UNICODE_TABLES_H\n\n")
    out.write("#include <stdint.h>\n\n")
    out.write("typedef struct {\n")
    out.write("    uint32_t lo;\n")
    out.write("    uint32_t hi;\n")
    out.write("} uc_range;\n")
    for name, cats, label in TABLES:
        ranges = collect_ranges(cats)
        out.write("\n/* %s (%s), inclusive ranges, sorted by lo. */\n"
                  % (label, "+".join(cats)))
        out.write("#define %s_COUNT %d\n" % (name.upper(), len(ranges)))
        out.write("static const uc_range %s[%s_COUNT] = {\n"
                  % (name, name.upper()))
        for i in range(0, len(ranges), 4):
            row = ", ".join("{0x%04X, 0x%04X}" % r for r in ranges[i:i + 4])
            out.write("    %s,\n" % row)
        out.write("};\n")
    out.write("\n#endif\n")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "src", "model", "tok_unicode_tables.h")
    with open(path, "w") as f:
        emit(f)
    print("wrote %s (Unicode %s)"
          % (os.path.relpath(path), unicodedata.unidata_version))


if __name__ == "__main__":
    sys.exit(main())
