#!/usr/bin/env python3
"""Byte-exact compare for temp-0 text parity gate.

Strips exactly one trailing newline from the mlxd side (the CLI
presentation newline added by src/main.c), then byte-compares.
Exit 0 on match, exit 1 with MISMATCH + first-divergence offset on mismatch.
"""
import sys

def main():
    if len(sys.argv) != 3:
        print("usage: parity_compare.py <oracle_file> <mlxd_file>", file=sys.stderr)
        sys.exit(2)

    oracle = open(sys.argv[1], "rb").read()
    mlxd = open(sys.argv[2], "rb").read()

    if mlxd.endswith(b"\n"):
        mlxd = mlxd[:-1]

    if oracle == mlxd:
        sys.exit(0)

    for i, (a, b) in enumerate(zip(oracle, mlxd)):
        if a != b:
            print(f"MISMATCH at byte {i}: oracle {a!r} vs mlxd {b!r}")
            sys.exit(1)

    print(f"MISMATCH at byte {min(len(oracle), len(mlxd))}: length differs (oracle {len(oracle)} vs mlxd {len(mlxd)})")
    sys.exit(1)

if __name__ == "__main__":
    main()
