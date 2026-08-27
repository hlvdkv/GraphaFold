#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-outputs/CaspRNA2-graphafold-timed}"
timing_file="${2:-outputs/CaspRNA2-graphafold-timed.wall_time.txt}"

mkdir -p "$(dirname "$output_dir")" "$(dirname "$timing_file")"

PYTHONNOUSERSITE=1 /usr/bin/time -p -o "$timing_file" \
  python evaluate.py \
  --checkpoint runs/graphafold-a100-prior128/best.pt \
  --benchmark-dir CaspRNA2 \
  --output-dir "$output_dir" \
  --device cuda \
  --amp bf16

printf 'Czas zapisano w: %s\n' "$timing_file"
