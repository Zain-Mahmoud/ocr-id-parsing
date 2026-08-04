#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 [train|val] [num_samples] [num_augments_per_sample]" >&2
  exit 1
fi

SPLIT="$1"
SIZE="$2"
AUGMENT_BATCHES="${3:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_GENERATION_ROOT="${DATA_GENERATION_ROOT:-$SCRIPT_DIR/../../data_generation}"

if [ ! -f "$DATA_GENERATION_ROOT/EGID.py" ]; then
  if [ -f "$SCRIPT_DIR/src/data_generation/EGID.py" ]; then
    DATA_GENERATION_ROOT="$SCRIPT_DIR/src/data_generation"
  else
    echo "Could not find data_generation/EGID.py in $DATA_GENERATION_ROOT or $SCRIPT_DIR/src/data_generation" >&2
    exit 1
  fi
fi

PYTHON_EXEC="$DATA_GENERATION_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON_EXEC" ]; then
  PYTHON_EXEC="$(command -v python3 || command -v python)"
fi

"$PYTHON_EXEC" "$DATA_GENERATION_ROOT/EGID.py" "$SPLIT" "$SIZE" --mode both --augment-batches "$AUGMENT_BATCHES" --out-root "$SCRIPT_DIR/../data/synthetic-ids" --resources-dir "$DATA_GENERATION_ROOT"
