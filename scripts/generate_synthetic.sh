#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 [train|val] [num_samples] [num_augments_per_sample] [card_type]" >&2
    exit 1
fi

SPLIT="$1"
SIZE="$2"
AUGMENT_BATCHES="${3:-10}"
CARD_TYPE="${4:-full}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SIBLING_DATAGENERATION="${REPO_ROOT}/../data_generation"

if [ -n "${DATA_GENERATION_ROOT:-}" ] && [ -d "$DATA_GENERATION_ROOT" ]; then
    GENERATOR_ROOT="$DATA_GENERATION_ROOT"
elif [ -d "$SIBLING_DATAGENERATION" ]; then
    GENERATOR_ROOT="$SIBLING_DATAGENERATION"
else
    echo "Unable to find data_generation root. Set DATA_GENERATION_ROOT to the external repo or add a local datagen/ directory." >&2
    exit 1
fi

PYTHON_EXEC="${GENERATOR_ROOT}/.venv/bin/python"
if [ ! -x "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="$(command -v python3 || command -v python)"
fi

export NO_ALBUMENTATIONS_UPDATE=1
OUT_ROOT="${REPO_ROOT}/data/synthetic-ids"

mkdir -p "$OUT_ROOT/$SPLIT/images"

"$PYTHON_EXEC" "$GENERATOR_ROOT/EGID.py" "$SPLIT" "$SIZE" --mode both --augment-batches "$AUGMENT_BATCHES" --card-type "$CARD_TYPE" --out-root "$OUT_ROOT" --resources-dir "$GENERATOR_ROOT"
