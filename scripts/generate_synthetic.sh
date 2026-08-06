#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 [train|val] [num_samples] [num_augments_per_sample] [card_type] [format]" >&2
  echo "  format defaults to 'csv' (unchanged behavior). Pass 'yolo' for detection dataset export." >&2
  exit 1
fi

SPLIT="$1"
SIZE="$2"
AUGMENT_BATCHES="${3:-10}"
CARD_TYPE="${4:-full}"
FORMAT="${5:-csv}"

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

if [ "$FORMAT" = "yolo" ]; then
  OUT_ROOT="${REPO_ROOT}/data/yolo-ids"
  mkdir -p "$OUT_ROOT/images/$SPLIT" "$OUT_ROOT/labels/$SPLIT"

  EXTRA_ARGS=()
  [ -n "${BACKGROUND_DIR:-}" ] && EXTRA_ARGS+=(--background-dir "$BACKGROUND_DIR")
  [ -n "${MAX_ROTATION_DEG:-}" ] && EXTRA_ARGS+=(--max-rotation-deg "$MAX_ROTATION_DEG")
  [ -n "${CARD_SCALE_MIN:-}" ] && EXTRA_ARGS+=(--card-scale-min "$CARD_SCALE_MIN")
  [ -n "${CARD_SCALE_MAX:-}" ] && EXTRA_ARGS+=(--card-scale-max "$CARD_SCALE_MAX")
  [ "${USE_BACKGROUNDS:-1}" = "0" ] && EXTRA_ARGS+=(--no-backgrounds)
  [ "${CAPTURE_NOISE:-1}" = "0" ] && EXTRA_ARGS+=(--no-capture-noise)

  "$PYTHON_EXEC" "$GENERATOR_ROOT/EGID.py" "$SPLIT" "$SIZE" \
    --format yolo \
    --use-backgrounds \
    --out-root "$OUT_ROOT" \
    --no-clean \
    --resources-dir "$GENERATOR_ROOT" \
    "${EXTRA_ARGS[@]}"
elif [ "$FORMAT" = "csv" ]; then
  OUT_ROOT="${REPO_ROOT}/data/synthetic-ids"
  mkdir -p "$OUT_ROOT/$SPLIT/images"

  "$PYTHON_EXEC" "$GENERATOR_ROOT/EGID.py" "$SPLIT" "$SIZE" \
    --mode both \
    --augment-batches "$AUGMENT_BATCHES" \
    --use-backgrounds \
    --card-type "$CARD_TYPE" \
    --no-clean \
    --out-root "$OUT_ROOT" \
    --resources-dir "$GENERATOR_ROOT" \
    --format csv
fi