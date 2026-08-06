#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CSV_ROOT="${REPO_ROOT}/data/synthetic-ids"
YOLO_ROOT="${REPO_ROOT}/data/yolo-ids"

usage() {
  echo "usage: $0 [-y] [train | val | all] [train | val | all ...]" >&2
  echo "  -y   skip the confirmation prompt" >&2
  exit 1
}

ASSUME_YES=0
if [ "${1:-}" = "-y" ]; then
  ASSUME_YES=1
  shift
fi

if [ "$#" -lt 1 ]; then
  usage
fi

# Validate every argument up front, before deleting anything.
for arg in "$@"; do
  case "$arg" in
    train|val|all) ;;
    *)
      echo "error: unrecognized split '${arg}' (expected train, val, or all)" >&2
      exit 1
      ;;
  esac
done


clean_dir_contents() {
  local dir="$1"
  local resolved
  resolved="$(cd "$dir" 2>/dev/null && pwd)" || return 0  

  case "$resolved" in
    "${REPO_ROOT}"/*) ;;
    *)
      echo "error: refusing to clean '${resolved}' — outside ${REPO_ROOT}" >&2
      exit 1
      ;;
  esac

  find "$resolved" -mindepth 1 -delete
}

clean_dir_contents_glob() {
  
  local dir="$1"
  local pattern="$2"
  local resolved
  resolved="$(cd "$dir" 2>/dev/null && pwd)" || return 0

  case "$resolved" in
    "${REPO_ROOT}"/*) ;;
    *)
      echo "error: refusing to clean '${resolved}' — outside ${REPO_ROOT}" >&2
      exit 1
      ;;
  esac

  find "$resolved" -maxdepth 1 -name "$pattern" -delete
}

clean_split() {
  local split="$1"
  echo "cleaning split: ${split}"

  clean_dir_contents        "${CSV_ROOT}/${split}/images"
  clean_dir_contents        "${CSV_ROOT}/${split}/line"
  clean_dir_contents_glob   "${CSV_ROOT}/${split}" "*.csv"

  clean_dir_contents        "${YOLO_ROOT}/images/${split}"
  clean_dir_contents        "${YOLO_ROOT}/labels/${split}"
}

# Resolve "all" -> both splits, de-duplicated, preserving explicit choices.
splits=()
for arg in "$@"; do
  if [ "$arg" = "all" ]; then
    splits+=(train val)
  else
    splits+=("$arg")
  fi
done
# de-dup while preserving order
declare -A seen=()
unique_splits=()
for s in "${splits[@]}"; do
  if [ -z "${seen[$s]:-}" ]; then
    unique_splits+=("$s")
    seen[$s]=1
  fi
done

echo "About to delete generated data for: ${unique_splits[*]}"
echo "  ${CSV_ROOT}/{split}/images, .../line, .../*.csv"
echo "  ${YOLO_ROOT}/images/{split}, ${YOLO_ROOT}/labels/{split}"

if [ "$ASSUME_YES" -ne 1 ]; then
  read -r -p "Proceed? [y/N] " confirm
  case "$confirm" in
    y|Y|yes|YES) ;;
    *) echo "aborted."; exit 0 ;;
  esac
fi

for split in "${unique_splits[@]}"; do
  clean_split "$split"
done

echo "done."