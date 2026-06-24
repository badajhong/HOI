#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

source "${REPO_ROOT}/scripts/source_retargeting_setup.sh"
cd "${REPO_ROOT}/src/holosoma_retargeting/holosoma_retargeting"

MOTIONS_DIR="${MOTIONS_DIR:-${REPO_ROOT}/train_r1/motions}"
RL_DIR="${RL_DIR:-${REPO_ROOT}/train_r1/rl}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
DATA_FORMAT="${DATA_FORMAT:-smplh}"
ROBOT="${ROBOT:-r1}"
RETRIES="${RETRIES:-5}"
FAILED_TASKS_FILE="${FAILED_TASKS_FILE:-${REPO_ROOT}/train_r1/failed_rl_tasks.txt}"

# Keep native MuJoCo / BLAS libraries quieter during long conversion batches.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

shopt -s nullglob
: > "${FAILED_TASKS_FILE}"

found=0
for object_dir in "${MOTIONS_DIR}"/*; do
  [[ -d "${object_dir}" ]] || continue

  object_name="$(basename "${object_dir}")"
  output_dir="${RL_DIR}/${object_name}"
  mkdir -p "${output_dir}"

  for input_file in "${object_dir}"/*.npz; do
    found=1
    file_name="$(basename "${input_file}" .npz)"
    output_name="${file_name%_original}"
    output_file="${output_dir}/${output_name}.npz"

    if [[ -f "${output_file}" ]]; then
      echo "[rl_data] Skip existing: ${output_file}"
      continue
    fi

    echo "[rl_data] ${object_name}: ${file_name}.npz -> ${output_file}"
    for attempt in $(seq 1 "${RETRIES}"); do
      echo "[rl_data] Running ${file_name} (${object_name}), attempt ${attempt}/${RETRIES}"
      if PYTHONFAULTHANDLER=1 python data_conversion/convert_data_format_mj.py \
        --input_file "${input_file}" \
        --robot "${ROBOT}" \
        --output_fps "${OUTPUT_FPS}" \
        --output_name "${output_file}" \
        --data_format "${DATA_FORMAT}" \
        --object_name "${object_name}" \
        --has_dynamic_object \
        --once; then
        if [[ -f "${output_file}" ]]; then
          break
        fi

        echo "[rl_data] Command succeeded but output is missing: ${output_file}" >&2
      fi

      echo "[rl_data] FAILED attempt ${attempt}/${RETRIES}: ${file_name} (${object_name})" >&2
    done

    if [[ ! -f "${output_file}" ]]; then
      echo "[rl_data] SKIP after retries: ${file_name} (${object_name})" >&2
      printf "%s %s %s\n" "${input_file}" "${object_name}" "${output_file}" >> "${FAILED_TASKS_FILE}"
    fi
  done
done

if [[ "${found}" == "0" ]]; then
  echo "[rl_data] No .npz files found under ${MOTIONS_DIR}" >&2
  exit 1
fi

echo "[rl_data] Done. RL motion files saved under ${RL_DIR}"
