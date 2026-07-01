#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

source "${REPO_ROOT}/scripts/source_retargeting_setup.sh"
cd "${REPO_ROOT}/src/holosoma_retargeting/holosoma_retargeting"

DEBUG="${DEBUG:-0}"
VISUALIZE="${VISUALIZE:-0}"
RETRIES="${RETRIES:-5}"
FAILED_TASKS_FILE="${FAILED_TASKS_FILE:-${REPO_ROOT}/train_r1/failed_tasks.txt}"

# Keep native solver libraries quieter and less crash-prone during long batches.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CVXPY_CANON_BACKEND="${CVXPY_CANON_BACKEND:-SCIPY}"

: > "${FAILED_TASKS_FILE}"

run_task() {
  local task_name="$1"
  local object_name="$2"
  local save_dir="${REPO_ROOT}/train_r1/motions/${object_name}"
  local output_file="${save_dir}/${task_name}_original.npz"

  mkdir -p "${save_dir}"

  if [[ -f "${output_file}" ]]; then
    echo "Skip existing: ${output_file}"
    return 0
  fi

  local args=(
    --data_path demo_data/OMOMO_new
    --task-type object_interaction_contact
    --task-name "$task_name"
    --data_format smplh
    --save_dir "${save_dir}"
    --task-config.object-name "$object_name"
    --robot r1
    --retargeter.penetration-tolerance 0.0
    --retargeter.surface-penetration-tolerance 0.0
    --retargeter.object-penetration-tolerance 0.02
    --retargeter.contact-threshold 0.01
    --retargeter.contact-source robot
  )

  if [[ "${DEBUG}" == "1" ]]; then
    args+=(--retargeter.debug)
  fi

  if [[ "${VISUALIZE}" == "1" ]]; then
    args+=(--retargeter.visualize)
  fi

  for attempt in $(seq 1 "${RETRIES}"); do
    echo "Running ${task_name} (${object_name}), attempt ${attempt}/${RETRIES}"
    if PYTHONFAULTHANDLER=1 python examples/robot_retarget.py "${args[@]}"; then
      if [[ -f "${output_file}" ]]; then
        return 0
      fi

      echo "Command succeeded but output is missing: ${output_file}" >&2
    fi

    echo "FAILED attempt ${attempt}/${RETRIES}: ${task_name} (${object_name})" >&2
  done

  echo "SKIP after retries: ${task_name} (${object_name})" >&2
  printf "%s %s\n" "${task_name}" "${object_name}" >> "${FAILED_TASKS_FILE}"
  return 0
}

run_task sub1_suitcase_050 suitcase
run_task sub1_suitcase_070 suitcase
run_task sub1_suitcase_001 suitcase

run_task sub17_smalltable_014 smalltable
run_task sub17_smalltable_010 smalltable
run_task sub6_smalltable_030 smalltable

run_task sub4_whitechair_029 whitechair
run_task sub4_whitechair_030 whitechair

run_task sub3_largebox_001 largebox
run_task sub3_largebox_020 largebox
run_task sub7_largebox_000 largebox

run_task sub5_plasticbox_030 plasticbox
run_task sub5_plasticbox_000 plasticbox
