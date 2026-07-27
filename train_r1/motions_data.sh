#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

source "${REPO_ROOT}/scripts/source_retargeting_setup.sh"
cd "${REPO_ROOT}/src/holosoma_retargeting/holosoma_retargeting"

DEBUG="${DEBUG:-0}"
VISUALIZE="${VISUALIZE:-0}"
RETRIES="${RETRIES:-5}"
FORCE="${FORCE:-1}"
CONTACT_TARGET_TOPK="${CONTACT_TARGET_TOPK:-5}"
FAILED_TASKS_FILE="${FAILED_TASKS_FILE:-${REPO_ROOT}/train_r1/failed_tasks.txt}"

case "${FORCE}" in
  -1|0|1) ;;
  *)
    echo "FORCE must be one of: 0 skip existing, 1 overwrite, -1 delete existing then regenerate" >&2
    exit 2
    ;;
esac

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
  local object_penetration_tolerance="$3"
  local save_dir="${REPO_ROOT}/train_r1/motions/${object_name}"
  local output_file="${save_dir}/${task_name}_original.npz"

  mkdir -p "${save_dir}"

  if [[ "${FORCE}" == "-1" && -f "${output_file}" ]]; then
    echo "Remove existing before regenerate: ${output_file}"
    rm -f "${output_file}"
  elif [[ "${FORCE}" != "1" && -f "${output_file}" ]]; then
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
    --retargeter.surface-penetration-tolerance 0.00
    --retargeter.object-penetration-tolerance "${object_penetration_tolerance}"
    --retargeter.contact-threshold 0.1
    --retargeter.contact-source robot
    --retargeter.contact-target-topk "${CONTACT_TARGET_TOPK}"
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

run_task sub1_suitcase_050 suitcase 0.01
# run_task sub1_suitcase_070 suitcase 0.01
run_task sub5_suitcase_000 suitcase 0.01

run_task sub17_smalltable_014 smalltable 0.01
run_task sub17_smalltable_010 smalltable 0.01
run_task sub6_smalltable_030 smalltable 0.01

run_task sub4_whitechair_029 whitechair 0.035
run_task sub4_whitechair_030 whitechair 0.01

run_task sub3_largebox_003 largebox 0.01
run_task sub3_largebox_020 largebox 0.01
run_task sub7_largebox_000 largebox 0.01

run_task sub5_plasticbox_030 plasticbox 0.01
run_task sub5_plasticbox_000 plasticbox 0.01
