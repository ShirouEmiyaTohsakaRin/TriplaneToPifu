#!/usr/bin/env bash
set -euo pipefail

# 默认使用当前环境的 python。
# 可以通过环境变量 PYTHON_BIN 或命令行 --python 指定。
PYTHON_BIN="${PYTHON_BIN:-python}"

CHECKPOINT_PATH=""
TEST_FOLDER_PATH=""
MANIFEST_PATH=""
COLOR_CHECKPOINT_PATH=""
RESULTS_PATH=""
NAME="triplane_pifu_eval"
GPU_SELECTION=""
VOL_RES="256"
MODEL_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash scripts/test.sh --checkpoint FILE --input DIR [OPTIONS] [-- MODEL_ARGS...]

Required:
  --checkpoint FILE   Trained hybrid checkpoint (new bundle or legacy state_dict)
  --input DIR         Single-view images and matching *_mask.png files
  --manifest FILE     Calibrated view groups; use instead of --input

Options:
  --color-checkpoint FILE   Optional PIFu RGB network checkpoint
  --output DIR        Output root (default: PROJECT_ROOT/results)
  --name NAME         Output subdirectory (default: triplane_pifu_eval)
  --python EXEC       Python executable (default: $PYTHON_BIN or python)
  --gpu ID|cpu        Select one physical GPU, or force CPU
                      Omit to preserve CUDA_VISIBLE_DEVICES
  --resolution N      Reconstruction grid resolution (default: 256)
  -h, --help          Show this help

Arguments after -- are passed unchanged to apps/eval.py.

Example:
  bash scripts/test.sh \
    --checkpoint ./checkpoints/experiment/netG_latest \
    --input ./test_images \
    --gpu 0 \
    -- --loadSize 512 --triplane_num_layers 4
EOF
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

require_value() {
    if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
        fail "Missing value for $1"
    fi
}

# ------------------------------------------------------------
# 1. 读取启动参数
# ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            require_value "$@"
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --input)
            require_value "$@"
            TEST_FOLDER_PATH="$2"
            shift 2
            ;;
        --manifest)
            require_value "$@"
            MANIFEST_PATH="$2"
            shift 2
            ;;
        --color-checkpoint)
            require_value "$@"
            COLOR_CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --output)
            require_value "$@"
            RESULTS_PATH="$2"
            shift 2
            ;;
        --name)
            require_value "$@"
            NAME="$2"
            shift 2
            ;;
        --python)
            require_value "$@"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --gpu)
            require_value "$@"
            GPU_SELECTION="$2"
            shift 2
            ;;
        --resolution)
            require_value "$@"
            VOL_RES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            MODEL_ARGS=("$@")
            break
            ;;
        *)
            fail "Unknown option: $1. Put model options after --."
            ;;
    esac
done

# ------------------------------------------------------------
# 2. 自动定位项目
# ------------------------------------------------------------

# 仓库目录不必叫 Tri，只需保留 scripts/test.sh 的目录结构。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EVAL_SCRIPT="${PROJECT_ROOT}/apps/eval.py"

# 不切换工作目录：
# 用户传入的相对路径始终相对于启动命令时所在的目录。
if [[ -z "${RESULTS_PATH}" ]]; then
    RESULTS_PATH="${PROJECT_ROOT}/results"
fi

# ------------------------------------------------------------
# 3. 检查解释器、输入路径与基本参数
# ------------------------------------------------------------
command -v -- "${PYTHON_BIN}" >/dev/null 2>&1 \
    || fail "Python not found: ${PYTHON_BIN}. Activate your environment or use --python."

[[ -f "${EVAL_SCRIPT}" ]] \
    || fail "Evaluation entry not found: ${EVAL_SCRIPT}"

[[ -n "${CHECKPOINT_PATH}" ]] \
    || fail "--checkpoint is required."

[[ -n "${TEST_FOLDER_PATH}" || -n "${MANIFEST_PATH}" ]] \
    || fail "--input or --manifest is required."
[[ -z "${TEST_FOLDER_PATH}" || -z "${MANIFEST_PATH}" ]] \
    || fail "Use either --input or --manifest."

[[ -f "${CHECKPOINT_PATH}" ]] \
    || fail "Checkpoint not found: ${CHECKPOINT_PATH}"

INPUT_ARGS=()
if [[ -n "${MANIFEST_PATH}" ]]; then
    [[ -f "${MANIFEST_PATH}" ]] || fail "Manifest not found: ${MANIFEST_PATH}"
    INPUT_ARGS+=(--eval_manifest "${MANIFEST_PATH}")
else
    [[ -d "${TEST_FOLDER_PATH}" ]] || fail "Input directory not found: ${TEST_FOLDER_PATH}"
    INPUT_ARGS+=(--test_folder_path "${TEST_FOLDER_PATH}")
fi
if [[ -n "${COLOR_CHECKPOINT_PATH}" ]]; then
    [[ -f "${COLOR_CHECKPOINT_PATH}" ]] || fail "Color checkpoint not found: ${COLOR_CHECKPOINT_PATH}"
    INPUT_ARGS+=(--load_netC_checkpoint_path "${COLOR_CHECKPOINT_PATH}")
fi

[[ "${VOL_RES}" =~ ^[1-9][0-9]*$ ]] \
    || fail "--resolution must be a positive integer."

(( VOL_RES >= 2 )) \
    || fail "--resolution must be at least 2."

# ------------------------------------------------------------
# 4. 设备选择
# ------------------------------------------------------------
if [[ "${GPU_SELECTION}" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""
elif [[ -n "${GPU_SELECTION}" ]]; then
    [[ "${GPU_SELECTION}" =~ ^[0-9]+$ ]] \
        || fail "--gpu must be a nonnegative integer or cpu."

    export CUDA_VISIBLE_DEVICES="${GPU_SELECTION}"
fi

# 不提供 --gpu 时，保留已有 CUDA_VISIBLE_DEVICES。
# eval.py 使用进程内的 cuda:0；CUDA 不可用时会使用 CPU。

# ------------------------------------------------------------
# 5. 执行推理
# ------------------------------------------------------------

# 深度中心、半径由新版 eval.py 从融合模型 checkpoint 恢复。
# 可选加载原 PIFu RGB 网络；几何始终使用三平面融合模型。
exec "${PYTHON_BIN}" -u "${EVAL_SCRIPT}" \
    --name "${NAME}" \
    --gpu_id 0 \
    --resolution "${VOL_RES}" \
    --results_path "${RESULTS_PATH}" \
    "${INPUT_ARGS[@]}" \
    --load_netG_checkpoint_path "${CHECKPOINT_PATH}" \
    "${MODEL_ARGS[@]}"
