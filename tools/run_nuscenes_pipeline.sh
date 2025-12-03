#!/usr/bin/env bash
set -euo pipefail

# 用法:
#   tools/run_nuscenes_pipeline.sh --scene scene-0103 --dataroot /data/sets/nuscenes [--version v1.0-mini]
#
# 说明:
# - 本脚本会输出到 output/<scene>/ 下
# - demo.py 运行后，若 saved_trajectories/result.txt 在仓库根目录，会被拷贝到对应 scene 目录
# - 对齐采用索引匹配（--by_index），与您当前流程一致

# 默认参数
VERSION="v1.0-mini"
SCENE=""
DATAROOT=""
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)    SCENE="$2"; shift 2 ;;
    --dataroot) DATAROOT="$2"; shift 2 ;;
    --version)  VERSION="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --scene scene-XXXX --dataroot /path/to/nuscenes [--version v1.0-mini]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "${SCENE}" || -z "${DATAROOT}" ]]; then
  echo "Error: --scene 和 --dataroot 必填"
  echo "Example: $0 --scene scene-0103 --dataroot /data/sets/nuscenes"
  exit 1
fi

OUTDIR="${REPO_ROOT}/output/${SCENE}"
IMAGES_DIR="${OUTDIR}/images"
CALIB_TXT="${OUTDIR}/camera_intrinsics.txt"
NUSC_TRAJ="${OUTDIR}/trajectory.txt"
EST_TXT_IN_SCENE="${OUTDIR}/saved_trajectories/result.txt"
ALIGNED_TXT="${OUTDIR}/dpvo_aligned.txt"
OVERLAY_DIR="${OUTDIR}/overlay_vis"

echo "=== [1/4] Extract nuScenes scene: ${SCENE} (${VERSION}) ==="
python "${REPO_ROOT}/tools/extract_nuscenes_scene.py" \
  --dataroot "${DATAROOT}" \
  --version  "${VERSION}" \
  --scene    "${SCENE}" \
  --output_dir "${OUTDIR}"

echo "=== [2/4] Run DPVO demo and save TUM trajectory ==="
python "${REPO_ROOT}/demo.py" \
  --imagedir="${IMAGES_DIR}" \
  --calib="${CALIB_TXT}" \
  --save_trajectory

# 将保存的 result.txt 放到对应场景目录（兼容你手动移动/已存在的情况）
mkdir -p "$(dirname "${EST_TXT_IN_SCENE}")"
if [[ -f "${REPO_ROOT}/saved_trajectories/result.txt" ]]; then
  cp -f "${REPO_ROOT}/saved_trajectories/result.txt" "${EST_TXT_IN_SCENE}"
  echo "Copied saved_trajectories/result.txt -> ${EST_TXT_IN_SCENE}"
elif [[ -f "${EST_TXT_IN_SCENE}" ]]; then
  echo "Found existing ${EST_TXT_IN_SCENE}"
else
  echo "Error: 未找到 result.txt，请确认 demo.py 已输出 saved_trajectories/result.txt"
  exit 1
fi

echo "=== [3/4] Align DPVO to nuScenes with Sim(3) (by index) ==="
python "${REPO_ROOT}/tools/align_dpvo_to_nuscenes.py" \
  --nusc "${NUSC_TRAJ}" --nusc_quat xyzw \
  --est  "${EST_TXT_IN_SCENE}" --est_quat xyzw \
  --out  "${ALIGNED_TXT}" \
  --by_index

echo "=== [4/4] Overlay visualization (3D + XY/XZ/YZ) ==="
python "${REPO_ROOT}/tools/visualize_nusc_and_est.py" \
  --nusc "${NUSC_TRAJ}" \
  --est  "${ALIGNED_TXT}" \
  --outdir "${OVERLAY_DIR}"

echo "✓ Done. Outputs in: ${OUTDIR}"
echo "  - Aligned traj: ${ALIGNED_TXT}"
echo "  - Overlay figs: ${OVERLAY_DIR}"

