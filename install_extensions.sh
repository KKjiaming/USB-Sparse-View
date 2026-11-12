#!/usr/bin/env bash
# Build the bundled CUDA extensions in the currently activated environment.
set -euo pipefail
CASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODULES_DIR="$CASE_DIR/third_party/fact-gs/fact_gs/submodules"
command -v nvcc >/dev/null || { echo 'CUDA toolkit (nvcc) is required; activate/configure CUDA first.' >&2; exit 1; }
export GLM_HOME="$MODULES_DIR/glm"
for module in TIGRE simple_knn fused_3d_tv fused_ssim gs_ct_rasterizer gs_voxelizer; do
    python -m pip install --no-build-isolation "$MODULES_DIR/$module"
done
