#!/usr/bin/env bash
set -e
 
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build git pkg-config curl ca-certificates \
  openmpi-bin libopenmpi-dev \
  libeigen3-dev libmetis-dev metis libparmetis-dev parmetis

# CUDA environment
CUDA_DIR="/usr/local/cuda-12.4"
if [ -d "$CUDA_DIR" ]; then
  if ! grep -q "cuda-12.4" "$HOME/.bashrc"; then
    {
      echo ""
      echo "# CUDA 12.4 environment"
      echo "export CUDA_HOME=$CUDA_DIR"
      echo 'export PATH=${CUDA_HOME}/bin:${PATH}'
      echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}'
    } >> "$HOME/.bashrc"
  fi
  export CUDA_HOME="$CUDA_DIR"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
fi

# Versions
cmake --version | head -n1 || true
nvcc --version 2>/dev/null | head -n3 || echo "nvcc not found (CUDA not on PATH yet)"
mpicxx --version | head -n1 || true
