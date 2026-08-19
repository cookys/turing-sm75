#!/usr/bin/env bash
# WSL2 + Windows NVIDIA driver: install COMPILER ONLY. Never install nvidia-driver*.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

have_nvcc() {
  [ -x /usr/local/cuda/bin/nvcc ] || [ -x /usr/lib/cuda/bin/nvcc ] || command -v nvcc >/dev/null 2>&1
}

# CUDA 13.1 + glibc 2.42+ (Ubuntu 26.04): rsqrt/rsqrtf exception-spec clash.
# Idempotent. Only the two host decls that collide with bits/mathcalls.h.
patch_rsqrt_noexcept() {
  local f patched=0
  for f in \
    /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h \
    /usr/lib/cuda/targets/x86_64-linux/include/crt/math_functions.h
  do
    [ -f "$f" ] || continue
    if grep -q 'rsqrt(double x) noexcept' "$f"; then
      echo "rsqrt noexcept already on $f"
      continue
    fi
    sudo cp --update=none "$f" "$f.bak-rsqrt"
    sudo sed -i \
      -e 's/double                 rsqrt(double x);/double                 rsqrt(double x) noexcept;/' \
      -e 's/float                  rsqrtf(float x);/float                  rsqrtf(float x) noexcept;/' \
      "$f"
    if grep -q 'rsqrt(double x) noexcept' "$f"; then
      echo "patched rsqrt noexcept: $f"
      patched=1
    else
      echo "failed to patch $f" >&2
      exit 1
    fi
  done
  if [ "$patched" -eq 0 ] && ! grep -q 'rsqrt(double x) noexcept' \
      /usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h 2>/dev/null; then
    echo "math_functions.h not found; nvcc headers missing?" >&2
    exit 1
  fi
}

# Toolkit only. Never pull nvidia-driver* / libnvidia-compute-*.
# nvcc compiles; cuobjdump/nvdisasm are required to claim HMMA vs FFMA.
PKGS=(cuda-nvcc-13-1 cuda-cuobjdump-13-1 cuda-nvdisasm-13-1 libcublas-13-1 libcublas-dev-13-1)
need_apt=0
for p in "${PKGS[@]}"; do
  dpkg -s "$p" >/dev/null 2>&1 || need_apt=1
done

if [ "$need_apt" -eq 1 ]; then
  echo "Installing toolkit packages (no driver): ${PKGS[*]}"
  if sudo apt-get update && sudo apt-get install -y --no-install-recommends "${PKGS[@]}"; then
    echo "apt install ok"
  else
    echo "apt failed. Add NVIDIA's CUDA apt repo for your Ubuntu, or install cuda-nvcc-13-1 another way. Do not apt install nvidia-driver*."
    exit 1
  fi
fi

if have_nvcc; then
  echo "nvcc: $(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
  (command -v nvcc >/dev/null && nvcc --version) || /usr/local/cuda/bin/nvcc --version
else
  echo "nvcc still missing after apt" >&2
  exit 1
fi

sudo mkdir -p /usr/local
if [ -d /usr/lib/cuda ] && [ ! -e /usr/local/cuda ]; then
  sudo ln -sfn /usr/lib/cuda /usr/local/cuda
fi

patch_rsqrt_noexcept

# login + interactive PATH
MARKER="# turing-sm75 wsl cuda"
if ! grep -q "$MARKER" "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<'BASH'

# turing-sm75 wsl cuda
export PATH="/usr/lib/wsl/lib:/usr/local/cuda/bin:/usr/lib/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
BASH
fi

echo "done. open a new shell or: source scripts/env.sh"
