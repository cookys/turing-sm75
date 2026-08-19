# WSL2: Windows NVIDIA driver lives here. Source before nvcc / llama-bench.
#   source scripts/env.sh
export PATH="/usr/lib/wsl/lib:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -d /usr/lib/cuda/bin ]; then
  export PATH="/usr/lib/cuda/bin:${PATH}"
fi
