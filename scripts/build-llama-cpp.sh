#!/usr/bin/env bash
# Clone (if needed), check out the pinned commit and build llama-server into
# third_party/llama.cpp/build/bin, where experiment.py expects it.
#
#   scripts/build-llama-cpp.sh                 # pinned commit, CUDA auto-detected
#   scripts/build-llama-cpp.sh --install-deps  # apt-get the toolchain first (Ubuntu/EC2)
#   scripts/build-llama-cpp.sh --cpu           # force a CPU-only build
#   scripts/build-llama-cpp.sh --clean         # wipe build/ before configuring
#
# Environment overrides:
#   LLAMA_CPP_REF   commit or tag to build (default: the calibration commit in README.md)
#   LLAMA_CPP_REPO  clone URL (default: upstream GitHub)
#   LLAMA_CPP_DIR   source directory (default: third_party/llama.cpp)
#   CUDA_ARCH       e.g. "75" for a T4; default: detected via nvidia-smi
#   JOBS            parallel build jobs (default: nproc)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="${LLAMA_CPP_REF:-797de63b2fa7396dcc8caca76834c699ac84e1c3}"  # b11716
REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
SRC="${LLAMA_CPP_DIR:-$ROOT/third_party/llama.cpp}"
BUILD="$SRC/build"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"

cuda=auto
install_deps=0
clean=0
for arg in "$@"; do
  case "$arg" in
    --cpu) cuda=off ;;
    --cuda) cuda=on ;;
    --install-deps) install_deps=1 ;;
    --clean) clean=1 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- toolchain

if (( install_deps )); then
  command -v apt-get >/dev/null || die "--install-deps only supports apt-based systems"
  log "Installing build dependencies"
  sudo apt-get update
  sudo apt-get install -y build-essential cmake git libssl-dev
fi

for tool in git cmake c++; do
  command -v "$tool" >/dev/null || die "$tool not found (try --install-deps)"
done

# The Deep Learning AMI ships CUDA under /usr/local/cuda without adding it to PATH.
if ! command -v nvcc >/dev/null && [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi

if [[ $cuda == auto ]]; then
  if command -v nvcc >/dev/null; then
    cuda=on
  else
    command -v nvidia-smi >/dev/null && echo "warning: GPU found but nvcc missing; building CPU-only" >&2
    cuda=off
  fi
fi
[[ $cuda == on ]] && ! command -v nvcc >/dev/null && die "--cuda given but nvcc not found"

# ---------------------------------------------------------------- source

if [[ ! -d $SRC/.git ]]; then
  [[ -e $SRC ]] && die "$SRC exists but is not a git checkout"
  log "Cloning $REPO into $SRC"
  mkdir -p "$(dirname "$SRC")"
  git clone "$REPO" "$SRC"
fi

if ! git -C "$SRC" diff --quiet HEAD; then
  die "$SRC has uncommitted changes; commit or stash them first"
fi

if ! git -C "$SRC" cat-file -e "$REF^{commit}" 2>/dev/null; then
  log "Fetching $REF"
  git -C "$SRC" fetch --tags origin
fi

log "Checking out $REF"
git -C "$SRC" checkout --quiet --detach "$REF"
git -C "$SRC" log -1 --format='    %h %ci %s'

# ---------------------------------------------------------------- build

cmake_args=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF)
if [[ $cuda == on ]]; then
  if [[ -z ${CUDA_ARCH:-} ]] && command -v nvidia-smi >/dev/null; then
    # "7.5" -> "75"; one entry per distinct GPU model. Building only the local
    # arch is far faster than llama.cpp's default multi-arch build.
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
      | tr -d ' .' | sort -u | paste -sd ';' || true)"
  fi
  cmake_args+=(-DGGML_CUDA=ON)
  [[ -n ${CUDA_ARCH:-} ]] && cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH")
  log "Building with CUDA (arch: ${CUDA_ARCH:-llama.cpp default})"
else
  log "Building CPU-only (Metal stays on by default on macOS)"
fi

if (( clean )); then
  rm -rf "$BUILD"
fi

cmake -S "$SRC" -B "$BUILD" "${cmake_args[@]}"
cmake --build "$BUILD" --config Release --target llama-server -j "$JOBS"

log "Done"
"$BUILD/bin/llama-server" --version 2>&1 | head -n 5 || true
echo "    binary: $BUILD/bin/llama-server"
