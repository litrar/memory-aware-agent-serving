#!/usr/bin/env bash
# Download the Qwen3-4B Q8_0 GGUF weights into models/, where experiment.py
# expects them. Interrupted downloads resume; the result is checked against
# the SHA-256 published on Hugging Face.
#
#   scripts/download-model.sh
#
# Environment overrides:
#   MODEL_DIR       destination directory (default: models)
#   HF_REVISION     repo commit to download from (default: pinned below)
#   HF_TOKEN        sent as a bearer token if set (not needed for this public repo)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="Qwen/Qwen3-4B-GGUF"
FILE="Qwen3-4B-Q8_0.gguf"
REVISION="${HF_REVISION:-bc640142c66e1fdd12af0bd68f40445458f3869b}"
SHA256="8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300"
DIR="${MODEL_DIR:-$ROOT/models}"
DEST="$DIR/$FILE"
URL="https://huggingface.co/$REPO/resolve/$REVISION/$FILE"

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

mkdir -p "$DIR"

if [[ -f $DEST ]]; then
  log "Verifying existing $DEST"
  if [[ $(sha256 "$DEST") == "$SHA256" ]]; then
    echo "    already present and verified"
    exit 0
  fi
  die "$DEST exists but its SHA-256 does not match; remove it and rerun"
fi

# Download to .part so a half-finished file is never mistaken for the model.
log "Downloading $URL"
auth=()
if command -v wget >/dev/null; then
  [[ -n ${HF_TOKEN:-} ]] && auth=(--header "Authorization: Bearer $HF_TOKEN")
  wget -c ${auth[@]+"${auth[@]}"} -O "$DEST.part" "$URL"
elif command -v curl >/dev/null; then
  [[ -n ${HF_TOKEN:-} ]] && auth=(-H "Authorization: Bearer $HF_TOKEN")
  curl -fL -C - ${auth[@]+"${auth[@]}"} -o "$DEST.part" "$URL"
else
  die "need wget or curl"
fi

log "Verifying SHA-256"
actual="$(sha256 "$DEST.part")"
if [[ $actual != "$SHA256" ]]; then
  # Resuming a corrupt file would never fix it, so start over next time.
  rm -f "$DEST.part"
  die "SHA-256 mismatch (got $actual); partial file removed, rerun to retry"
fi

mv "$DEST.part" "$DEST"
echo "    saved: $DEST"
