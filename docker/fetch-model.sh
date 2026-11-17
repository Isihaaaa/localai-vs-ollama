#!/bin/sh
# Downloads the Core model file into the shared volume (once) and records its
# sha256 in ./.state so the harness can put it into result files.
set -eu

target="/models/${MODEL_FILE}"

if [ ! -f "$target" ]; then
    echo "Downloading ${MODEL_FILE} (resumable)..."
    curl -fL --retry 5 --retry-delay 5 -C - -o "${target}.part" "$MODEL_URL"
    mv "${target}.part" "$target"
    rm -f "${target}.sha256"
fi

if [ ! -f "${target}.sha256" ]; then
    echo "Computing sha256 of ${MODEL_FILE}..."
    sha256sum "$target" | cut -d' ' -f1 > "${target}.sha256"
fi

actual="$(cat "${target}.sha256")"
if [ -n "${MODEL_SHA256:-}" ] && [ "$actual" != "$MODEL_SHA256" ]; then
    echo "Checksum mismatch for ${MODEL_FILE}: expected ${MODEL_SHA256}, got ${actual}" >&2
    exit 1
fi

mkdir -p /state
cp "${target}.sha256" "/state/${MODEL_FILE}.sha256"
echo "${MODEL_FILE} ready, sha256 ${actual}"
