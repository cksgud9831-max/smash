#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Verifying package integrity..."
sha256sum -c MANIFEST.sha256
