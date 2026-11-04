#!/bin/bash
# Fetch only p1_train.npy (ARCTIC protocol p1 = allocentric views) out of ARCTIC's splits.zip with
# HTTP range requests, instead of downloading the whole archive. Needs an ARCTIC account.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORK="${ARCTIC_ALLO_WORK:-data/arctic_allo_work}"
: "${ARCTIC_USERNAME:?export your ARCTIC account username}" "${ARCTIC_PASSWORD:?export your ARCTIC account password}"
SPLITS_URL="https://download.is.tue.mpg.de/download.php?domain=arctic&resume=1&sfile=arctic_release/c7216c3b205186106a1f8326ed7b948f838e4907e69b21c8b3c87bb69d87206e/v1_0/data/splits.zip"
mkdir -p "$WORK"
exec "${PYTHON:-python}" "$HERE/partial_zip_backoff.py" "$SPLITS_URL" "p1_train.npy" "$WORK/p1_train.npy"
