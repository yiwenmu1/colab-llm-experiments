#!/usr/bin/env bash
# Run from Mac. Existing llm runtime + mounted Drive required.
# --release explicitly opts into stopping this whole Colab session after task completion/failure.
set -euo pipefail
release=false
if [[ "${1:-}" == "--release" ]]; then release=true; shift; fi
minutes="${1:-60}"
case "$minutes" in ''|*[!0-9]*) echo 'Minutes must be an integer' >&2; exit 2;; esac
if (( minutes < 1 )); then exit 2; fi
# Failure to flush means retain runtime for recovery; do not mask release errors.
cleanup() {
  result=$?
  if $release; then
    if ssh -o BatchMode=yes colab-gpu "sync"; then
      colab exec -s llm -f "$(dirname "$0")/flush_drive.py" && colab stop -s llm || {
        echo 'Release incomplete: inspect Drive and Colab session manually.' >&2
        exit 1
      }
    else
      echo 'SSH unavailable; retained runtime for manual recovery.' >&2
      exit 1
    fi
  fi
  exit "$result"
}
trap cleanup EXIT
ssh -o BatchMode=yes colab-gpu "cd /content/llm-project && python -u reproduction/automation/job.py --state-dir /content/drive/MyDrive/colab-llm-experiments/jobs/review_v1 --timeout-minutes $minutes -- python -u reproduction/automation/evaluate.py --out /content/drive/MyDrive/colab-llm-experiments/public_subset_eval --run-name qwen3_8b_nf4_ctx2048_v1 --limit 10"
