#!/usr/bin/env bash
set -euo pipefail
python -m causal_utility.cli run --config configs/smoke.yaml --output results/smoke --figures figures/smoke
