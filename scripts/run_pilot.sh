#!/usr/bin/env bash
set -euo pipefail
python -m causal_utility.cli run --config configs/pilot.yaml --output results/pilot --figures figures/pilot
