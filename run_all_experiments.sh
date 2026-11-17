#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

run() {
    local name="$1"
    shift
    echo "=== Running ${name} $* ==="
    python3 -u "scripts/${name}.py" "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

# exact coverage under the linear Gaussian model
run exact_coverage

# simulation study mimicking projections of heat-related mortality
run simulation_projection

# reanalysis of the heat-related mortality projections in London
run download_data
run reanalysis_london

# tables and figures of the paper
run make_tables
run create_figures
