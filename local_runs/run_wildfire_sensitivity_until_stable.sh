#!/usr/bin/env bash
# Run grant-style sensitivity screening: re-test retained sensitive covariates
# until an entire round moves none into the insensitive set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SCREENING_MODE=until_stable
exec bash "${SCRIPT_DIR}/run_wildfire_sensitivity.sh"
