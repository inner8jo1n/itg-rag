#!/usr/bin/env bash
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT
FAILED=0

run_step() {
    local name="$1"
    shift
    if "$@" > "$LOG" 2>&1; then
        echo -e "${GREEN}✔${NC} ${name}"
    else
        echo -e "${RED}✘${NC} ${name}"
        echo
        cat "$LOG"
        echo
        FAILED=1
    fi
}

run_step "Lint (ruff check)"          uv run ruff check .
run_step "Format check (ruff format)" uv run ruff format --check .
run_step "Type check (ty)"            uv run ty check
run_step "Tests (pytest)"             uv run pytest -q

echo
if [ "$FAILED" -ne 0 ]; then
    echo -e "${RED}Some checks failed ✘${NC}"
    exit 1
fi
echo -e "${GREEN}All checks passed ✔${NC}"
