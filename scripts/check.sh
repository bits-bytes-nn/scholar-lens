#!/usr/bin/env bash
# The checks CI runs, in one place: `scripts/check.sh` runs both groups,
# `scripts/check.sh lint` or `scripts/check.sh test` one of them. Extra
# arguments go to pytest (`scripts/check.sh test -x -k publisher`).
set -euo pipefail
cd "$(dirname "$0")/.."

lint() {
  poetry run ruff check scholar_lens scripts tests eval
  poetry run black --check scholar_lens scripts tests eval
  poetry run mypy  # files: [tool.mypy] in pyproject.toml
  poetry run mypy --strict scholar_lens/domain scholar_lens/application
  poetry run lint-imports
}

test() {
  poetry run pytest tests/ -p no:cacheprovider \
    --cov=scholar_lens --cov=scripts --cov-report=term-missing \
    --cov-report=xml --cov-fail-under=93 "$@"
}

case "${1:-all}" in
  lint) lint ;;
  test) shift; test "$@" ;;
  all) lint; test ;;
  *) echo "usage: $0 [lint|test [pytest args]|all]" >&2; exit 2 ;;
esac
