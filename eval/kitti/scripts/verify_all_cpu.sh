#!/usr/bin/env bash
# CPU fixtures for this standalone repository only.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$root"
export PYTHONPATH="$root/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_ADDOPTS=
export CUDA_VISIBLE_DEVICES=
python="${PYTHON:-python3}"

# Snapshot actual local content/identity, including ignored and already-dirty files.
# Existing authored edits are allowed only when verification leaves them unchanged.
before="$("$python" -B "$root/scripts/snapshot_repository.py" "$root")"
git_state=false
if [[ -e "$root/.git" ]]; then
  git_state=true
  while IFS= read -r path; do
    case "$path" in
      results/*|outputs/*|prepared/*|.runtime/*|runtime/*|logs/*|*/__pycache__/*|*.pyc)
        printf 'Tracked generated artifact: %s\n' "$path" >&2
        exit 1
        ;;
    esac
  done < <(git ls-files)
fi

"$python" -B -m pytest -q -p no:cacheprovider "$root/tests"
"$python" -B -m kitti_eval --help >/dev/null
for script in "$root"/scripts/*.sh; do
  bash -n "$script"
done

after="$("$python" -B "$root/scripts/snapshot_repository.py" "$root")"
if [[ "$before" != "$after" ]]; then
  printf 'Generated repository changes detected during CPU verification.\n' >&2
  diff -u --label before --label after <(printf '%s\n' "$before") <(printf '%s\n' "$after") >&2 || true
  exit 1
fi
if "$git_state"; then
  git diff --check
fi
printf 'CPU verification passed for kitti_eval; no real-data or GPU validation.\n'
