#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
hook_path="$repo_root/.git/hooks/pre-commit"
cp "$repo_root/scripts/pre-commit.sh" "$hook_path"
chmod +x "$hook_path"
echo "Installed $hook_path"
