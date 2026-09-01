#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if ! /exec-daemon/gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated. Run:"
  echo "  gh auth login --hostname github.com --git-protocol https --web"
  exit 1
fi

git remote remove github 2>/dev/null || true
git remote add github "https://github.com/Ckck12/Route-Guied-CBF-QP.git"

git push github main:main
echo "Pushed to https://github.com/Ckck12/Route-Guied-CBF-QP"
echo "Pages will update at https://ckck12.github.io/Route-Guied-CBF-QP/ within 1-2 minutes."
