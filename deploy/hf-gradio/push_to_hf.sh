#!/usr/bin/env bash
# Sync ollive Gradio app to a Hugging Face Docker Space and push.
# Usage: ./deploy/hf-gradio/push_to_hf.sh KN123/ollive-gradio
set -euo pipefail

SPACE_ID="${1:-KN123/ollive-gradio}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${ROOT}/.hf-space-sync"
RSYNC_EXCLUDES=(
  --exclude='__pycache__/'
  --exclude='*.py[cod]'
)

if [[ -f "${ROOT}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a && source "${ROOT}/.env" && set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required (set in .env or environment)." >&2
  exit 1
fi

rm -rf "${WORK}"
git clone "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE_ID}" "${WORK}"

rsync -a \
  "${ROOT}/app.py" \
  "${ROOT}/config.py" \
  "${ROOT}/logging_config.py" \
  "${WORK}/"

mkdir -p "${WORK}/api"
rsync -a "${ROOT}/api/__init__.py" "${WORK}/api/__init__.py"
rsync -a --delete --delete-excluded "${RSYNC_EXCLUDES[@]}" \
  "${ROOT}/api/agent_tools/" "${WORK}/api/agent_tools/"

for _pkg in assistants evaluation llm memory tools; do
  mkdir -p "${WORK}/${_pkg}"
  rsync -a --delete --delete-excluded "${RSYNC_EXCLUDES[@]}" \
    "${ROOT}/${_pkg}/" "${WORK}/${_pkg}/"
done

cp "${ROOT}/deploy/hf-gradio/Dockerfile.space" "${WORK}/Dockerfile"
cp "${ROOT}/deploy/hf-gradio/requirements.txt" "${WORK}/requirements.txt"
cp "${ROOT}/deploy/hf-gradio/README.space.md" "${WORK}/README.md"

cat > "${WORK}/.dockerignore" <<'EOF'
.git
.gitattributes
__pycache__
**/__pycache__
*.py[cod]
.env
.venv
venv
results
*.log
.DS_Store
EOF

cd "${WORK}"
git add -A
if git diff --staged --quiet; then
  echo "No changes to push."
  exit 0
fi

git commit -m "Deploy ollive Gradio Docker Space"
git push

echo "Pushed to https://huggingface.co/spaces/${SPACE_ID}"
