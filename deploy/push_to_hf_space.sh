#!/usr/bin/env bash
# Push the runtime subset of this repo to a Hugging Face Space (Docker SDK).
#
# The Space gets its own git history: this repo's history contains large
# binaries committed without LFS, which the Hugging Face Hub rejects. The
# Space repo tracks files >10 MB (faiss_index.bin, graph.json) with git LFS.
#
# Usage:  deploy/push_to_hf_space.sh [owner/space-name] [work-dir]
# Needs:  git, git-lfs, and an HF login with git credentials
#         (`hf auth login --add-to-git-credential`).
set -euo pipefail

SPACE_ID="${1:-aryantalati51/supplementsrx-ai}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
WORK="${2:-$(mktemp -d)}/space"
SRV="Phase 5 RAG&Frontend"

FILES=(
  README.md Dockerfile .dockerignore requirements.txt
  "$SRV/main.py" "$SRV/rag_functionality.py" "$SRV/phase4_langgraph.py"
  "$SRV/graph_backend.py" "$SRV/index.html" "$SRV/faiss_index.bin"
  "$SRV/supplement_names.json" "$SRV/supplement_metadata.json"
  "$SRV/embedding_config.json"
  Knowledge_Graph/graph.json Knowledge_Graph/neo4j_communities.json
)

git lfs version >/dev/null

rm -rf "$WORK"
# Skip downloading LFS blobs from the Space; every tracked file is replaced below.
GIT_LFS_SKIP_SMUDGE=1 git clone "https://huggingface.co/spaces/$SPACE_ID" "$WORK"
cd "$WORK"
git lfs install --local >/dev/null

# Mirror the file list exactly (drop anything no longer shipped).
git ls-files -z | xargs -0 -r git rm -q --cached --ignore-unmatch -- >/dev/null
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +

cat > .gitattributes <<'EOF'
*.bin filter=lfs diff=lfs merge=lfs -text
Knowledge_Graph/graph.json filter=lfs diff=lfs merge=lfs -text
EOF

for f in "${FILES[@]}"; do
  mkdir -p "$(dirname "$f")"
  cp "$REPO_ROOT/$f" "$f"
done

git add -A
if git diff --cached --quiet; then
  echo "Space already up to date."
  exit 0
fi
SRC_REV="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
git commit -q -m "Deploy SupplementsRx demo (source ${SRC_REV})"
echo "LFS-tracked files:"; git lfs ls-files
git push origin HEAD:main
echo "Pushed to https://huggingface.co/spaces/$SPACE_ID"
