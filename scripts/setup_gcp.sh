#!/usr/bin/env bash
# setup_gcp.sh — one-shot GCP bootstrap for ClinicalRAG
# Run from the project root after: gcloud auth login
#
# Usage: bash scripts/setup_gcp.sh

set -euo pipefail

PROJECT_ID="clinicalrag-project"
REGION="europe-west3"
REPO="clinicalrag"
SA_NAME="clinicalrag-deployer"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
KEY_FILE="/tmp/clinicalrag-sa-key.json"

# Read secrets from .env (must exist in project root)
ENV_FILE="$(dirname "$0")/../.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env not found at $ENV_FILE"
  exit 1
fi
_env_val() { grep -E "^$1=" "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"'"'" ; }

ANTHROPIC_API_KEY="$(_env_val ANTHROPIC_API_KEY)"
PINECONE_API_KEY="$(_env_val PINECONE_API_KEY)"
PINECONE_INDEX_NAME="$(_env_val PINECONE_INDEX_NAME)"
PINECONE_ENVIRONMENT="$(_env_val PINECONE_ENVIRONMENT)"

for var in ANTHROPIC_API_KEY PINECONE_API_KEY PINECONE_INDEX_NAME PINECONE_ENVIRONMENT; do
  [[ -z "${!var}" ]] && { echo "ERROR: $var is empty in .env"; exit 1; }
done

echo "==> Using project: $PROJECT_ID"
gcloud config set project "$PROJECT_ID" --quiet

# ── 1. Enable required APIs ───────────────────────────────────────────────────
echo "==> Enabling APIs..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  iam.googleapis.com \
  --quiet

# ── 2. Create Artifact Registry repository ───────────────────────────────────
echo "==> Creating Artifact Registry repository..."
if ! gcloud artifacts repositories describe "$REPO" --location="$REGION" --quiet 2>/dev/null; then
  gcloud artifacts repositories create "$REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --description="ClinicalRAG container images" \
    --quiet
  echo "    Created."
else
  echo "    Already exists, skipping."
fi

# ── 3. Create service account ─────────────────────────────────────────────────
echo "==> Creating service account: $SA_NAME..."
if ! gcloud iam service-accounts describe "$SA_EMAIL" --quiet 2>/dev/null; then
  gcloud iam service-accounts create "$SA_NAME" \
    --display-name="ClinicalRAG GitHub Actions deployer" \
    --quiet
  echo "    Created."
else
  echo "    Already exists, skipping."
fi

# ── 4. Grant IAM roles ────────────────────────────────────────────────────────
echo "==> Granting IAM roles..."
for ROLE in \
  roles/artifactregistry.writer \
  roles/run.admin \
  roles/iam.serviceAccountUser \
  roles/secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$ROLE" \
    --quiet > /dev/null
  echo "    $ROLE"
done

# ── 5. Create Secret Manager secrets ─────────────────────────────────────────
echo "==> Storing secrets in Secret Manager..."
_upsert_secret() {
  local name="$1" value="$2"
  if gcloud secrets describe "$name" --quiet 2>/dev/null; then
    echo "$value" | gcloud secrets versions add "$name" --data-file=- --quiet
    echo "    Updated $name"
  else
    echo "$value" | gcloud secrets create "$name" \
      --data-file=- \
      --replication-policy=automatic \
      --quiet
    echo "    Created $name"
  fi
}

_upsert_secret "ANTHROPIC_API_KEY"   "$ANTHROPIC_API_KEY"
_upsert_secret "PINECONE_API_KEY"    "$PINECONE_API_KEY"
_upsert_secret "PINECONE_INDEX_NAME" "$PINECONE_INDEX_NAME"
_upsert_secret "PINECONE_ENVIRONMENT" "$PINECONE_ENVIRONMENT"

# ── 6. Generate service account key ──────────────────────────────────────────
echo "==> Generating service account key → $KEY_FILE"
gcloud iam service-accounts keys create "$KEY_FILE" \
  --iam-account="$SA_EMAIL" \
  --quiet

# ── 7. Print GitHub secrets ───────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GCP bootstrap complete. Add these two secrets to GitHub:"
echo "  github.com/adyasha95/clinicalrag-pipeline/settings/secrets/actions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Secret name : GCP_PROJECT_ID"
echo "Secret value: $PROJECT_ID"
echo ""
echo "Secret name : GOOGLE_CREDENTIALS"
echo "Secret value: (contents of $KEY_FILE — paste the entire JSON below)"
echo ""
cat "$KEY_FILE"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Once both secrets are added, push any commit to main"
echo "  to trigger the full CI/Deploy pipeline."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
