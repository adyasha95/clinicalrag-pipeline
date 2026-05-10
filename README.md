# ClinicalRAG

Retrieval-augmented generation over ClinicalTrials.gov records.  
Ask natural-language questions; get grounded answers citing NCT IDs.

---

## CI/CD — GitHub Secrets

The deploy workflow (`.github/workflows/deploy.yml`) requires the following
secrets to be configured in **GitHub → Settings → Secrets and variables →
Actions → New repository secret**.

### Required secrets

| Secret name | What to put there |
|---|---|
| `GOOGLE_CREDENTIALS` | JSON key for a GCP service account with the roles below. Generate it in **IAM → Service Accounts → Keys → Add key → JSON**, then paste the entire file contents. |
| `GCP_PROJECT_ID` | Your Google Cloud project ID (e.g. `my-project-123`). |
| `ANTHROPIC_API_KEY` | Anthropic API key — stored in **GCP Secret Manager** *and* mirrored here so the workflow can pass it to Cloud Run via `--update-secrets`. |
| `PINECONE_API_KEY` | Pinecone API key (same dual-storage pattern). |
| `PINECONE_INDEX_NAME` | Name of the Pinecone index (e.g. `clinicalrag`). |
| `PINECONE_ENVIRONMENT` | Pinecone environment (e.g. `us-east-1-aws`). |

> **Note:** `ANTHROPIC_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, and
> `PINECONE_ENVIRONMENT` must also exist as secrets in **GCP Secret Manager**
> with the same names. Cloud Run pulls them at runtime via `--update-secrets`;
> the GitHub secrets are not injected into the container directly.

### Required GCP service account roles

The service account whose JSON key is stored in `GOOGLE_CREDENTIALS` needs:

| Role | Why |
|---|---|
| `roles/artifactregistry.writer` | Push Docker images |
| `roles/run.admin` | Create / update Cloud Run services |
| `roles/iam.serviceAccountUser` | Act as the Cloud Run runtime service account |
| `roles/secretmanager.secretAccessor` | Let Cloud Run read secrets at deploy time |

### Artifact Registry setup (one-time)

Before the first push, create the repository:

```bash
gcloud artifacts repositories create clinicalrag \
  --repository-format=docker \
  --location=europe-west3 \
  --description="ClinicalRAG container images"
```

### Workflow summary

```
push to main
  └── test        pytest tests/ (mocked, no live API calls)
        └── build   docker build + push → europe-west3-docker.pkg.dev/.../app:<sha>
              └── deploy  gcloud run deploy clinicalrag-prod
                            --min-instances=0 --max-instances=3 --memory=1Gi
```
