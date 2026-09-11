# Deploy — Sicurre-ML Server Setup

This folder contains server-side configuration that is **gitignored** and must
be placed on the Hetzner server manually before first deploy.  
Never commit real credentials or private keys to this folder.

---

## Why a separate Linux user?

The Sicurre application stack runs under its own `sicurre-prod` user.
This repo (the ML inference service) uses `sicurre-ml-prod`, an isolated
user with its own home directory, `authorized_keys`, Docker Compose project,
and permissions.

| Linux user | Owns | Path |
|---|---|---|
| `sicurre-prod` | Sicurre application stack | `/home/sicurre-prod/` |
| `sicurre-ml-prod` | **This repo — ML inference** | `/home/sicurre-ml-prod/` |

---

## 1. Create the `sicurre-ml-prod` user on the server

SSH into the server as your admin user (e.g. root or your existing sudoer):

```bash
ssh <admin-user>@<server-ip>
```

Create the user and add it to the `docker` group (required to run `docker compose`):

```bash
sudo useradd --create-home --shell /bin/bash sicurre-ml-prod
sudo usermod -aG docker sicurre-ml-prod
```

Verify:

```bash
groups sicurre-ml-prod
# Expected output: sicurre-ml-prod : sicurre-ml-prod docker
```

---

## 2. Create a dedicated SSH deploy key pair

Do this **on your local machine** (not on the server):

```bash
ssh-keygen -t ed25519 -C "github-actions-sicurre-ml-deploy" \
  -f ~/.ssh/sicurre_ml_deploy_key -N ""
```

This creates:
- `~/.ssh/sicurre_ml_deploy_key`     — **private key** (goes into GitHub secrets)
- `~/.ssh/sicurre_ml_deploy_key.pub` — **public key** (goes on the server)

Add the public key to the server under the `sicurre-ml-prod` user:

```bash
# On the server as admin:
sudo -u sicurre-ml-prod mkdir -p /home/sicurre-ml-prod/.ssh
sudo -u sicurre-ml-prod tee /home/sicurre-ml-prod/.ssh/authorized_keys \
  < /path/to/sicurre_ml_deploy_key.pub
sudo chmod 700 /home/sicurre-ml-prod/.ssh
sudo chmod 600 /home/sicurre-ml-prod/.ssh/authorized_keys
```

Test the key locally:

```bash
ssh -i ~/.ssh/sicurre_ml_deploy_key sicurre-ml-prod@<server-ip> "echo OK"
```

---

## 3. Create the deploy directory

As `sicurre-ml-prod` on the server:

```bash
sudo -u sicurre-ml-prod mkdir -p /home/sicurre-ml-prod/sicurre-ml/deploy/alloy
```

Or SSH in directly once the key is in place:

```bash
ssh -i ~/.ssh/sicurre_ml_deploy_key sicurre-ml-prod@<server-ip>
mkdir -p ~/sicurre-ml/deploy/alloy
mkdir -p ~/sicurre-ml/deploy/grafana/dashboards
mkdir -p ~/sicurre-ml/deploy/nginx
```

---

## 4. Create the `.env` file on the server

Copy `.env.example` from this repo to the server and fill in real values:

```bash
scp -i ~/.ssh/sicurre_ml_deploy_key \
  .env.example sicurre-ml-prod@<server-ip>:~/sicurre-ml/.env

ssh -i ~/.ssh/sicurre_ml_deploy_key sicurre-ml-prod@<server-ip>
nano ~/sicurre-ml/.env   # fill in all values
```

The `.env` file is **never written by the CD workflow** — you own it on the
server.  The only value CD injects is `IMAGE_TAG` (the Docker image SHA to run).

Create a second, independently owned telemetry environment file from
`deploy/env.alloy.example`:

```bash
cp deploy/env.alloy.example deploy/env.alloy
chmod 600 deploy/env.alloy
```

Populate it with the existing Sicurre Grafana Cloud Prometheus, Loki, and OTLP
destination credentials. Do not put these values in `config.alloy` or commit
`deploy/env.alloy`.

---

## 5. Authenticate Docker to GHCR

The `sicurre-ml-prod` user needs to pull private images from GitHub Container Registry.

Create a GitHub **Personal Access Token (PAT)** with `read:packages` scope at:
https://github.com/settings/tokens

Then on the server:

```bash
ssh -i ~/.ssh/sicurre_ml_deploy_key sicurre-ml-prod@<server-ip>
echo "<YOUR_PAT>" | docker login ghcr.io -u <your-github-username> --password-stdin
```

Docker saves the credentials to `~/.docker/config.json` — this persists across
reboots and container restarts.

---

## 6. Copy server-side config files — AUTOMATED

`deploy/alloy/config.alloy` and `deploy/nginx/api.sicurre.com.conf` are now
committed to the repository. The CD workflow copies them to the server
automatically on every deploy. **No manual `scp` step needed.**

The only one-time manual action required is installing the nginx vhost on the
host nginx instance (whichever nginx container serves traffic). Do this once
after the first deploy:

```bash
# As admin on the server, find where the host nginx reads its vhosts from:
ssh -i ~/.ssh/sicurre_ml_deploy_key sicurre-ml-prod@<server-ip>
# Then, as admin / sudo:
sudo cp ~/sicurre-ml/deploy/nginx/api.sicurre.com.conf \
  /path/to/nginx/conf.d/api.sicurre.com.conf
sudo nginx -t && sudo nginx -s reload
```

> After the first install, nginx config updates are deployed automatically
> by CD — but nginx must be reloaded separately if the vhost file changes.

---

## 7. Add GitHub repository secrets

Go to **Settings → Secrets and variables → Actions** in this repository and add:

| Secret name        | Value                                                      |
|--------------------|------------------------------------------------------------|
| `HETZNER_HOST`     | `<server-ip>`                                                  |
| `HETZNER_USER`     | `sicurre-ml-prod`                                               |
| `HETZNER_SSH_KEY`  | Contents of `~/.ssh/sicurre_ml_deploy_key` (the private key)      |
| `DEPLOY_PATH`      | `/home/sicurre-ml-prod/sicurre-ml`                              |
| `GHCR_OWNER`       | Your GitHub username (used in image path)                       |
| `GRAFANA_URL`      | Grafana Cloud stack URL used by dashboard provisioning          |
| `GRAFANA_API_TOKEN`| Token with folder/dashboard read-write access                   |

The following secrets are also needed at deploy time (CD writes `IMAGE_TAG` to
`.env` on the server; all other secrets must already be in the server's `.env`
and are **not** stored in GitHub):

- `HF_TOKEN`, `HF_USERNAME`, `REPO_NAME`
- `INFERENCE_API_KEY`
- `MISTRAL_API_KEY`, `GROQ_API_KEY` (LLM tier; leave unset to run without it)

The server-owned `deploy/env.alloy` additionally requires:

- `GRAFANA_PROMETHEUS_REMOTE_WRITE_URL`
- `GRAFANA_PROMETHEUS_METRICS_USERNAME`
- `GRAFANA_PROMETHEUS_METRICS_API_TOKEN`
- `GRAFANA_LOKI_URL`, `GRAFANA_LOKI_USER`, `GRAFANA_LOKI_WRITE_API_TOKEN`
- `GRAFANA_OPEN_TELEMETRY_OTLP_ENDPOINT`
- `GRAFANA_OPEN_TELEMETRY_INSTANCE_ID`
- `GRAFANA_OPEN_TELEMETRY_API_TOKEN`

The non-secret `OTEL_TRACE_SAMPLE_PERCENT` setting belongs in the repository-root
`.env`; Compose defaults it to `10` when it is omitted. Grafana credentials remain
exclusively in `deploy/env.alloy`.

---

## 8. First deploy

Once all of the above is done, push to `main` or trigger the CD workflow manually
from the **Actions** tab in GitHub. The workflow will:

1. Build and push the Docker image to GHCR.
2. Copy `docker-compose.prod.yml` to the server.
3. SSH in as `sicurre-ml-prod` and run `docker compose up -d`.
4. Poll `https://api.sicurre.com/v1/health` until it returns 200.
5. Poll `https://api.sicurre.com/v1/ready` until it returns 200 (model loaded).

---

## Troubleshooting

**`docker: permission denied`** — `sicurre-ml-prod` is not in the `docker` group.
Re-run step 1 and log out / log back in.

**`pull access denied for ghcr.io/...`** — Docker not authenticated to GHCR.
Re-run step 5.

**`/v1/ready` stays 503`** — ONNX model is downloading. This can take ~15 min
on a first cold start. Check `docker logs sicurre-ml-app-1 --tail 50`.

**nginx 502** — The app container is not running or bound to the wrong port.
Check `docker compose -f docker-compose.prod.yml ps` as `sicurre-ml-prod`.

## Post-deployment validation

Every deployment and every promotion runs
[`scripts/validate_deployment.py`](scripts/validate_deployment.py) against the
container before the change is accepted. Nothing about a deploy is trusted on
the strength of the image having started.

It checks, in order:

| Check | What it establishes |
|-------|---------------------|
| `GET /v1/health` (60 attempts) | The process is answering |
| `GET /v1/ready` (180 attempts) | The ONNX session finished loading — this is the slow one, since the model may still be downloading |
| `POST /v1/classify`, authenticated | A real classification succeeds with a bearer token, not just that the route exists |
| Response contract fields | The response carries the fields the public contract promises |
| Verdict value | The verdict is inside the published set, so an unknown label cannot reach a caller |
| Identity headers | The response identifies the model that produced it |
| `GET /v1/manifest`, authenticated | `schema_version` is 1 and the model identity matches what was deployed |

Any failure raises and the calling workflow stops.

**Where it runs.** `cd.yml` runs it after deploying to the host.
`promote-model.yml` runs it after pinning a newly promoted model, and a failure
there triggers the rollback path — pointers restored, incumbent restarted,
`rolled_back` reported to Sicurre.

**Running it by hand** against a container you have just started locally:

```bash
INFERENCE_BASE_URL=http://127.0.0.1:8000 \
INFERENCE_API_KEY="$INFERENCE_API_KEY" \
uv run python deploy/scripts/validate_deployment.py
```

It exits non-zero on the first failed check and prints which one, so it is
usable as a smoke test after any local change to the serving path.

A companion script,
[`scripts/validate_observability.py`](scripts/validate_observability.py),
checks that a request produces a privacy-safe trace and an authentication log
without leaking credentials or message content.
