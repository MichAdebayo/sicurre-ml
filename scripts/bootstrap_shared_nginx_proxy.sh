#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# bootstrap_shared_nginx_proxy.sh
#
# Sets up the shared nginx-proxy on the Hetzner server.
# Uploads prebuilt config files from this repo — no manual heredoc typing needed.
#
# CERT REQUIREMENT (read before running):
#   This script installs nginx-proxy for sicurre.com traffic.
#   It needs a Cloudflare Origin CA certificate for the sicurre.com zone.
#   This is NOT the same as the vinse.app cert — they are separate Cloudflare zones.
#
#   How to generate the sicurre.com cert (one-time, ~5 min):
#     1. Cloudflare dashboard → select sicurre.com zone
#     2. SSL/TLS → Origin Server → Create Certificate
#     3. Hostnames: sicurre.com  *.sicurre.com
#     4. Validity: 15 years
#     5. Download Certificate → save as:  deploy/nginx/ssl/sicurre-origin.pem
#     6. Download Private Key  → save as:  deploy/nginx/ssl/sicurre-origin-key.pem
#   Then run this script with --cert and --cert-key pointing to those files.
#   The ssl/ folder is gitignored — keys never enter the repo.
#
# Usage:
#   scripts/bootstrap_shared_nginx_proxy.sh \
#     --user root \
#     --cert deploy/nginx/ssl/sicurre-origin.pem \
#     --cert-key deploy/nginx/ssl/sicurre-origin-key.pem
#
# Optional flags:
#   --host <ip>       Default: 77.42.67.255
#   --key  <path>     SSH private key (default: ssh agent)
#   --with-api-vhost  Also install api.sicurre.com.conf into conf.d
# ─────────────────────────────────────────────────────────────────────────────

HOST="77.42.67.255"
REMOTE_USER=""
SSH_KEY=""
CERT_FILE=""
CERT_KEY_FILE=""
WITH_API_VHOST="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)       HOST="$2";          shift 2 ;;
    --user)       REMOTE_USER="$2";   shift 2 ;;
    --key)        SSH_KEY="$2";       shift 2 ;;
    --cert)       CERT_FILE="$2";     shift 2 ;;
    --cert-key)   CERT_KEY_FILE="$2"; shift 2 ;;
    --with-api-vhost) WITH_API_VHOST="true"; shift ;;
    -h|--help)    sed -n '1,50p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Validate required arguments ──────────────────────────────────────────────
if [[ -z "$REMOTE_USER" ]]; then
  echo "ERROR: --user is required."
  echo "Run with --help for usage."
  exit 1
fi

if [[ -z "$CERT_FILE" || -z "$CERT_KEY_FILE" ]]; then
  echo ""
  echo "ERROR: --cert and --cert-key are required."
  echo ""
  echo "You need a Cloudflare Origin CA certificate for the sicurre.com zone."
  echo "Steps:"
  echo "  1. Cloudflare dashboard → sicurre.com zone"
  echo "  2. SSL/TLS → Origin Server → Create Certificate"
  echo "  3. Hostnames: sicurre.com  *.sicurre.com   (covers api.sicurre.com)"
  echo "  4. Download cert → deploy/nginx/ssl/sicurre-origin.pem"
  echo "  5. Download key  → deploy/nginx/ssl/sicurre-origin-key.pem"
  echo "  6. Rerun: scripts/bootstrap_shared_nginx_proxy.sh --user $REMOTE_USER \\"
  echo "            --cert deploy/nginx/ssl/sicurre-origin.pem \\"
  echo "            --cert-key deploy/nginx/ssl/sicurre-origin-key.pem"
  echo ""
  echo "NOTE: The ssl/ folder is gitignored — cert files never enter the repo."
  exit 1
fi

for f in "$CERT_FILE" "$CERT_KEY_FILE"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: File not found: $f"
    exit 1
  fi
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

COMPOSE_FILE="$REPO_ROOT/deploy/nginx/nginx-proxy.compose.yml"
REDIRECT_FILE="$REPO_ROOT/deploy/nginx/00-redirect.conf"
API_VHOST_FILE="$REPO_ROOT/deploy/nginx/api.sicurre.com.conf"

for f in "$COMPOSE_FILE" "$REDIRECT_FILE"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Required file not found: $f"
    exit 1
  fi
done

if [[ "$WITH_API_VHOST" == "true" && ! -f "$API_VHOST_FILE" ]]; then
  echo "ERROR: Missing API vhost file: $API_VHOST_FILE"
  exit 1
fi

SSH_ARGS=(-o StrictHostKeyChecking=accept-new)
if [[ -n "$SSH_KEY" ]]; then
  # Key-based auth — non-interactive
  SSH_ARGS+=(-i "$SSH_KEY" -o BatchMode=yes)
else
  # Password auth — terminal will prompt once per command
  echo "No --key provided; will prompt for password on each connection."
  echo "Tip: run 'ssh-copy-id root@$HOST' after bootstrap to set up key auth."
fi

TARGET="${REMOTE_USER}@${HOST}"

# ── Upload files ─────────────────────────────────────────────────────────────
echo "==> Uploading config files to $TARGET"
scp "${SSH_ARGS[@]}" "$COMPOSE_FILE"   "$TARGET:~/nginx-proxy.compose.yml"
scp "${SSH_ARGS[@]}" "$REDIRECT_FILE"  "$TARGET:~/00-redirect.conf"
scp "${SSH_ARGS[@]}" "$CERT_FILE"      "$TARGET:~/sicurre-origin.pem"
scp "${SSH_ARGS[@]}" "$CERT_KEY_FILE"  "$TARGET:~/sicurre-origin-key.pem"

if [[ "$WITH_API_VHOST" == "true" ]]; then
  scp "${SSH_ARGS[@]}" "$API_VHOST_FILE" "$TARGET:~/api.sicurre.com.conf"
fi

# ── Remote install ────────────────────────────────────────────────────────────
echo "==> Installing on server and starting nginx-proxy"
ssh "${SSH_ARGS[@]}" "$TARGET" "WITH_API_VHOST='$WITH_API_VHOST' bash -s" <<'REMOTE'
set -euo pipefail

SUDO=""
if [[ $(id -u) -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
fi

$SUDO mkdir -p /opt/nginx-proxy/conf.d /opt/nginx-proxy/ssl

$SUDO cp ~/nginx-proxy.compose.yml       /opt/nginx-proxy/docker-compose.yml
$SUDO cp ~/00-redirect.conf              /opt/nginx-proxy/conf.d/00-redirect.conf

# Cert files are specific to sicurre.com zone — not copied from Vinse.
$SUDO cp ~/sicurre-origin.pem            /opt/nginx-proxy/ssl/origin.pem
$SUDO cp ~/sicurre-origin-key.pem        /opt/nginx-proxy/ssl/origin-key.pem
$SUDO chmod 640 /opt/nginx-proxy/ssl/origin.pem /opt/nginx-proxy/ssl/origin-key.pem

if [[ "$WITH_API_VHOST" == "true" ]]; then
  $SUDO cp ~/api.sicurre.com.conf /opt/nginx-proxy/conf.d/api.sicurre.com.conf
fi

cd /opt/nginx-proxy
$SUDO docker compose -f docker-compose.yml up -d

echo "--- nginx -t output ---"
$SUDO docker exec nginx-proxy nginx -t
$SUDO docker exec nginx-proxy nginx -s reload

# Clean up temp uploads
$SUDO rm -f ~/nginx-proxy.compose.yml ~/00-redirect.conf \
             ~/sicurre-origin.pem ~/sicurre-origin-key.pem \
             ~/api.sicurre.com.conf 2>/dev/null || true

echo "nginx-proxy is up. Vhosts installed:"
ls /opt/nginx-proxy/conf.d/
REMOTE

echo "==> Done. nginx-proxy is running on $HOST."
