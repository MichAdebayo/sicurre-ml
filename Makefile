.PHONY: serve serve-reload health classify-test phishtank-test install lint typecheck test

# ── Inference API ──────────────────────────────────────────────────────────────

## Start the inference API (production mode, port 8000)
serve:
	uv run uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

## Start with auto-reload for local development
serve-reload:
	uv run uvicorn src.serving.app:app --reload --host 127.0.0.1 --port 8000

## Liveness check (no auth needed)
health:
	curl -s http://localhost:8000/v1/health | python3 -m json.tool

## Readiness check (confirms ONNX model is loaded)
ready:
	curl -s http://localhost:8000/v1/ready | python3 -m json.tool

## Test a phishing classification (requires INFERENCE_API_KEY in env)
classify-test:
	@test -n "$$INFERENCE_API_KEY" || (echo "ERROR: INFERENCE_API_KEY not set" && exit 1)
	curl -s -X POST http://localhost:8000/v1/classify \
	  -H "Authorization: Bearer $$INFERENCE_API_KEY" \
	  -H "Content-Type: application/json" \
	  -d '{"text": "Votre compte Ameli a été suspendu. Cliquez ici: http://ameli-remboursement.tk/verify", "use_llm": false}' \
	  | python3 -m json.tool

## Verify the sicurre app PhishTank endpoint is reachable (requires INTERNAL_API_KEY in env)
phishtank-test:
	@test -n "$$INTERNAL_API_KEY" || (echo "ERROR: INTERNAL_API_KEY not set" && exit 1)
	curl -s \
	  -H "Authorization: Bearer $$INTERNAL_API_KEY" \
	  "$$PHISHTANK_ENDPOINT_URL" \
	  | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'OK — {d[\"count\"]} URLs from {d[\"source\"]} at {d[\"generated_at\"]}')"

# ── Dev tooling ────────────────────────────────────────────────────────────────

install:
	uv sync

lint:
	uv run ruff check src .github/scripts tests

typecheck:
	uv run mypy src .github/scripts tests --ignore-missing-imports

test:
	uv run pytest

check: lint typecheck test
