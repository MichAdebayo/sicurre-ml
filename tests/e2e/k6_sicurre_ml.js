import http from "k6/http";
import { check, sleep } from "k6";

const baseUrl = __ENV.BASE_URL || "https://api.sicurre.com";
const apiKey = __ENV.INFERENCE_API_KEY;
const useLlm = (__ENV.USE_LLM || "false").toLowerCase() === "true";
const profile = __ENV.PROFILE || "warm";

if (!apiKey) {
  throw new Error("INFERENCE_API_KEY is required");
}

const profiles = {
  cold: {
    executor: "shared-iterations",
    vus: 1,
    iterations: 1,
    maxDuration: "2m",
  },
  warm: {
    executor: "shared-iterations",
    vus: 1,
    iterations: 10,
    maxDuration: "3m",
  },
  concurrent5: {
    executor: "shared-iterations",
    vus: 5,
    iterations: 5,
    maxDuration: "3m",
  },
  sustained: {
    executor: "constant-arrival-rate",
    rate: 1,
    timeUnit: "1s",
    duration: "15m",
    preAllocatedVUs: 2,
    maxVUs: 5,
  },
  burst5: {
    executor: "constant-arrival-rate",
    rate: 5,
    timeUnit: "1s",
    duration: "30s",
    preAllocatedVUs: 5,
    maxVUs: 10,
  },
};

export const options = {
  scenarios: { [profile]: profiles[profile] || profiles.warm },
  thresholds: {
    http_req_duration: [`p(95)<${useLlm ? 8000 : 1000}`],
    checks: ["rate>0.98"],
  },
  discardResponseBodies: false,
  noConnectionReuse: false,
};

export default function () {
  const response = http.post(
    `${baseUrl}/v1/classify`,
    JSON.stringify({
      subject: "Test de performance interne",
      sender: "monitor@sicurre.internal",
      text: "Bonjour, ceci est un message de validation sans lien.",
      use_virustotal: false,
      use_llm: useLlm,
    }),
    {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      timeout: useLlm ? "15s" : "3s",
      tags: { service: "sicurre-ml", mode: useLlm ? "llm" : "local" },
    },
  );

  const burstMayLimit = profile === "burst5" || profile === "concurrent5";
  check(response, {
    "response is successful or intentionally rate-limited": (res) =>
      res.status === 200 || (burstMayLimit && res.status === 429),
    "429 includes Retry-After": (res) =>
      res.status !== 429 || Number(res.headers["Retry-After"]) >= 1,
    "successful response follows contract": (res) => {
      if (res.status !== 200) return true;
      const body = res.json();
      return (
        ["safe", "phishing"].includes(body.verdict) &&
        typeof body.composite_score === "number" &&
        Boolean(res.headers["X-Sicurre-Model-Revision"])
      );
    },
  });

  sleep(0.1);
}
