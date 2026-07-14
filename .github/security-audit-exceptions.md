# Dependency audit exceptions

Last reviewed: 2026-07-14

The blocking production audit temporarily ignores only these Starlette
advisories: `PYSEC-2026-161`, `PYSEC-2026-249`, `PYSEC-2026-248`,
`PYSEC-2026-2281`, and `PYSEC-2026-2280`.

The container scan maps the applicable findings to `CVE-2026-48818` and
`CVE-2026-54283`. Those two identifiers are listed in the version-controlled
`.trivyignore`; no other Trivy result is suppressed.

FastAPI 0.136.1 requires Starlette `<0.53.0`, while the advisory database lists
fixes in the incompatible Starlette 1.x line. Forcing Starlette 1.x would create
an unsupported framework combination.

Compensating controls:

- Nginx accepts only the configured Sicurre ML virtual host and limits request
  bodies to 64 KiB.
- `TrustedHostMiddleware` rejects untrusted or malformed Host headers.
- Authentication uses a constant-time service bearer credential and does not
  make authorization decisions from reconstructed URLs or paths.
- The only body-bearing public route accepts a bounded Pydantic JSON contract;
  the application does not parse URL-encoded or multipart forms.
- The application does not mount or use Starlette `StaticFiles`, so the UNC-path
  behavior covered by CVE-2026-48818 is unreachable.
- The public proxy blocks documentation, schema, metrics, and manifest routes.

Remove each exception as soon as a FastAPI release supports its fixed Starlette
version. The non-blocking all-groups audit continues to report every training
and tooling advisory, including vulnerabilities without available fixes.
