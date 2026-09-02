# API Contract And Coverage Policy

## Objective

The runtime FastAPI application and its Pydantic request and response models
are the executable source for the inference OpenAPI contract. The reviewed
contract remains checked in at `docs/api/openapi.yaml` so changes are visible
in pull requests and available without starting the service.

The repository must also measure the complete `src/` package. A high score for
only the training modules must not be presented as repository-wide coverage.

## OpenAPI Workflow

1. Change a FastAPI route or Pydantic model.
2. Run `make openapi` to regenerate `docs/api/openapi.yaml` deterministically.
3. Review the generated diff as an API contract change.
4. Run `make openapi-check`, or rely on CI, to reject contract drift.

Generation does not replace human review of descriptions, examples, security
requirements, status codes or compatibility impact.

## Coverage Gates

Two gates serve different purposes:

| Gate | Scope | Minimum |
|---|---|---:|
| Training core | `src/config`, `src/data`, `src/model`, `src/training` | 90% |
| Full source | all of `src/` | 80% |

The full-source gate uses deterministic fakes for HTTP providers, ONNX Runtime,
MLflow, Hugging Face and storage boundaries. Live integration checks validate
credentials and external compatibility separately. Network access is never
required merely to exercise application control flow.

Coverage exclusions are reserved for genuinely non-executable typing branches
or platform guards. External calls, error handling and rollback paths must not
be excluded solely because they require test doubles.

## Future Threshold

After the 80% full-source gate remains stable, raise it toward 85% alongside
branch coverage. Do not raise the number by adding tests that only duplicate
implementation details or assertions without behavioral value.
