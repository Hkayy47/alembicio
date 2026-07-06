# SECURITY.md

**Embedding vectors are sensitive data.** Research (vec2vec, arXiv:2505.12540) shows
vectors alone — with no access to the model or documents — support attribute inference
and partial inversion of the underlying text. Treat vector stores, exported anchors, and
fitted mapping artifacts (`.npz`) with the same care as the source documents. alembicio
never transmits vectors anywhere except the configured store and never logs document
text or secrets.

Secrets enter only via `env:` indirection in `embmigrate.yaml`. Anchor samples for
mapping mode are drawn from your corpus and stay local.

Report vulnerabilities via the repository's private security advisory channel; do not
open public issues for exploitable findings.
