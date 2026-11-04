# Repository constraints

- Keep ONNX acquisition, export and reference checks independent of vc-rs. Do not
  add inference-engine or real-time audio code here.
- Treat app version, checkpoint, feature layer, projection and waveform
  preprocessing as separate parts of compatibility. Never silently substitute
  one embedder or a randomly initialized final projection for another.
- Pin source revisions and SHA-256 before loading weights. Use weights_only=True,
  no remote model code, and fail on unexpected/missing backbone state keys.
- Numerical parity means the tested feature tensors only. Distinguish imported
  ONNX smoke checks, fairseq parity, CPU ONNX parity, GPU runtime and listening
  validation. Do not promote one kind of evidence into another.
- Keep caches, environments, model files, audio and local paths out of Git.
- Use meaningful shape/numerical checks for changes to extraction/export.
- Commit, push and publish only when explicitly requested.

## Documentation

- Write for readers who have not seen the development conversation. Prioritize
  model selection, usage, compatibility limits and reproducible verification.
  Lead with the conclusion or action the reader needs.
- Decide whether information helps a reader use, assess, reproduce or maintain
  the tool before adding it to public documentation. Being safe to disclose is
  not, by itself, a reason to publish it.
- Keep the README focused on getting started and choosing the right model.
  Put supporting explanations in focused documents and link to evidence where
  it supports a specific claim; do not make every raw record required reading.
- Keep the repository simple. Store work logs, download/HTTP logs, full API
  responses, superseded summaries and intermediate diagnostics under ignored
  artifacts/, outside Git and source archives. Add public documents or evidence
  files only when they contribute distinct information needed by readers.
- Describe stable behavior and dated findings rather than work progress.
  Avoid conversation-dependent wording such as "this time", "previously" or
  "added" without a concrete version or condition. Do not present locally
  generated packages as publicly available releases.
- Publish concise verification summaries and reproduction steps. Identify the
  source revision/checkpoint and artifact hashes, input/output scope, relevant
  environment, tolerances, observed errors and untested areas. Preserve failed
  conditions and the reasons for tolerance changes in the explanation.
- Retaining evidence does not require committing or publishing every record.
  Include detailed tensor comparisons, intermediate outputs and CPU diagnostics
  only when they provide useful support beyond the summary. Keep other evidence
  locally; do not delete underlying records merely to shorten documentation.
- Include environment details when they affect reproduction or interpretation.
  Explain the relevance of tool-specific conditions such as sandboxing; omit
  incidental execution history, personal paths, machine identifiers and audio.
  Do not generalize a local failure to all users or environments.
- Keep historical measurements distinct from current instructions and test
  results. Do not rewrite recorded evidence to imply a new validation run.
  Preserve upstream license texts and source documents verbatim, including their
  provenance and hashes; put explanations in our own documents.
