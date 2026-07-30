# Cosmos3 startup acceptance seed

`representative_request.multipart` is a real-image, model-native request used only
for the two typed acceptance calls in `scripts/cosmos3_guarded_cutover.sh`.
`representative_request_headers.json` supplies its multipart content type and the
SHA-256 of the embedded JPEG (`6842f3be...69a8ba4`).

The request carries an opaque `source_id` (`cosmos3-acceptance-image-00`), not a
server filesystem path. It contains no model weights, credentials, or evaluator
targets. Benchmark sweeps use separate run-root artifacts and are not bundled.
