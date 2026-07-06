# Changelog

## v2.2-onnx — 2026-07-03

### Fixed
- TensorRT engine-cache detection now looks for the `.engine` files ONNX
  Runtime actually writes (was `*.trt`), so the "first run will compile the
  engine" warning no longer fires on every start when a cache exists.
- Output-format detection now accepts custom models with fewer than 80
  classes (standard format requires `attrs >= 5`, i.e. 4 box coords + at
  least one class). Previously a valid 10-class export like `[1, 14, 8400]`
  was rejected at startup.
- E2E post-processing filters NMS zero-padding rows explicitly, so they can
  no longer leak into results when a client sends `min_confidence=0.0`.
- Requests whose `Content-Length` exceeds `MAX_IMAGE_BYTES` (plus base64/
  framing headroom) are rejected with HTTP 413 before the body is buffered.
- Corrected the `MAX_IMAGE_PIXELS` comment (the default is a 40MP cap, not
  200MP) and the stale `run_inference_sync` docstring.
- CPU-fallback startup warning no longer assumes CUDA was expected
  (was misleading on non-NVIDIA hosts).

### Added
- `NMS_IOU` environment variable — NMS IoU threshold for standard-format
  models (default 0.45, previously hardcoded).
- `INFER_CONCURRENCY` environment variable — bounds concurrent inference
  calls with a semaphore (default 2) so request bursts queue instead of
  oversubscribing the GPU/CPU.
- `TRT_FP16` now accepts `true`/`yes`/`on`/`1` (was `int()`-only and crashed
  on `TRT_FP16=true`).

### Changed
- Per-class NMS is a single `cv2.dnn.NMSBoxes` call using class-offset
  coordinates on numpy arrays, replacing the per-class Python loop with
  list conversions.
- Input blob construction casts uint8 channels on assignment and normalizes
  in place (one `/= 255.0`), eliminating per-channel float temporaries;
  winning class scores reuse the argmax result via `np.take_along_axis`.
- Extracted `boxes_to_predictions()` — the shared letterbox-inversion /
  clamp / degenerate-box-filter / response-building tail of both
  post-processors.
- Warm-up uses a single all-zeros blob directly (the previous
  letterbox-and-fill of a zeros image was a no-op).
- JSON request parsing and base64 image decoding now run off the event loop
  in a single worker-thread hop (new `parse_json_request()` helper). With a
  10MB image the JSON body is ~14MB of base64; parsing it on the loop
  blocked all other requests for tens of milliseconds. Error responses are
  unchanged.

## v2.1-onnx — 2026-03-02

- Prior release (before this changelog was introduced).
