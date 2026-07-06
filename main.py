# main.py

# siyolo -- A Simple YOLO Inference Server (ONNX Version)
# A drop-in replacement for DeepStack AI and/or CodeProject.AI running YOLO models via ONNX Runtime
#
# https://github.com/nemesis2/siyolo
# Released under the MIT License, see included LICENSE
# Last Updated: 2026-07-03 -- By nemesis2

VERSION = "v2.2-onnx"


# Server configuration — all settings are overridable via environment variables

import os

SERVER_LISTEN = os.environ.get("SERVER_LISTEN", "127.0.0.1")                # ip address(es) to bind to (127.0.0.1)
SERVER_PORT = int(os.environ.get("SERVER_PORT", 32168))                     # listening port (32168/5000)
UVICORN_LOG = os.environ.get("UVICORN_LOG", "warning")                      # uvicorn log level (warning)
MINIMUM_CONFIDENCE = float(os.environ.get("MINIMUM_CONFIDENCE", 0.65))      # default confidence if not set in post (0.65)
NMS_IOU = float(os.environ.get("NMS_IOU", 0.45))                            # NMS IoU threshold for standard-format models (0.45)
SERVER_MODEL = os.environ.get("YOLO_MODEL", "yolo26x.onnx")                 # YOLO ONNX model to use

# Image target dimensions (Must match what the model was exported at!)
IMG_SZ_X = int(os.environ.get("IMG_SZ_X", 1152))                            # image width for inference (default 1152)
IMG_SZ_Y = int(os.environ.get("IMG_SZ_Y", 640))                             # image height for inference; must be a multiple of 32!

MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", 10 * 1024 * 1024))  # 10MB Max image size
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", 40_000_000))      # 40MP safety cap
KEEPALIVE_TIMEOUT = int(os.environ.get("KEEPALIVE_TIMEOUT", 70))            # HTTP keep-alive idle timeout in seconds (70)
INFER_CONCURRENCY = int(os.environ.get("INFER_CONCURRENCY", 2))             # max concurrent inference calls (2)

# TensorRT / CoreML configuration
TRT_FP16 = os.environ.get("TRT_FP16", "1").strip().lower() in ("1", "true", "yes", "on")
TRT_CACHE_DIR = os.environ.get("TRT_CACHE_DIR", "")                         # empty = use model directory
WARMUP_RUNS = int(os.environ.get("WARMUP_RUNS", 5))                         # warm-up inference passes at startup


import ast
import sys
import json
import time
import base64
import asyncio
import logging
import platform

from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException


# Logging setup
# If stdout is a terminal, include timestamps. If piped to syslog (systemd
# service), omit them — journald adds its own timestamps.
# os.isatty(1) checks stdout and can return True even when stdout is
# redirected to a pipe while stdin is a terminal.
try:
    is_tty = os.isatty(sys.stdout.fileno())
except Exception:
    is_tty = False
log_format = '%(asctime)s [%(levelname)s] %(message)s' if is_tty else '[%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%H:%M:%S')
logger = logging.getLogger("siyolo")


# ONNX output format constants
#
# FORMAT_STANDARD: [1, attrs, N]
#   Default export from Ultralytics (YOLOv8/v9/v10/v26 with end2end=False).
#   Raw candidate boxes requiring confidence filtering + NMS.
#   Column layout: [cx, cy, w, h, class0_prob, class1_prob, ...]
#
# FORMAT_E2E: [1, N, 6]
#   End-to-end export with NMS embedded in the model graph.
#   Pre-filtered and pre-NMS'd detections, ready to consume directly.
#   Column layout: [x_min, y_min, x_max, y_max, confidence, class_id]
#   Rows with confidence == 0.0 are NMS padding and are filtered out.
FORMAT_STANDARD = "standard"
FORMAT_E2E      = "e2e"


# Utility functions

def get_cpu_name() -> str:
    """Return the CPU model name string for startup logging."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def build_providers(base_dir: Path) -> list:
    """
    Build the ONNX Runtime execution provider list in best-available order.
    Priority: TensorRT → CUDA → CoreML → CPU
    TensorRT and CUDA are available on Linux/Windows with a CUDA GPU.
    CoreML is available on macOS (and Apple Silicon in particular).
    CPU is always available as the final fallback.
    TensorRT options:
      - FP16 enabled if TRT_FP16=1 (default) and the GPU supports it.
        ONNX Runtime checks capability at runtime; if the GPU doesn't support
        FP16 it silently falls back to FP32 for those ops.
      - Engine cache: serialized .trt files are written to TRT_CACHE_DIR
        (default: model directory). First run compiles the engine (slow, ~1-5
        min); subsequent runs load from cache (fast, ~seconds).
    CoreML options:
      - MLComputeUnits=ALL requests ANE + GPU + CPU in that priority order.
        ONNX Runtime / CoreML dispatches each op to the best available unit.
    """
    trt_cache = TRT_CACHE_DIR if TRT_CACHE_DIR else str(base_dir / "models")
    trt_opts = {
        "trt_fp16_enable":                "True" if TRT_FP16 else "False",
        "trt_engine_cache_enable":        "True",
        "trt_engine_cache_path":          trt_cache,
        "trt_max_workspace_size":         "4294967296",   # 4 GB
        "trt_builder_optimization_level": "3",
    }
    coreml_opts = {
        "MLComputeUnits": "ALL",   # ANE > GPU > CPU, CoreML decides per-op
    }

    available = ort.get_available_providers()
    providers = []

    if "TensorrtExecutionProvider" in available:
        import ctypes
        import glob

        # try common paths, custom/non-standard paths will cause it to fail
        # but stops error messages if trt not available
        if sys.platform == "win32":
            trt_libs = glob.glob("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/bin/nvinfer.dll")
        else:
            trt_libs = glob.glob("/usr/lib/x86_64-linux-gnu/libnvinfer.so.*") or \
                       glob.glob("/usr/local/lib/libnvinfer.so.*") or \
                       glob.glob("/usr/lib/aarch64-linux-gnu/libnvinfer.so.*")  # ARM (Jetson)

        trt_available = False
        for lib in trt_libs:
            try:
                ctypes.CDLL(lib)
                trt_available = True
                break
            except OSError:
                continue

        if trt_available:
            providers.append(("TensorrtExecutionProvider", trt_opts))

    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")

    if "CoreMLExecutionProvider" in available:
        providers.append(("CoreMLExecutionProvider", coreml_opts))

    providers.append("CPUExecutionProvider")   # always last, always present
    return providers


def decode_image(img_data: bytes) -> np.ndarray:
    """
    Decode raw image bytes to an OpenCV BGR ndarray.
    Runs in a thread via asyncio.to_thread() to avoid blocking the event loop.
    Thread-safe: no shared state.
    """
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def parse_json_request(body: bytes, min_confidence: float) -> tuple:
    """
    Parse a JSON detection request body and decode its base64 image payload.
    With a 10MB image the body is ~14MB of base64 inside JSON; json.loads and
    b64decode are both CPU-bound at that size, so this runs in a thread via
    asyncio.to_thread() (one hop for both) to avoid blocking the event loop.
    Thread-safe: no shared state.
    Returns (raw_image_bytes, min_confidence).
    Raises HTTPException(400) on any validation failure.
    """
    body_json = json.loads(body)
    if "image" not in body_json:
        raise HTTPException(status_code=400, detail="Missing 'image' in JSON body")
    # Override min_confidence from JSON body if provided, then re-validate.
    # Re-validation is required because the form-level check in the endpoint
    # only covers the multipart path; a JSON client can supply any float value.
    try:
        min_confidence = float(body_json.get("min_confidence", min_confidence))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid min_confidence value")
    if not (0.0 <= min_confidence <= 1.0):
        raise HTTPException(status_code=400, detail="min_confidence must be between 0.0 and 1.0")
    img_b64 = body_json["image"]
    if not isinstance(img_b64, str):
        raise HTTPException(status_code=400, detail="'image' must be a base64 string")
    # Strip optional data URI prefix: "data:image/jpeg;base64,<data>"
    if img_b64.startswith("data:"):
        img_b64 = img_b64.split(",", 1)[1]
    raw_data = base64.b64decode(img_b64, validate=True)
    if not raw_data or len(raw_data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Invalid or oversized image")
    return raw_data, min_confidence


def letterbox(img: np.ndarray, new_shape: tuple, color=(114, 114, 114)) -> tuple:
    """
    Resize image to a 32-pixel-multiple rectangle, keeping aspect ratio,
    padding with gray to prevent accuracy-destroying squashing.

    new_shape is (width, height).

    Returns:
        padded_img: The letterboxed OpenCV image at exactly new_shape dimensions.
        ratio:      Uniform scale factor applied to both axes (new / old).
        pad:        (left, top) integer pixel offsets of the padding actually
                    applied by copyMakeBorder. Used by the post-processors to
                    reverse the letterbox transform and map inference-space
                    coordinates back to original image space.
    """
    shape = img.shape[:2]  # [height, width]

    # Uniform scale ratio — fit the longer side to new_shape
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])

    # Compute unpadded dimensions after scaling
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # (width, height)
    dw = new_shape[0] - new_unpad[0]   # total horizontal padding needed
    dh = new_shape[1] - new_unpad[1]   # total vertical padding needed

    dw /= 2   # split evenly across left / right
    dh /= 2   # split evenly across top / bottom

    if shape[::-1] != new_unpad:   # only resize if dimensions actually changed
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Compute the integer pixel offsets actually applied to each border.
    # The -0.1 / +0.1 bias ensures consistent rounding when dw/dh is a
    # half-integer (e.g. 0.5 → top=0, bottom=1 rather than both rounding to 1).
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # Return the actual applied left/top offsets so post-processors can invert
    # the transform exactly. Returning the raw float dw/dh halves would introduce
    # a sub-pixel error when the total padding is odd.
    return img, r, (left, top)


# ONNX model output format detection
def detect_output_format(session: ort.InferenceSession) -> str:
    """
    Inspect the ONNX session's first output tensor shape and return the
    appropriate format constant.
    Standard export: [1, 5+, N]  — raw candidates, needs filtering + NMS
    E2E export:      [1, N, 6]   — pre-filtered, pre-NMS, ready to consume
    Raises RuntimeError on unrecognised shapes (including dynamic-shape
    exports where dims are None/strings) so a bad model file is caught
    at startup rather than silently producing wrong results at inference time.
    """
    shape = session.get_outputs()[0].shape
    logger.info(f"Model output shape: {shape}")
    if len(shape) == 3:
        _, dim1, dim2 = shape
        # E2E format: last dimension is exactly 6 [x1,y1,x2,y2,conf,cls]
        if isinstance(dim2, int) and dim2 == 6:  # some models can export dim2 as None
            logger.info("Detected end-to-end (E2E) ONNX format [1, N, 6] — NMS is embedded in model.")
            return FORMAT_E2E
        # Standard format: second dim is the attribute count (4 box coords +
        # at least 1 class score, so >= 5; 84 for 80-class COCO), last dim is
        # the candidate pool size (e.g. 8400 or 15120)
        if isinstance(dim1, int) and isinstance(dim2, int) and dim1 >= 5 and dim2 > dim1:  # some models can export dim1/dim2 as None
            logger.info("Detected standard ONNX format [1, attrs, N] — will apply confidence filtering + NMS.")
            return FORMAT_STANDARD
    raise RuntimeError(
        f"Unrecognised ONNX output shape {shape}. "
        f"Expected [1, 5+, N] (standard) or [1, N, 6] (end-to-end). "
        f"Re-export your model or check your Ultralytics export options."
    )


# Post-processing — shared tail
def boxes_to_predictions(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    img_shape: tuple,
    ratio: float,
    pad: tuple,
    names: dict
) -> list:
    """
    Shared post-processing tail: map inference-space [x_min, y_min, x_max, y_max]
    float boxes back to original image space and build the response dicts.
    Steps:
      1. Remove letterbox padding and rescale to original image dimensions.
         pad is (left, top) — the integer pixel offsets returned by letterbox().
      2. Clamp to image boundaries. Clamping uses w_orig / h_orig (not
         w_orig-1 / h_orig-1) because x_max and y_max are exclusive boundaries —
         a valid box touching the right or bottom edge of the image should have
         x_max == w_orig, not w_orig-1. Clamping to w_orig-1 would cause those
         boxes to fail the degenerate-box filter.
      3. Discard degenerate boxes (zero-area after int truncation).
      4. Build response dicts — Python loop unavoidable for named-key output.
    Mutates boxes_xyxy in place; callers must pass a buffer they own.
    """
    h_orig, w_orig = img_shape[:2]

    # Remove letterbox padding and rescale to original image dimensions.
    # pad[0] = left offset (applied to x), pad[1] = top offset (applied to y).
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad[0]) / ratio
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad[1]) / ratio

    # Clamp to image boundaries.
    # x_min / y_min clamp to 0; x_max clamps to w_orig; y_max clamps to h_orig.
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, w_orig)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, h_orig)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, w_orig)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, h_orig)
    boxes_int = boxes_xyxy.astype(np.int32)

    # Discard degenerate boxes (zero-area after int truncation)
    valid     = (boxes_int[:, 2] > boxes_int[:, 0]) & (boxes_int[:, 3] > boxes_int[:, 1])
    boxes_int = boxes_int[valid]
    scores    = scores[valid]
    class_ids = class_ids[valid]

    predictions = []
    for box, score, cls in zip(boxes_int, scores, class_ids):
        cls_id = int(cls)
        label  = names.get(cls_id, f"class_{cls_id}")
        predictions.append({
            "confidence": round(float(score), 4),
            "label":      label,
            "x_min":      int(box[0]),
            "y_min":      int(box[1]),
            "x_max":      int(box[2]),
            "y_max":      int(box[3])
        })
    return predictions


# Post-processing — end-to-end (E2E) format
def postprocess_e2e(
    output_raw: np.ndarray,
    img_shape: tuple,
    ratio: float,
    pad: tuple,
    min_confidence: float,
    names: dict
) -> list:
    """
    Post-process an end-to-end [1, N, 6] ONNX model output.
    The model has already performed NMS internally. Each row contains:
      [x_min, y_min, x_max, y_max, confidence, class_id]
    Coordinates are in inference-image space (IMG_SZ_X × IMG_SZ_Y);
    boxes_to_predictions() maps them back to original image space.
    """
    detections = output_raw[0]  # [N, 6]

    # Vectorized confidence filter. The explicit > 0.0 term eliminates NMS
    # zero-padding rows even when the caller passes min_confidence == 0.0.
    conf = detections[:, 4]
    mask = (conf > 0.0) & (conf >= min_confidence)
    filtered = detections[mask]
    if not filtered.size:
        return []

    return boxes_to_predictions(filtered[:, :4].copy(), filtered[:, 4], filtered[:, 5],
                                img_shape, ratio, pad, names)


# Post-processing — standard format
def postprocess_standard(
    output_raw: np.ndarray,
    img_shape: tuple,
    ratio: float,
    pad: tuple,
    min_confidence: float,
    names: dict
) -> list:
    """
    Post-process a standard [1, attrs, N] ONNX model output.
    Steps:
      1. Transpose [1, attrs, N] -> [N, attrs]
      2. Vectorized confidence-threshold filter
      3. Convert [cx, cy, w, h] -> [x_min, y_min, w, h] for OpenCV NMS
      4. Single-pass per-class NMS (cv2.dnn.NMSBoxes with class offsets)
      5. Convert survivors to [x_min, y_min, x_max, y_max]
      6. Map back to original image space via boxes_to_predictions()
    pad is (left, top) — the integer pixel offsets returned by letterbox().
    """
    # [1, attrs, N] -> [N, attrs]
    output    = output_raw[0].T

    # Compute max class score and winning class for every candidate box
    raw_boxes = output[:, :4]
    num_attrs = output.shape[1]

    # Detect whether objectness score exists.
    # 85+ attrs = YOLOv5-style with objectness; 84 = YOLOv8-style without.
    # take_along_axis reuses the argmax result instead of a second full scan.
    if num_attrs >= 85:
        obj_conf    = output[:, 4]
        class_probs = output[:, 5:]
        class_ids   = np.argmax(class_probs, axis=1)
        max_scores  = obj_conf * np.take_along_axis(class_probs, class_ids[:, None], axis=1)[:, 0]
    else:
        class_probs = output[:, 4:]
        class_ids   = np.argmax(class_probs, axis=1)
        max_scores  = np.take_along_axis(class_probs, class_ids[:, None], axis=1)[:, 0]

    # Confidence filter — drop low-confidence candidates before NMS
    mask = max_scores >= min_confidence
    if not np.any(mask):
        return []

    filtered_boxes  = raw_boxes[mask]
    filtered_scores = max_scores[mask].astype(np.float32)
    class_ids       = class_ids[mask]

    # Convert [cx, cy, w, h] -> [x_min, y_min, w, h] for cv2.dnn.NMSBoxes
    x_c, y_c, w_box, h_box = filtered_boxes.T
    x_min = x_c - (w_box / 2.0)
    y_min = y_c - (h_box / 2.0)

    # Single-pass per-class NMS: shift each class's boxes by an offset larger
    # than any possible coordinate so boxes of different classes can never
    # overlap, letting one NMSBoxes call replace a per-class Python loop.
    offset    = class_ids.astype(np.float32) * (4.0 * max(IMG_SZ_X, IMG_SZ_Y))
    nms_boxes = np.stack([x_min + offset, y_min + offset, w_box, h_box], axis=1).astype(np.float32)
    nms_keep  = cv2.dnn.NMSBoxes(nms_boxes, filtered_scores, 0.0, NMS_IOU)
    if len(nms_keep) == 0:
        return []
    keep_idx = np.asarray(nms_keep, dtype=int).flatten()

    # Restore global confidence ordering
    order    = filtered_scores[keep_idx].argsort()[::-1]
    keep_idx = keep_idx[order]

    # Convert kept boxes from [cx, cy, w, h] directly to [x_min, y_min, x_max, y_max]
    # in a single step, avoiding intermediate final_boxes copy + re-conversion.
    kept = filtered_boxes[keep_idx]
    boxes_xyxy = np.empty((len(kept), 4), dtype=np.float32)
    boxes_xyxy[:, 0] = kept[:, 0] - kept[:, 2] / 2   # x_min
    boxes_xyxy[:, 1] = kept[:, 1] - kept[:, 3] / 2   # y_min
    boxes_xyxy[:, 2] = kept[:, 0] + kept[:, 2] / 2   # x_max
    boxes_xyxy[:, 3] = kept[:, 1] + kept[:, 3] / 2   # y_max

    return boxes_to_predictions(boxes_xyxy, filtered_scores[keep_idx], class_ids[keep_idx],
                                img_shape, ratio, pad, names)


# Synchronous inference dispatcher
def run_inference_sync(app: FastAPI, img: np.ndarray, min_confidence: float) -> tuple:
    """
    Run the full inference pipeline synchronously.
    Intended to be called via asyncio.to_thread() to avoid blocking the event loop.
    Pipeline:
      1. Letterbox: resize with aspect-ratio preservation, pad to IMG_SZ_X × IMG_SZ_Y
      2. Build the input blob directly: BGR->RGB channel reorder into a
         preallocated NCHW float32 buffer, then one in-place /255 normalize
         (no per-channel temporaries, no redundant second resize).
      3. Run ONNX session (thread-safe; no lock required)
      4. Route to appropriate post-processor with ratio + pad for coordinate inversion
    Timing:
      infer_ms    — GPU/CPU kernel time only (session.run)
      pipeline_ms — full pipeline including pre/post-processing
    Returns: (predictions, infer_ms, pipeline_ms)
    """
    start_pipeline = time.perf_counter()

    padded_img, ratio, pad = letterbox(img, new_shape=(IMG_SZ_X, IMG_SZ_Y))

    # Per-call buffer — must not be shared across concurrent requests (data race).
    # uint8 channels are cast to float32 on assignment; the single in-place
    # divide normalizes all channels without per-channel temporaries.
    buf = np.empty((1, 3, IMG_SZ_Y, IMG_SZ_X), dtype=np.float32)
    buf[0, 0] = padded_img[:, :, 2]  # R
    buf[0, 1] = padded_img[:, :, 1]  # G
    buf[0, 2] = padded_img[:, :, 0]  # B
    buf /= 255.0

    start_infer = time.perf_counter()
    outputs = app.state.session.run(None, {app.state.input_name: buf})
    infer_ms = int((time.perf_counter() - start_infer) * 1000)
    if app.state.output_format == FORMAT_E2E:
        predictions = postprocess_e2e(outputs[0], img.shape, ratio, pad, min_confidence, app.state.names)
    else:
        predictions = postprocess_standard(outputs[0], img.shape, ratio, pad, min_confidence, app.state.names)
    pipeline_ms = int((time.perf_counter() - start_pipeline) * 1000)
    return predictions, infer_ms, pipeline_ms

# FastAPI lifespan — model loading, warm-up, and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Simple YOLO ONNX worker starting")
    BASE_DIR   = Path(__file__).resolve().parent
    MODEL_PATH = BASE_DIR / "models" / SERVER_MODEL

    # Determine best available execution provider
    providers = build_providers(BASE_DIR)
    # Bound concurrent inference so request bursts queue instead of
    # oversubscribing the GPU/CPU with parallel session.run calls.
    app.state.infer_sem = asyncio.Semaphore(INFER_CONCURRENCY)

    try:
        app.state.session    = ort.InferenceSession(str(MODEL_PATH), providers=providers)
        app.state.input_name = app.state.session.get_inputs()[0].name

        # Detect output format at startup — raises on unrecognised shape so
        # a wrong model file fails loudly rather than producing silent garbage.
        app.state.output_format = detect_output_format(app.state.session)

        # Extract class names from ONNX metadata.
        # Ultralytics embeds names as a string-repr dict on export, e.g.:
        #   "{'0': 'person', '1': 'bicycle', ...}"
        # Keys may be strings or ints depending on exporter version;
        # we normalise to int keys for consistent lookup in postprocessors.
        meta_dict = app.state.session.get_modelmeta().custom_metadata_map
        if "names" in meta_dict:
            try:
                raw_names = ast.literal_eval(meta_dict["names"])
                if isinstance(raw_names, dict):
                    app.state.names = {int(k): str(v) for k, v in raw_names.items()}
                elif isinstance(raw_names, list):
                    app.state.names = {i: str(v) for i, v in enumerate(raw_names)}
                else:
                    raise ValueError("Unsupported names metadata format")
                logger.info(f"Loaded {len(app.state.names)} class names from ONNX metadata.")
            except Exception as e:
                logger.warning(f"Failed to parse ONNX metadata names: {e}")
                app.state.names = {}
        else:
            logger.warning("No class names found in ONNX metadata — outputting integer class IDs.")
            app.state.names = {}

    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")

    # Log which execution provider was actually selected
    provider_used = (
        app.state.session.get_providers()[0]
        if app.state.session.get_providers()
        else "CPUExecutionProvider"
    )
    if provider_used == "TensorrtExecutionProvider":
        logger.info(f"Running model {SERVER_MODEL} @ {IMG_SZ_X}x{IMG_SZ_Y}px on TensorRT"
                    f" (FP16={'on' if TRT_FP16 else 'off'})")
        # Warn about first-run engine compilation only when no cached engine exists.
        # ONNX Runtime's TensorRT EP writes cache files with an .engine extension;
        # once compiled, they are reused on all subsequent starts.
        trt_cache = TRT_CACHE_DIR if TRT_CACHE_DIR else str(BASE_DIR / "models")
        if not any(Path(trt_cache).glob("*.engine")):
            logger.warning("No TensorRT engine cache found — first run will compile the engine "
                           "(may take several minutes). Subsequent starts will use the cache.")
    elif provider_used == "CUDAExecutionProvider":
        logger.info(f"Running model {SERVER_MODEL} @ {IMG_SZ_X}x{IMG_SZ_Y}px on CUDA GPU")
    elif provider_used == "CoreMLExecutionProvider":
        logger.info(f"Running model {SERVER_MODEL} @ {IMG_SZ_X}x{IMG_SZ_Y}px on CoreML (ANE/GPU/CPU)")
    else:
        logger.warning(f"No accelerated execution provider available — running on CPU ({get_cpu_name()}). "
                        "Check your onnxruntime build and CUDA/cuDNN installation if acceleration was expected.")

    logger.info("Performing warm-up runs...")
    # An all-zeros blob is sufficient to trigger kernel compilation / caching.
    warmup_buf = np.zeros((1, 3, IMG_SZ_Y, IMG_SZ_X), dtype=np.float32)
    for _ in range(WARMUP_RUNS):
        app.state.session.run(None, {app.state.input_name: warmup_buf})
    logger.info(f"Simple YOLO ONNX server {VERSION} ready and listening on {SERVER_LISTEN}:{SERVER_PORT}")

    # Run
    yield

    # Shutdown
    logger.info("Simple YOLO ONNX worker exiting")


# FastAPI application
app = FastAPI(title="Simple YOLO Inference Server", version=VERSION, lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Lightweight health check — returns server status and version."""
    return {"status": "ok", "version": VERSION}


@app.post("/v1/vision/detection")
async def detect(
    request:        Request,
    image:          UploadFile = File(None),
    min_confidence: float      = Form(MINIMUM_CONFIDENCE)
) -> dict:
    """
    DeepStack / CodeProject.AI compatible object detection endpoint.
    Accepts images via:
      - Multipart form-data: 'image' file field + optional 'min_confidence' float field
      - JSON body: {"image": "<base64>", "min_confidence": 0.65}
                   base64 may include a data URI prefix (data:image/...;base64,)
    Returns:
      {
        "success":     true,
        "count":       <int>,
        "inferenceMs": <int>,   -- GPU/CPU kernel time only
        "totalMs":     <int>,   -- full pipeline including pre/post-processing
        "predictions": [
          {"confidence": float, "label": str,
           "x_min": int, "y_min": int, "x_max": int, "y_max": int},
          ...
        ]
      }
    """
    # Guard: model must be loaded before accepting requests
    if not hasattr(request.app.state, 'session'):
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate form-supplied min_confidence before doing any further work
    if not (0.0 <= min_confidence <= 1.0):
        raise HTTPException(status_code=400, detail="min_confidence must be between 0.0 and 1.0")

    # Reject oversized payloads by header before buffering the body.
    # The 1.4 factor allows for base64 (+33%) and JSON/multipart framing
    # overhead. Bodies without a Content-Length are still caught by the
    # per-path MAX_IMAGE_BYTES checks below.
    content_length = request.headers.get("content-length", "")
    if content_length.isdigit() and int(content_length) > int(MAX_IMAGE_BYTES * 1.4):
        raise HTTPException(status_code=413, detail="Request body too large")

    raw_data = None
    # Image ingestion — multipart/form-data
    if image is not None:
        try:
            raw_data = await image.read()
            if not raw_data or len(raw_data) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=400, detail="Invalid or oversized image")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"File read error: {e}")

    # Image ingestion — application/json with base64 payload.
    # JSON parsing and base64 decoding of a multi-MB body are CPU-bound, so
    # both run off the event loop in a single thread hop.
    elif "application/json" in request.headers.get("content-type", ""):
        try:
            body = await request.body()
            raw_data, min_confidence = await asyncio.to_thread(
                parse_json_request, body, min_confidence
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"JSON decode error: {e}")

    # No valid image source found
    else:
        raise HTTPException(status_code=400, detail="No image provided / Invalid Content-Type")

    # Image decode (offloaded to thread to avoid blocking the event loop)
    try:
        img = await asyncio.to_thread(decode_image, raw_data)
        if img is None:
            raise ValueError("Unsupported or corrupt image format")
        if img.shape[0] * img.shape[1] > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=400, detail="Image dimensions too large")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")

    # Inference (offloaded to thread; ONNX Runtime session.run is thread-safe).
    # The semaphore bounds concurrent inference to INFER_CONCURRENCY.
    async with request.app.state.infer_sem:
        predictions, infer_ms, pipeline_ms = await asyncio.to_thread(
            run_inference_sync, request.app, img, min_confidence
        )
    return {
        "success":     True,
        "count":       len(predictions),
        "inferenceMs": infer_ms,
        "totalMs":     pipeline_ms,
        "predictions": predictions
    }


# Entry point — direct execution via `python main.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=SERVER_LISTEN, port=SERVER_PORT, log_level=UVICORN_LOG.lower(), timeout_keep_alive=KEEPALIVE_TIMEOUT)
