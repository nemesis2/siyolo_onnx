# siyolo_onnx: A Simple YOLO Inference Server (ONNX version)

## Overview

A lightweight, locally-hosted Ultralytics YOLO inference server written in Python using FastAPI.

siyolo was designed as a drop-in replacement for DeepStack and CodeProject.AI,  
providing fast local object detection without heavyweight AI frameworks or runtime environments.

Simple, fast and locally-hosted. No Docker, no .NET needed, just Python.  
A single-process, LAN-focused inference server for local object detection workloads.
This is the ONNX version, smaller venv footprint and allows running newer models on older hardware.

---

* Suitable for NVR and home automation
* Minimal dependencies (no .NET or Docker required)
* Fast startup with CUDA warm-up
* Supports TensorRT, CoreML, CPU and CUDA (Linux and Windows)
* Defensive filtering of NaN / invalid detections
* DeepStack-compatible REST API
* Optimized for low-memory environments
* Supports multipart/form-data and application/json (base64)

    
---

## Linux Installation & Setup Guide

### Install from git

```
cd /opt
sudo git clone https://github.com/nemesis2/siyolo_onnx.git
```

### Directory Setup

Create the model directory:
```
sudo mkdir -p /opt/siyolo_onnx/models
cd /opt/siyolo_onnx
```


### Install Python Virtual Environment

⚠ Linux: Use Python 3.10 for older CPUs/Maxwell GPUs. Windows can use Python 3.10–3.12.
```   
python3 -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip
```

### Python Package Requirements

⚠ Linux: The correct combination may vary depending on CPU/GPU and distribution.
 
```
pip install -r requirements.txt
```

### Create System User

```
sudo useradd -r siyolo
sudo chown -R siyolo:siyolo /opt/siyolo_onnx
```

### Configure systemd Service 

Create /opt/siyolo_onnx/siyolo.service:

```
[Unit]
Description=A Simple YOLO Inference Server
After=network.target

[Service]
Type=simple
User=siyolo
Group=siyolo
WorkingDirectory=/opt/siyolo_onnx
ExecStart=/opt/siyolo_onnx/venv/bin/python3.10 -u /opt/siyolo_onnx/main.py
Restart=always
RestartSec=5

# Environment
Environment=TMPDIR=/dev/shm
Environment=PYTHONUNBUFFERED=1
Environment=YOLO_MODEL=yolo26x.pt

# Optional limits
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

### Testing the Server

Activate venv and run manually first:

```
source /opt/siyolo_onnx/venv/bin/activate 
python3.10 main.py
```
Expected output should be similar to:

```
[INFO] Simple YOLO ONNX worker starting
[INFO] Model output shape: [1, 300, 6]
[INFO] Detected end-to-end (E2E) ONNX format [1, N, 6] — NMS is embedded in model.
[INFO] Loaded 80 class names from ONNX metadata.
[INFO] Running model yolo26x.onnx @ 1152x640px on CUDA GPU
[INFO] Performing warm-up runs...
[INFO] Simple YOLO ONNX server v2.1-onnx ready and listening on 127.0.0.1:32168
```

### Verify Inferencing

Test image: https://deepstack.readthedocs.io/en/latest/_images/family-and-dog.jpg

```
curl -s -X POST -F 'image=@family-and-dog.jpg' 'http://127.0.0.1:32168/v1/vision/detection' | jq '.'
```

Should return something like:
 
```jsonc
{
  "success": true,
  "count": 3,
  "inferenceMs": 74,
  "totalMs": 98,
  "predictions": [
    {
      "confidence": 0.9401,
      "label": "person",
      "x_min": 294,
      "y_min": 85,
      "x_max": 442,
      "y_max": 519
    },
    {
      "confidence": 0.9371,
      "label": "dog",
      "x_min": 650,
      "y_min": 344,
      "x_max": 793,
      "y_max": 540
    },
    {
      "confidence": 0.9249,
      "label": "person",
      "x_min": 443,
      "y_min": 113,
      "x_max": 601,
      "y_max": 523
    }
  ]
}
```


### Enable and start:

```
sudo ln -s /opt/siyolo_onnx/siyolo.service /etc/systemd/system/siyolo_onnx.service
sudo systemctl daemon-reload
sudo systemctl enable siyolo
sudo systemctl start siyolo
sudo systemctl status siyolo
```

---

## Notes

⚠ This server does not implement authentication, rate limiting or request throttling. Do not expose directly to the public Internet. ⚠ 

* Default inference size: 1152x640 (configurable in main.py; NOTE: requires new ONNX model port!)
* Place OONX models in /opt/siyolo_onnx/models
