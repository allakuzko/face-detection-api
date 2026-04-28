# Face Detection API 🦀

A high-performance face detection REST API built with Rust + YOLOv8 ONNX Runtime.

## Stack
- **Rust** + **axum** — async web server
- **ort** — ONNX Runtime bindings
- **image** — image decoding and preprocessing
- **Model** — YOLOv8n fine-tuned for face detection

## Features
- ✅ Face detection from uploaded images
- ✅ Returns bounding boxes with confidence scores
- ✅ Supports JPEG, PNG, WebP formats
- ✅ Inference time logging
- ✅ Health check

## Setup

### 1. Download ONNX Runtime
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-1.24.2.tgz
tar -xzf onnxruntime-linux-x64-1.24.2.tgz
sudo cp onnxruntime-linux-x64-1.24.2/lib/* /usr/local/lib/
sudo ldconfig
```

### 2. Download and convert model
```bash
pip install huggingface_hub ultralytics onnx
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('arnabdhar/YOLOv8-Face-Detection', local_dir='model')
"
python3 -c "
from ultralytics import YOLO
model = YOLO('model/model.pt')
model.export(format='onnx', imgsz=640)
"
```

### 3. Run
```bash
export ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so.1.24.2
cargo run --release
```

## API

### `POST /detect`
Upload an image and detect faces.

**Request:** `multipart/form-data` with field `image`

```bash
curl -X POST http://localhost:3000/detect \
  -F "image=@photo.jpg"
```

**Response:**
```json
{
  "faces": [
    {
      "x": 120.5,
      "y": 45.3,
      "width": 80.2,
      "height": 90.1,
      "confidence": 0.98
    },
    {
      "x": 310.1,
      "y": 60.7,
      "width": 75.4,
      "height": 85.9,
      "confidence": 0.95
    }
  ],
  "total_faces": 2,
  "inference_ms": 15
}
```

### `GET /health`
Health check.

**Response:**
```json
{"status": "ok", "version": "0.1.0"}
```

## Error Handling
All errors return a JSON response:
```json
{"error": "Failed to decode image: unsupported format"}
```
