from pathlib import Path
import io

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

BASE = Path(__file__).parent
IMGSZ = 640
CONF_THRES = 0.25   # ultralytics default
IOU_THRES = 0.7     # ultralytics default NMS IoU

app = FastAPI(title="OCR YOLO ID Card API")

id_card_session = ort.InferenceSession(
    str(BASE / "detect_id_card.onnx"), providers=["CPUExecutionProvider"]
)
digits_session = ort.InferenceSession(
    str(BASE / "detect_arabic_numbers.onnx"), providers=["CPUExecutionProvider"]
)


def letterbox(image: Image.Image, size: int = IMGSZ):
    """Resize keeping aspect ratio and pad to size x size (YOLO-style)."""
    w, h = image.size
    r = min(size / w, size / h)
    new_w, new_h = round(w * r), round(h * r)
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None]  # (1, 3, H, W)
    return arr, r, pad_x, pad_y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Greedy non-max suppression. boxes: (N, 4) as x1, y1, x2, y2."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-7)
        order = rest[iou <= iou_thres]
    return keep


def detect(session: ort.InferenceSession, image: Image.Image) -> list[dict]:
    """Run a YOLOv8 ONNX model and return detections in original image coordinates."""
    inp, r, pad_x, pad_y = letterbox(image)
    output = session.run(None, {session.get_inputs()[0].name: inp})[0][0]  # (4 + nc, N)
    preds = output.T                                                        # (N, 4 + nc)

    xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    classes = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)

    mask = scores > CONF_THRES
    xywh, scores, classes = xywh[mask], scores[mask], classes[mask]
    if scores.size == 0:
        return []

    # center-xywh -> xyxy, then undo the letterbox transform
    cx, cy, w, h = xywh.T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / r

    W, H = image.size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)

    # per-class NMS: offset each class so boxes of different classes never overlap
    offsets = classes[:, None] * 10000.0
    keep = nms(boxes + offsets, scores, IOU_THRES)

    return [
        {"class": int(classes[i]), "box": boxes[i], "score": float(scores[i])}
        for i in keep
    ]


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    extracted_number = ""

    for card in detect(id_card_session, image):
        x1, y1, x2, y2 = map(int, card["box"])
        crop = image.crop((x1, y1, x2, y2))

        digits = detect(digits_session, crop)
        digits.sort(key=lambda d: d["box"][0])  # left to right
        extracted_number += "".join(str(d["class"]) for d in digits)

    return {"id_number": extracted_number}