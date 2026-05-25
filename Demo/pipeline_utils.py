import cv2
import numpy as np
import base64
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

yolo_model = None
cls_model = None

CLASS_NAMES = {0: "glass", 1: "metal", 2: "paper", 3: "plastic"}
IMG_SIZE = 224

def init_models():
    global yolo_model, cls_model
    if yolo_model is None:
        print("Đang tải YOLO model...")
        yolo_model = YOLO("../runs11s_50/detect/train/weights/best.pt")
    if cls_model is None:
        print("Đang tải Classification model...")
        cls_model = load_model("../garbage_classifier_MobileNet (2).keras")


def preprocess_image(img):

    img_uint8 = img.astype(np.uint8)

    img_uint8 = cv2.resize(img_uint8, (IMG_SIZE, IMG_SIZE))

    denoised = cv2.bilateralFilter(
        img_uint8,
        d=3,
        sigmaColor=40,
        sigmaSpace=40
    )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.1,
        tileGridSize=(8,8)
    )

    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))

    contrast_img = cv2.cvtColor(
        lab_eq,
        cv2.COLOR_LAB2BGR
    )

    final_img = contrast_img.astype(np.float32)

    final_img = preprocess_input(final_img)

    return final_img

def denoise(img):
    return cv2.bilateralFilter(img, 5, 60, 60)

def segment_grabcut(img):
    h, w = img.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    bgModel = np.zeros((1,65), np.float64)
    fgModel = np.zeros((1,65), np.float64)

    rect = (int(0.05*w), int(0.05*h), int(0.9*w), int(0.9*h))

    try:
        cv2.grabCut(img, mask, rect, bgModel, fgModel, 10, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

        if np.sum(mask2) < 0.1 * h * w:
            mask2 = np.ones((h, w), dtype=np.uint8)

    except:
        mask2 = np.ones((h, w), dtype=np.uint8)

    return mask2 * 255

def refine_mask(mask):
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))

    return cv2.Canny(gray, lower, upper)

def img_to_base64(img, is_rgb=True):
    if len(img.shape) == 3 and is_rgb:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img
        
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

def pad_replicate(img, pad=200):

    padded = cv2.copyMakeBorder(
        img,
        top=pad,
        bottom=pad,
        left=pad,
        right=pad,
        borderType=cv2.BORDER_REPLICATE
    )

    return padded

def process_detection_and_pipeline(image_bytes):
    init_models()
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image")

    results = yolo_model.predict(frame, conf=0.1,iou=0.45,agnostic_nms=True, verbose=False)
    target_crops = []
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                padded_crop = pad_replicate(
                    cropped
                )

                target_crops.append(padded_crop.copy())
                processed_img = preprocess_image(cropped)
                processed_img = np.expand_dims(processed_img, axis=0)
                preds = cls_model.predict(processed_img, verbose=0)
                class_idx = np.argmax(preds[0])
                predicted_class = CLASS_NAMES.get(class_idx, "Unknown")
                

                try:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    (tw, th), _ = cv2.getTextSize(predicted_class, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1), (x1 + tw + 10, y1 + th + 15), (0, 255, 0), -1)
                    cv2.putText(frame, predicted_class, (x1 + 5, y1 + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                except:
                    pass

    output_yolo_b64 = img_to_base64(frame, is_rgb=False)

    pipelines = []

    if len(target_crops) > 0:
        for idx, target_crop in enumerate(target_crops):
            img_rgb = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (224, 224))
            
            blur = denoise(img_rgb)
            mask = segment_grabcut(blur)
            mask = refine_mask(mask)
            segmented = apply_mask(blur, mask)
            edges = edge_detection(segmented)
            
            pipelines.append({
                "title": f"Vật thể {idx + 1}",
                "Original": img_to_base64(img_rgb),
                "Denoise": img_to_base64(blur),
                "Segmented": img_to_base64(segmented),
                "Edges": img_to_base64(edges, is_rgb=False)
            })
    else:
        img_rgb = cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (224, 224))
        
        blur = denoise(img_rgb)
        mask = segment_grabcut(blur)
        mask = refine_mask(mask)
        segmented = apply_mask(blur, mask)
        edges = edge_detection(segmented)

        pipelines.append({
            "title": "Toàn bộ ảnh (Không nhận diện được vật thể)",
            "Original": img_to_base64(img_rgb),
            "Denoise": img_to_base64(blur),
            "Segmented": img_to_base64(segmented),
            "Edges": img_to_base64(edges, is_rgb=False)
        })

    return {
        "yolo_result": output_yolo_b64,
        "pipelines": pipelines
    }
