# =============================================
# app.py
# Privacy-First Image Sanitizer (Faces + QR + EXIF scrub)
# ---------------------------------------------
# UI mode:     streamlit run app.py -- --ui
# CLI mode:    python app.py --input path/to/img.jpg --output sanitized.png --method blur [--yolo --weights weights/yolov8n-face.pt]
# Self-tests:  python app.py --selftest
# Notes:       Running without arguments shows CLI usage help only.
# Deps (pip):  opencv-python pillow numpy (streamlit optional for UI)
# Optional:    ultralytics  (for YOLOv8 face/person detection)
# =============================================
import io
import os
import argparse
from typing import List, Tuple, Iterable, Optional

import cv2
import numpy as np
from PIL import Image

# ---------------------------
# Core helpers (pure Python)
# ---------------------------

def drop_exif(pil_img: Image.Image) -> Image.Image:
    data = list(pil_img.getdata())
    img_no_exif = Image.new(pil_img.mode, pil_img.size)
    img_no_exif.putdata(data)
    return img_no_exif


def cv2_from_pil(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def pil_from_cv2(img_cv: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def blur_region(img: np.ndarray, x: int, y: int, w: int, h: int, method: str = "blur") -> None:
    H, W = img.shape[:2]
    if w <= 0 or h <= 0:
        return
    margin = max(2, min(w, h) // 10)
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(W, x + w + margin)
    y1 = min(H, y + h + margin)

    roi_ext = img[y0:y1, x0:x1]
    if roi_ext.size == 0:
        return

    if method == "pixelate":
        h_small = max(1, (y1 - y0) // 12)
        w_small = max(1, (x1 - x0) // 12)
        roi_small = cv2.resize(roi_ext, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
        roi_pix = cv2.resize(roi_small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        img[y0:y1, x0:x1] = roi_pix
    else:
        k = max(15, (w + h) // 20)
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(roi_ext, (k, k), 0)
        img[y0:y1, x0:x1] = blurred


# ---------------------------
# Classical detectors
# ---------------------------

def detect_faces(img_gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        return []
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def detect_qr(img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    det = cv2.QRCodeDetector()
    # Try newer signature first
    try:
        retval, decoded_info, points, _ = det.detectAndDecodeMulti(img_bgr)
        boxes: List[Tuple[int, int, int, int]] = []
        if retval and points is not None:
            for box in points:
                pts = box.astype(int).reshape(-1, 2)
                x, y, w, h = cv2.boundingRect(pts)
                boxes.append((x, y, w, h))
        return boxes
    except TypeError:
        pass
    except Exception:
        return []
    # Fallback older signature
    try:
        bbox, decoded_info, _ = det.detectAndDecodeMulti(img_bgr)
        boxes: List[Tuple[int, int, int, int]] = []
        if bbox is not None:
            for box in bbox:
                pts = box.astype(int).reshape(-1, 2)
                x, y, w, h = cv2.boundingRect(pts)
                boxes.append((x, y, w, h))
        return boxes
    except Exception:
        return []


# ---------------------------
# YOLOv8 (optional) detectors
# ---------------------------

def detect_yolo_faces(img_bgr: np.ndarray, weights_path: Optional[str] = None, conf: float = 0.25) -> List[Tuple[int, int, int, int]]:
    """Detect faces using YOLOv8 face model if available.

    - If ultralytics is not installed or weights are missing, return [].
    - Recommended weights: weights/yolov8n-face.pt (you provide file).
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        return []

    # Choose weights
    if weights_path and os.path.exists(weights_path):
        w = weights_path
    elif os.path.exists("weights/yolov8n-face.pt"):
        w = "weights/yolov8n-face.pt"
    else:
        return []

    try:
        model = YOLO(w)
        # BGR -> RGB for Ultralytics
        results = model.predict(img_bgr[:, :, ::-1], imgsz=640, conf=conf, verbose=False)
        boxes: List[Tuple[int, int, int, int]] = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes.xyxy.cpu().numpy().astype(int):
                x1, y1, x2, y2 = b[:4]
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                boxes.append((x, y, w, h))
        return boxes
    except Exception:
        return []


# ---------------------------
# Utils
# ---------------------------

def parse_manual_boxes(spec: str) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    if not spec:
        return boxes
    for line in spec.replace(";", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            x, y, w, h = map(int, parts)
            boxes.append((x, y, w, h))
        except Exception:
            continue
    return boxes


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a, area_b = aw * ah, bw * bh
    return inter / float(area_a + area_b - inter + 1e-6)


def merge_boxes(boxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.55) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        keep = True
        for i, ob in enumerate(out):
            if iou(b, ob) > iou_thresh:
                # merge by union
                x = min(b[0], ob[0])
                y = min(b[1], ob[1])
                x2 = max(b[0] + b[2], ob[0] + ob[2])
                y2 = max(b[1] + b[3], ob[1] + ob[3])
                out[i] = (x, y, x2 - x, y2 - y)
                keep = False
                break
        if keep:
            out.append(b)
    return out


# ---------------------------
# Sanitization pipeline
# ---------------------------

def sanitize_image(
    pil_img: Image.Image,
    method: str = "blur",
    auto_detect: bool = True,
    manual_boxes: Iterable[Tuple[int, int, int, int]] = (),
    use_yolo: bool = False,
    yolo_weights: Optional[str] = None,
) -> Tuple[Image.Image, List[Tuple[int, int, int, int, str]]]:
    pil_no_exif = drop_exif(pil_img.convert("RGB"))
    img_bgr = cv2_from_pil(pil_no_exif)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    boxes_labeled: List[Tuple[int, int, int, int, str]] = []
    if auto_detect:
        # Classical
        faces = detect_faces(img_gray)
        qrs = detect_qr(img_bgr)
        for (x, y, w, h) in faces:
            boxes_labeled.append((x, y, w, h, "face"))
        for (x, y, w, h) in qrs:
            boxes_labeled.append((x, y, w, h, "qr"))
        # YOLO
        if use_yolo:
            yboxes = detect_yolo_faces(img_bgr, weights_path=yolo_weights)
            for (x, y, w, h) in yboxes:
                boxes_labeled.append((x, y, w, h, "yolo_face"))

    for (x, y, w, h) in manual_boxes:
        boxes_labeled.append((x, y, w, h, "manual"))

    # Merge overlapping boxes to avoid over-blur with different detectors
    merged: List[Tuple[int, int, int, int, str]] = []
    raw_rects = [(x, y, w, h) for (x, y, w, h, _) in boxes_labeled]
    raw_rects = merge_boxes(raw_rects, iou_thresh=0.6)
    for (x, y, w, h) in raw_rects:
        merged.append((x, y, w, h, "merged"))

    sanitized = img_bgr.copy()
    for (x, y, w, h, _) in merged:
        blur_region(sanitized, x, y, w, h, method)

    return pil_from_cv2(sanitized), merged


# --------------------------------------
# Streamlit UI
# --------------------------------------

def run_streamlit_app() -> None:
    try:
        import streamlit as st
    except Exception as e:
        raise SystemExit("Streamlit not installed. Install with 'pip install streamlit'." f" Original: {e}")

    st.set_page_config(page_title="Privacy-First Image Sanitizer", layout="wide")
    st.title("Privacy-First Image Sanitizer (Track 7 Demo)")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        method = st.radio("Masking method", ["blur", "pixelate"], horizontal=True)
        auto_run = st.checkbox("Auto-detect (faces + QR)", value=True)
        use_yolo = st.checkbox("Use YOLOv8 face (if available)", value=False)
        yolo_weights = st.text_input("YOLO weights path (optional)", value="weights/yolov8n-face.pt")

    manual_note = st.expander("Optional: Add manual boxes")
    with manual_note:
        manual_boxes_text = st.text_area("Manual boxes (x,y,w,h)", value="")

    if uploaded:
        pil = Image.open(uploaded).convert("RGB")
        manual_boxes = parse_manual_boxes(manual_boxes_text)

        pil_no_exif = drop_exif(pil)
        img_bgr = cv2_from_pil(pil_no_exif)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        boxes_preview: List[Tuple[int, int, int, int, str]] = []
        if auto_run:
            boxes_preview.extend([(x, y, w, h, "face") for (x, y, w, h) in detect_faces(img_gray)])
            boxes_preview.extend([(x, y, w, h, "qr") for (x, y, w, h) in detect_qr(img_bgr)])
            if use_yolo:
                for (x, y, w, h) in detect_yolo_faces(img_bgr, weights_path=yolo_weights):
                    boxes_preview.append((x, y, w, h, "yolo_face"))
        for (x, y, w, h) in manual_boxes:
            boxes_preview.append((x, y, w, h, "manual"))

        # merge for preview also
        merged_preview = merge_boxes([(x, y, w, h) for (x, y, w, h, _) in boxes_preview], iou_thresh=0.6)
        preview = img_bgr.copy()
        for (x, y, w, h) in merged_preview:
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(preview, "det", (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        with col_left:
            st.subheader("Preview")
            st.image(pil_from_cv2(preview), use_container_width=True)

        sanitized_pil, used = sanitize_image(
            pil,
            method=method,
            auto_detect=auto_run,
            manual_boxes=manual_boxes,
            use_yolo=use_yolo,
            yolo_weights=yolo_weights if yolo_weights.strip() else None,
        )

        with col_right:
            st.subheader("Sanitized Output")
            st.image(sanitized_pil, use_container_width=True)
            buf = io.BytesIO()
            sanitized_pil.save(buf, format="PNG")
            st.download_button("Download sanitized image", data=buf.getvalue(), file_name="sanitized.png", mime="image/png")
            if len(used) == 0:
                st.warning("No detections or manual boxes. Exported unchanged (EXIF stripped).")
    else:
        with col_left:
            st.info("Upload an image to begin.")


# -------------------------
# CLI
# -------------------------

CLI_USAGE = ("\nUsage (CLI)\n"
    "  python app.py --input path/to/input.jpg --output sanitized.png --method blur [--yolo --weights weights/yolov8n-face.pt]\n\n"
    "Options\n"
    "  --input    Path to input image (required)\n"
    "  --output   Path to output PNG file (default: sanitized.png)\n"
    "  --method   Masking method: blur or pixelate (default: blur)\n"
    "  --no-auto  Disable auto detection\n"
    "  --boxes    Manual boxes: 'x,y,w,h; x,y,w,h'\n"
    "  --yolo     Use YOLOv8 face detection if ultralytics + weights are available\n"
    "  --weights  Path to YOLO weights (default: weights/yolov8n-face.pt)\n\n"
)

def run_cli(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Privacy-First Image Sanitizer (CLI)", add_help=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="sanitized.png")
    parser.add_argument("--method", choices=["blur", "pixelate"], default="blur")
    parser.add_argument("--no-auto", action="store_true")
    parser.add_argument("--boxes", default="")
    parser.add_argument("--yolo", action="store_true", help="Use YOLOv8 face detection")
    parser.add_argument("--weights", default="weights/yolov8n-face.pt", help="YOLO weights path")
    if argv is None:
        argv = os.sys.argv[1:]
    if not argv:
        print(CLI_USAGE)
        return
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        print(CLI_USAGE)
        return
    pil = Image.open(args.input).convert("RGB")
    manual_boxes = parse_manual_boxes(args.boxes)
    sanitized, used = sanitize_image(
        pil,
        method=args.method,
        auto_detect=not args.no_auto,
        manual_boxes=manual_boxes,
        use_yolo=args.yolo,
        yolo_weights=args.weights,
    )
    sanitized.save(args.output, format="PNG")
    if len(used) == 0:
        print("Warning: no detections or manual boxes. Exported unchanged (EXIF stripped).")
    print(f"Saved sanitized image to: {args.output}")


# -------------------------
# Self-tests
# -------------------------

def _make_striped_img(size: int = 100, box: Tuple[int, int, int, int] = (20, 20, 40, 40)):
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    x, y, w, h = box
    for i in range(y, y + h):
        for j in range(x, x + w):
            if (j - x) % 4 < 2:
                img[i, j] = np.array([0, 0, 0], dtype=np.uint8)
    return img, box


def test_blur_region_reduces_variance():
    img, (x, y, w, h) = _make_striped_img()
    before_var = np.var(img[y:y + h, x:x + w].astype(np.float32))
    blur_region(img, x, y, w, h, method="blur")
    after_var = np.var(img[y:y + h, x:x + w].astype(np.float32))
    assert after_var < before_var


def test_pixelate_changes_region():
    img, (x, y, w, h) = _make_striped_img()
    before = img[y:y + h, x:x + w].copy()
    blur_region(img, x, y, w, h, method="pixelate")
    after = img[y:y + h, x:x + w]
    assert np.any(before != after)


def test_sanitize_image_manual_box_changes_pixels():
    base = np.full((80, 80, 3), 255, dtype=np.uint8)
    base[20:60, 20:60] = np.array([10, 200, 10], dtype=np.uint8)
    pil = Image.fromarray(base[:, :, ::-1])
    out_pil, boxes = sanitize_image(pil, method="blur", auto_detect=False, manual_boxes=[(20, 20, 40, 40)])
    out = np.array(out_pil)[:, :, ::-1]
    assert len(boxes) == 1 and boxes[0][4] == "manual"
    assert np.any(out[20:60, 20:60] != base[20:60, 20:60])


def test_parse_manual_boxes_variants():
    spec = "10,20,30,40;  5,6,7,8\n50,60,70,80"
    boxes = parse_manual_boxes(spec)
    assert (10, 20, 30, 40) in boxes and (5, 6, 7, 8) in boxes and (50, 60, 70, 80) in boxes


def test_no_auto_no_boxes_keeps_image_same():
    base = np.full((40, 40, 3), 123, dtype=np.uint8)
    pil = Image.fromarray(base[:, :, ::-1])
    out_pil, boxes = sanitize_image(pil, method="blur", auto_detect=False, manual_boxes=[])
    out = np.array(out_pil)[:, :, ::-1]
    assert boxes == []
    assert np.array_equal(out, base)


def test_drop_exif_preserves_pixels():
    base = np.zeros((10, 10, 3), dtype=np.uint8)
    base[2:8, 2:8] = np.array([200, 50, 10], dtype=np.uint8)
    pil = Image.fromarray(base[:, :, ::-1])
    no_exif = drop_exif(pil)
    out = np.array(no_exif)[:, :, ::-1]
    assert np.array_equal(out, base)


def test_run_cli_no_args_does_not_exit():
    run_cli(argv=[])


def run_selftests() -> None:
    tests = [
        test_blur_region_reduces_variance,
        test_pixelate_changes_region,
        test_sanitize_image_manual_box_changes_pixels,
        test_parse_manual_boxes_variants,
        test_no_auto_no_boxes_keeps_image_same,
        test_drop_exif_preserves_pixels,
        test_run_cli_no_args_does_not_exit,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"[OK] {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {t.__name__}: {e}")
    if failures:
        raise SystemExit(f"Selftests failed: {failures} test(s)")
    print("All selftests passed.")


if __name__ == "__main__":
    import sys
    clean_args = [arg for arg in sys.argv[1:] if not arg.startswith('--server.')]
    if "--selftest" in clean_args:
        run_selftests()
    elif "--ui" in clean_args or len(clean_args) == 0 or any('--server.' in arg for arg in sys.argv):
        run_streamlit_app()
    else:
        run_cli()
