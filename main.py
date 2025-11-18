import sys
import numpy as np
import cv2
from ultralytics import YOLO
import easyocr
import re
from collections import defaultdict, deque

# ----------------- CONFIG -----------------
INPUT_VIDEO = "vehicle.mp4"
OUTPUT_VIDEO = "output_video.mp4"
MODEL_PATH = "license_plate_best.pt"
CONF_THRESH = 0.3

# Display settings (only affects the on-screen preview, not the saved video)
DISPLAY_SCALE = 1.0     # 1.0 = full size, <1.0 = scale down for display only (e.g. 0.6)
DISPLAY_WINDOW_NAME = "Annotated Video"
# ------------------------------------------

# --- model / OCR ---
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)

plate_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")

def correct_plate_format(ocr_text):
    mapping_num_to_alpha = {'0':'O','1':'I','5':'S','8':'B'}
    mapping_alpha_to_num = {'O':'0','I':'1','Z':'2','S':'5','B':'8'}

    ocr_text = ocr_text.upper().replace(" ","")
    if len(ocr_text) != 7:
        return ""

    corrected = []
    for i, ch in enumerate(ocr_text):
        if i < 2 or i >= 4:
            if ch.isdigit() and ch in mapping_num_to_alpha:
                corrected.append(mapping_num_to_alpha[ch])
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""
        else:
            if ch.isalpha() and ch in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[ch])
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""
    return "".join(corrected)

def recognize_plate(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return ""

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    try:
        ocr_result = reader.readtext(plate_resized, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        if len(ocr_result) > 0:
            candidate = correct_plate_format(ocr_result[0])
            if candidate and plate_pattern.match(candidate):
                return candidate
    except Exception:
        pass
    return ""

# history for stabilizing plate reads
plate_history = defaultdict(lambda: deque(maxlen=10))
plate_final = {}

def get_box_id(x1,y1,x2,y2):
    return f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"

def get_stable_plate(box_id, new_text):
    if new_text:
        plate_history[box_id].append(new_text)
        most_common = max(set(plate_history[box_id]), key=plate_history[box_id].count)
        plate_final[box_id] = most_common
    return plate_final.get(box_id, "")

# ----------------- I/O Setup -----------------
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"ERROR: Could not open input video: {INPUT_VIDEO}")
    sys.exit(1)

# Read the first frame to get accurate dimensions
ret, first_frame = cap.read()
if not ret:
    print("ERROR: Could not read first frame from video.")
    cap.release()
    sys.exit(1)

frame_h, frame_w = first_frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 25.0

print(f"Detected video resolution: {frame_w}x{frame_h}, FPS: {fps}")

# Create VideoWriter using the real frame size
fourcc = cv2.VideoWriter_fourcc(*'avc1')   # H.264 / AVC

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_w, frame_h))
if not out.isOpened():
    print("ERROR: Could not open VideoWriter. Check codec/filename.")
    cap.release()
    sys.exit(1)

# We will *attempt* to show frames. If a GUI error occurs, we disable live display.
show_enabled = True
window_created = False

# prepare display window (we'll try; if it fails we'll catch)
try:
    cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
    disp_w = int(frame_w * DISPLAY_SCALE)
    disp_h = int(frame_h * DISPLAY_SCALE)
    try:
        cv2.resizeWindow(DISPLAY_WINDOW_NAME, disp_w, disp_h)
    except Exception:
        pass
    window_created = True
    print("INFO: Created display window (trying to show frames).")
except Exception as e:
    show_enabled = False
    print("WARN: Could not create display window. Live display disabled. Exception:", repr(e))

CONF_THRESH = CONF_THRESH

def process_and_show(frame):
    results = model(frame, verbose=False)

    for r in results:
        boxes = getattr(r, "boxes", [])
        for box in boxes:
            try:
                if hasattr(box.conf, "cpu"):
                    conf = float(box.conf.cpu().item())
                else:
                    conf = float(box.conf)
            except Exception:
                try:
                    conf = float(np.array(box.conf).item())
                except Exception:
                    continue

            if conf < CONF_THRESH:
                continue

            try:
                xy = box.xyxy.cpu().numpy() if hasattr(box.xyxy, "cpu") else np.array(box.xyxy)
            except Exception:
                try:
                    xy = np.array(box.xyxy)
                except Exception:
                    continue

            if xy.ndim > 1:
                xy = xy[0]
            x1, y1, x2, y2 = map(int, xy)

            x1 = max(0, min(x1, frame.shape[1]-1))
            x2 = max(0, min(x2, frame.shape[1]-1))
            y1 = max(0, min(y1, frame.shape[0]-1))
            y2 = max(0, min(y2, frame.shape[0]-1))
            if x2 <= x1 or y2 <= y1:
                continue

            plate_crop = frame[y1:y2, x1:x2]
            text = recognize_plate(plate_crop)
            box_id = get_box_id(x1, y1, x2, y2)
            stable_text = get_stable_plate(box_id, text)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)

            if plate_crop is not None and plate_crop.size > 0:
                overlay_h, overlay_w = 150, 400
                plate_resized = cv2.resize(plate_crop, (overlay_w, overlay_h))

                oy1 = max(0, y1 - overlay_h - 40)
                ox1 = x1
                oy2, ox2 = oy1 + overlay_h, ox1 + overlay_w

                if ox2 > frame.shape[1]:
                    ox1 = max(0, frame.shape[1] - overlay_w)
                    ox2 = ox1 + overlay_w
                if oy2 <= frame.shape[0] and ox2 <= frame.shape[1]:
                    frame[oy1:oy2, ox1:ox2] = plate_resized

                    if stable_text:
                        cv2.putText(frame, stable_text, (ox1, oy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 6)
                        cv2.putText(frame, stable_text, (ox1, oy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

    return frame

# Process first frame and show
annotated = process_and_show(first_frame.copy())
out.write(annotated)

# Save a debug snapshot so you can open it with the OS image viewer
try:
    cv2.imwrite("debug_first_frame.png", annotated)
    print("Saved debug_first_frame.png (inspect to confirm what should display).")
except Exception as e:
    print("Failed to write debug_first_frame.png:", repr(e))

if show_enabled and window_created:
    try:
        # print frame shape for debugging
        print("DEBUG: first annotated frame.shape =", annotated.shape)
        display_frame = annotated if DISPLAY_SCALE == 1.0 else cv2.resize(annotated, (int(frame_w*DISPLAY_SCALE), int(frame_h*DISPLAY_SCALE)))
        cv2.imshow(DISPLAY_WINDOW_NAME, display_frame)
        cv2.waitKey(1)  # allow GUI to update
    except Exception as e:
        show_enabled = False
        print("WARN: imshow failed on first frame, disabling live display. Exception:", repr(e))

frame_count = 1
# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated = process_and_show(frame)
    out.write(annotated)

    frame_count += 1
    # occasionally save a debug snapshot (every 200 frames) to inspect
    if frame_count % 200 == 0:
        try:
            cv2.imwrite(f"debug_frame_{frame_count}.png", annotated)
            print(f"Saved debug_frame_{frame_count}.png")
        except Exception:
            pass

    if show_enabled and window_created:
        try:
            display_frame = annotated if DISPLAY_SCALE == 1.0 else cv2.resize(annotated, (int(frame_w*DISPLAY_SCALE), int(frame_h*DISPLAY_SCALE)))
            cv2.imshow(DISPLAY_WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested quit.")
                break
        except Exception as e:
            show_enabled = False
            print("WARN: imshow failed during loop, disabling live display. Exception:", repr(e))
            # continue but do not attempt to show further frames

cap.release()
out.release()
try:
    cv2.destroyAllWindows()
except Exception:
    pass

print("Done. Annotated video saved as:", OUTPUT_VIDEO)
