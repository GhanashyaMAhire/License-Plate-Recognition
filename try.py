import sys
import numpy as np
import cv2
from ultralytics import YOLO
import easyocr
import re
from collections import defaultdict, deque

# --- model / OCR ---
model = YOLO('license_plate_best.pt')
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
    # plate_crop is a numpy array (BGR)
    if plate_crop is None or plate_crop.size == 0:
        return ""

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plate_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    try:
        # easyocr readtext: detail=0 returns list of strings
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

# --- I/O ---
input_video = "vehicle.mp4"
output_video = "output_video.mp4"   # <<--- ensure extension

cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"ERROR: Could not open input video: {input_video}")
    sys.exit(1)

# get properties safely
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or np.isnan(fps):
    fps = 25.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

# Define the codec and output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_w, frame_h))

if not out.isOpened():
    print("ERROR: Could not open VideoWriter. Check output filename and codec.")
    print("Wrote attempted filename:", output_video)
    cap.release()
    sys.exit(1)

# Attempt to detect whether imshow is usable.
# On headless builds, cv2.imshow exists but will raise — we use a heuristic:
can_show = True
try:
    # quick check: is OpenCV built with GUI support?
    build_info = cv2.getBuildInformation()
    # look for "GUI:                          YES" or absence of "headless"
    if "GUI:" in build_info and ("YES" not in build_info.split("GUI:")[-1]):
        can_show = False
    # also detect common headless: opencv-python-headless
    if "headless" in build_info.lower():
        can_show = False
except Exception:
    # If getBuildInformation fails, still try to show but be cautious
    can_show = False

CONF_THRESH = 0.3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for r in results:
        boxes = getattr(r, "boxes", [])
        for box in boxes:
            # safe extraction of confidence as scalar
            try:
                conf = float(box.conf.cpu().item()) if hasattr(box.conf, "cpu") else float(box.conf)
            except Exception:
                try:
                    conf = float(box.conf)
                except Exception:
                    continue

            if conf < CONF_THRESH:
                continue

            xy = box.xyxy.cpu().numpy() if hasattr(box.xyxy, "cpu") else np.array(box.xyxy)
            # some boxes may come as Nx4; take the first row
            if xy.ndim > 1:
                xy = xy[0]
            x1, y1, x2, y2 = map(int, xy)

            # clip coordinates
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

            cv2.rectangle(frame, (x1, y1), (x2,y2), (0,255,0), 3)

            if plate_crop is not None and plate_crop.size >0:
                overlay_h, overlay_w = 150, 400
                plate_resized = cv2.resize(plate_crop, (overlay_w, overlay_h))

                oy1 = max(0, y1 - overlay_h - 40)
                ox1 = x1
                oy2, ox2 = oy1 + overlay_h, ox1 + overlay_w

                # ensure overlay fits horizontally, adjust if necessary
                if ox2 > frame.shape[1]:
                    ox1 = max(0, frame.shape[1] - overlay_w)
                    ox2 = ox1 + overlay_w
                if oy2 <= frame.shape[0] and ox2 <= frame.shape[1]:
                    frame[oy1:oy2, ox1:ox2] = plate_resized

                    if stable_text:
                        cv2.putText(frame, stable_text, (ox1, oy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 6)
                        cv2.putText(frame, stable_text, (ox1, oy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)

    out.write(frame)

    # show if GUI is available
    if can_show:
        try:
            cv2.imshow("Annotated Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            # If imshow fails at runtime (headless...), disable it
            can_show = False

cap.release()
out.release()
if can_show:
    cv2.destroyAllWindows()
else:
    # ensure no leftover window calls error
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

print("Annotated video saved as:", output_video)
