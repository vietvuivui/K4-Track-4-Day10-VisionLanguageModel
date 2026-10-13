"""Bước 2: nhãn YOLO → JSON grounding (thang 0–1000 hoặc pixel) và prompt."""
import json

COORD_SCALES = ("norm1000", "pixel")

_PROMPT_HEAD = (
    "Locate every road damage in this dashcam image. "
    "Classes: pothole, crack, manhole. "
    "Answer with ONLY a JSON list and no other text, in the form "
    '[{"label": "<class>", "bbox_2d": [x1, y1, x2, y2]}]. '
)
_PROMPT_NORM1000 = (
    "Coordinates are integers on a 0-1000 scale relative to the image width and height "
    "(0 = left/top edge, 1000 = right/bottom edge). "
)
_PROMPT_PIXEL = (
    "Coordinates are integer pixel positions in the {w}x{h} image "
    "(x from 0 to {w}, y from 0 to {h}). "
)
_PROMPT_TAIL = "If there is no damage, answer []."


def _scale(coord, width, height):
    if coord == "norm1000":
        return 1000, 1000
    if coord == "pixel":
        return width, height
    raise ValueError(f"coord không hợp lệ: {coord!r} (chọn {COORD_SCALES})")


def build_prompt(coord="norm1000", width=640, height=360):
    _scale(coord, width, height)
    body = _PROMPT_NORM1000 if coord == "norm1000" else _PROMPT_PIXEL.format(w=width, h=height)
    return _PROMPT_HEAD + body + _PROMPT_TAIL


def _clamp(v, hi):
    return min(max(int(round(v)), 0), hi)


def _fix_degenerate(a, b, hi):
    """Box quá nhỏ có thể làm tròn thành x2 == x1; score.parse_output coi đó là JSON hỏng."""
    if b > a:
        return a, b
    if a < hi:
        return a, a + 1
    return hi - 1, hi


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    xc, yc, w, h = yolo
    sx, sy = _scale(coord, width, height)
    x1, x2 = _clamp((xc - w / 2) * sx, sx), _clamp((xc + w / 2) * sx, sx)
    y1, y2 = _clamp((yc - h / 2) * sy, sy), _clamp((yc + h / 2) * sy, sy)
    x1, x2 = _fix_degenerate(x1, x2, sx)
    y1, y2 = _fix_degenerate(y1, y2, sy)
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_boxes(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    # trên→dưới, trái→phải: thứ tự ổn định cho model học
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return boxes


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    return json.dumps(objects_to_boxes(objects, coord, width, height), separators=(",", ":"))


def box_to_pixel(bbox, coord="norm1000", width=640, height=360):
    """bbox [x1,y1,x2,y2] theo thang `coord` → toạ độ pixel (float) để vẽ."""
    sx, sy = _scale(coord, width, height)
    x1, y1, x2, y2 = bbox
    return (x1 / sx * width, y1 / sy * height, x2 / sx * width, y2 / sy * height)


def yolo_to_pixel(yolo, width=640, height=360):
    xc, yc, w, h = yolo
    return ((xc - w / 2) * width, (yc - h / 2) * height,
            (xc + w / 2) * width, (yc + h / 2) * height)
