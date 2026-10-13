"""Bước 9 (E5): reward IoU kiểm chứng được cho GRPO. Python thuần, test được không cần GPU."""
from scripts.score import IOU_THR, iou_xyxy, parse_output

VALID_BONUS = 0.2


def completion_text(completion):
    """TRL đưa completion dạng chuỗi hoặc dạng hội thoại [{"role", "content"}]."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        completion = [completion]
    parts = []
    for msg in completion or []:
        content = msg.get("content") if isinstance(msg, dict) else msg
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts += [c.get("text", "") for c in content if isinstance(c, dict)]
    return "".join(parts)


def gt_boxes(objects):
    """objects dạng split (yolo chuẩn hoá) → [{"label","bbox"}] trên thang 0–1000, như score.py."""
    out = []
    for o in objects:
        xc, yc, w, h = o["yolo"]
        out.append({"label": o["label"], "bbox": (
            (xc - w / 2) * 1000, (yc - h / 2) * 1000, (xc + w / 2) * 1000, (yc + h / 2) * 1000)})
    return out


def iou_reward(completion, gt):
    """0 nếu JSON hỏng; 1 nếu cả gt và pred rỗng; ngược lại 0.2 + 0.8·ΣIoU_khớp / max(n_gt, n_pred)."""
    pred = parse_output(completion_text(completion))
    if pred is None:
        return 0.0
    if not gt and not pred:
        return 1.0
    pairs = sorted(((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                    for i, g in enumerate(gt) for j, p in enumerate(pred)
                    if g["label"] == p["label"]), reverse=True)
    used_g, used_p, total = set(), set(), 0.0
    for v, i, j in pairs:
        if v < IOU_THR:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        total += v
    return VALID_BONUS + (1 - VALID_BONUS) * total / max(len(gt), len(pred), 1)
