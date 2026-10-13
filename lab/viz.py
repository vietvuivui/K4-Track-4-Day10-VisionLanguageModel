"""Vẽ box đúng (xanh) và box dự đoán (đỏ) lên ảnh; chọn ảnh dự đoán tốt/xấu để minh hoạ."""
from pathlib import Path

from PIL import Image, ImageDraw

from lab.labels import box_to_pixel, yolo_to_pixel
from lab.reward import gt_boxes
from scripts.labcommon import read_jsonl
from scripts.score import _match, _to_norm1000, parse_output

GT_COLOR, PRED_COLOR = (0, 220, 0), (255, 40, 40)


def draw(image, gt_objects=(), preds=(), coord="norm1000", caption=None):
    im = image.convert("RGB").copy()
    w, h = im.size
    d = ImageDraw.Draw(im)
    for o in gt_objects:
        box = yolo_to_pixel(o["yolo"], w, h)
        d.rectangle(box, outline=GT_COLOR, width=2)
        d.text((box[0] + 2, box[3] - 12), o["label"], fill=GT_COLOR)
    for p in preds:
        box = box_to_pixel(p["bbox"], coord, w, h)
        d.rectangle(box, outline=PRED_COLOR, width=2)
        d.text((box[0] + 2, box[1] + 1), p["label"], fill=PRED_COLOR)
    if caption:
        canvas = Image.new("RGB", (w, h + 22), (0, 0, 0))
        canvas.paste(im, (0, 0))
        ImageDraw.Draw(canvas).text((4, h + 5), caption, fill=(255, 255, 255))
        im = canvas
    return im


def image_stats(entries, preds, coord="norm1000"):
    """Mỗi ảnh: (entry, pred đã parse, tp, fp, fn) theo đúng luật ghép của score.py."""
    rows = []
    for e in entries:
        parsed = parse_output(preds.get(e["image_id"]))
        pred = _to_norm1000(parsed, coord, e["width"], e["height"]) if parsed is not None else []
        per, _ = _match(gt_boxes(e["objects"]), pred)
        tp, fp, fn = (sum(v[k] for v in per.values()) for k in range(3))
        rows.append((e, parsed, tp, fp, fn))
    return rows


def make_figures(pred_path, data_dir="data", split="test", out_dir="report/figures",
                 coord="norm1000", n_good=3, n_bad=3, tag="best"):
    entries = {e["image_id"]: e for e in read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")}
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(pred_path)}
    rows = image_stats([entries[i] for i in preds if i in entries], preds, coord)
    good = sorted((r for r in rows if r[2] > 0 and r[3] + r[4] == 0), key=lambda r: -r[2])[:n_good]
    bad = sorted(rows, key=lambda r: -(r[3] + r[4]))[:n_bad]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for kind, group in (("dung", good), ("sai", bad)):
        for k, (e, parsed, tp, fp, fn) in enumerate(group, 1):
            cap = f"{e['image_id']}  TP={tp} FP={fp} FN={fn}" + ("  (JSON hỏng)" if parsed is None else "")
            with Image.open(Path(data_dir) / e["image_path"]) as im:
                fig = draw(im, e["objects"], parsed or [], coord, cap)
            name = f"{tag}_{kind}_{k}.jpg"
            fig.save(out_dir / name, quality=90)
            lines.append(f"| ![]({name}) | {kind} | `{e['image_id']}` | {tp} | {fp} | {fn} |")
    md = ["| Ảnh | Loại | image_id | TP | FP | FN |", "|---|---|---|---|---|---|", *lines]
    (out_dir / f"{tag}_captions.md").write_text(
        "Xanh = box đúng (GT), đỏ = box dự đoán.\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    print(f"đã ghi {len(lines)} ảnh vào {out_dir}")
    return lines
