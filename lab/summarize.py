"""Dựng bảng tổng hợp REPORT.md từ report/evidence/*.json (số lấy thẳng từ file bằng chứng).

  python -m lab.summarize            # in bảng
  python -m lab.summarize --write    # thay bảng giữa hai dấu <!-- TABLE --> trong REPORT.md
"""
import argparse
import json
import re
from pathlib import Path

SETUPS = {"e0": "zero-shot", "e1": "LoRA LLM", "e2": "+ projector", "e3": "+ LoRA ViT"}
START, END = "<!-- TABLE:START -->", "<!-- TABLE:END -->"
HEADER = ("| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh "
          "| F1 pothole | F1 crack | F1 manhole | parse_valid |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|")


def _order(stem):
    return (stem[:2], stem)


def row(stem, d):
    meta = d.get("meta", {})
    pc = d["per_class"]
    setup = SETUPS.get(stem, stem[3:].replace("_", " = ", 1) if stem.startswith("e4_") else d["name"])
    tp = meta.get("trainable_params", 0)
    return (f"| {d['name']} | {setup} | {tp:,} | {meta.get('peak_vram_gb', '—')} "
            f"| {meta.get('train_minutes', '—')} | {d['micro']['f1']:.3f} "
            f"| [{d['f1_ci95'][0]:.3f}, {d['f1_ci95'][1]:.3f}] | {d['fp_per_image']:.2f} "
            f"| {pc['pothole']['f1']:.3f} | {pc['crack']['f1']:.3f} | {pc['manhole']['f1']:.3f} "
            f"| {d['parse_valid_rate']:.3f} |")


def build_table(ev_dir="report/evidence"):
    files = sorted(Path(ev_dir).glob("e*.json"), key=lambda p: _order(p.stem))
    rows = [row(p.stem, json.loads(p.read_text(encoding="utf-8"))) for p in files]
    return HEADER + "\n" + "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence", default="report/evidence")
    ap.add_argument("--report", default="REPORT.md")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    table = build_table(a.evidence)
    print(table)
    if a.write:
        p = Path(a.report)
        text = p.read_text(encoding="utf-8")
        if START not in text:
            raise SystemExit(f"không thấy {START} trong {p}")
        text = re.sub(re.escape(START) + r".*?" + re.escape(END),
                      lambda _: f"{START}\n{table}\n{END}", text, flags=re.S)
        p.write_text(text, encoding="utf-8")
        print(f"đã cập nhật bảng trong {p}")


if __name__ == "__main__":
    main()
