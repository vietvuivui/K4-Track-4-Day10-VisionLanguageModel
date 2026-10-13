"""CLI chạy từng thí nghiệm. Mỗi lệnh là một tiến trình riêng nên GPU được giải phóng sau khi xong.

  python -m lab.run train   --config configs/e1.yaml --out OUT/e1 [--set lora.r=64] [--resume]
  python -m lab.run predict --config configs/e1.yaml --out OUT/e1 [--split val --limit 40]
  python -m lab.run score   --config configs/e1.yaml --out OUT/e1 --evidence report/evidence/e1.json
  python -m lab.run score   --config configs/e1.yaml --out OUT/e1 --split val --limit 40   # chấm nhanh
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from lab.config import load_config


def pred_file(out_dir, split, limit=None):
    return Path(out_dir) / (f"pred_{split}.jsonl" if not limit else f"pred_{split}_{limit}.jsonl")


def quick_score(cfg, out_dir, split, limit, data_dir="data", coord=None):
    """Chấm nhanh trên `limit` ảnh đầu của split để gỡ lỗi (không ghi bằng chứng)."""
    from scripts.labcommon import read_jsonl
    from scripts.score import evaluate

    entries = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")[:limit]
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(pred_file(out_dir, split, limit))}
    res = evaluate(entries, preds, coord=coord or cfg["coord_scale"], n_boot=200)
    m = res["micro"]
    print(f"[{split}:{limit}] valid={res['parse_valid_rate']:.3f} P={m['precision']:.3f} "
          f"R={m['recall']:.3f} F1={m['f1']:.3f} FP/img={res['fp_per_image']:.2f}")
    return res


def score(cfg, out_dir, evidence, split="test", limit=None, name=None, coord=None):
    cmd = [sys.executable, "scripts/score.py", "--pred", str(pred_file(out_dir, split, limit)),
           "--split", split, "--name", name or cfg["name"], "--out", str(evidence),
           "--coord", coord or cfg["coord_scale"]]
    run_json = Path(out_dir) / "run.json"
    if cfg.get("lora"):
        if not run_json.exists():
            sys.exit(f"thiếu {run_json}: cần meta trainable_params/peak_vram_gb/train_minutes")
        run = json.loads(run_json.read_text(encoding="utf-8"))
        for k in ("trainable_params", "peak_vram_gb", "train_minutes"):
            cmd += ["--meta", f"{k}={run[k]}"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "predict", "score"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="thư mục kết quả (nên đặt trên Google Drive)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--evidence", help="đường dẫn json do score.py ghi")
    ap.add_argument("--name", help="tên trong file bằng chứng (mặc định: name trong config)")
    ap.add_argument("--coord", choices=["norm1000", "pixel"], help="ghi đè --coord khi chấm")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    if a.cmd == "train":
        from lab import vlm
        if not cfg.get("lora"):
            sys.exit("config này không có lora (E0 là zero-shot, chỉ cần predict)")
        run_json = out / "run.json"
        if a.resume and run_json.exists():
            done = json.loads(run_json.read_text(encoding="utf-8"))
            if len(done.get("losses", [])) >= cfg["max_steps"]:
                print(f"{run_json} đã đủ {cfg['max_steps']} bước: bỏ qua train")
                return
        vlm.train(cfg, out, a.data, resume=a.resume)
    elif a.cmd == "predict":
        from lab import vlm
        vlm.predict(cfg, out, pred_file(out, a.split, a.limit), a.data, a.split, a.limit)
    elif a.limit:
        quick_score(cfg, out, a.split, a.limit, a.data, a.coord)
    else:
        if not a.evidence:
            sys.exit("score cần --evidence")
        score(cfg, out, a.evidence, a.split, a.limit, a.name, a.coord)


if __name__ == "__main__":
    main()
