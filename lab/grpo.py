"""Bước 9 (E5, tuỳ chọn): GRPO với reward IoU, khởi tạo từ adapter SFT tốt nhất.

  python -m lab.grpo --base-config configs/e2.yaml --base-out OUT/e2 --out OUT/e5
  python -m lab.run predict --config configs/e2.yaml --out OUT/e5
  python -m lab.run score --config configs/e2.yaml --out OUT/e5 --name E5 --evidence report/evidence/e5.json

Đã viết theo API GRPOTrainer cho VLM (cột "prompt" dạng hội thoại + cột "image"). trl được ghim
1.9.2: nếu tham số nào của GRPOConfig bị đổi tên, đọc docs đúng phiên bản và sửa tại đây.
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch
import yaml

from lab.config import load_config
from lab.labels import build_prompt
from lab.reward import gt_boxes, iou_reward
from lab.vlm import EXTRA_FILE, load_image, load_model, load_projector, load_split


def build_dataset(data_dir, n_images, seed):
    from datasets import Dataset

    entries = load_split(data_dir, "train", fraction=1.0, seed=seed)
    entries = random.Random(seed).sample(entries, min(n_images, len(entries)))
    rows = {"prompt": [], "image": [], "gt": []}
    for e in entries:
        rows["prompt"].append([{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": build_prompt("norm1000", e["width"], e["height"])}]}])
        rows["image"].append(load_image(data_dir, e))
        rows["gt"].append(json.dumps(gt_boxes(e["objects"])))
    return Dataset.from_dict(rows)


def reward_iou(completions, gt, **kwargs):
    return [iou_reward(c, [{"label": b["label"], "bbox": tuple(b["bbox"])} for b in json.loads(g)])
            for c, g in zip(completions, gt)]


def main(argv=None):
    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-config", required=True, help="config của setup SFT tốt nhất")
    ap.add_argument("--base-out", required=True, help="thư mục chứa adapter/ (và projector) của setup đó")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grpo-config", default="configs/e5.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n-images", type=int, default=480)
    a = ap.parse_args(argv)

    cfg = load_config(a.base_config)
    g = yaml.safe_load(Path(a.grpo_config).read_text(encoding="utf-8"))["grpo"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(cfg["model_id"])
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, Path(a.base_out) / "adapter", is_trainable=True)
    if cfg["train_projector"]:                  # projector giữ nguyên (đóng băng) trong GRPO
        load_projector(model, Path(a.base_out) / EXTRA_FILE)
        shutil.copy(Path(a.base_out) / EXTRA_FILE, out / EXTRA_FILE)
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n

    args = GRPOConfig(
        output_dir=str(out / "trainer"), learning_rate=g["learning_rate"], beta=g["beta"],
        num_generations=g["num_generations"], max_completion_length=g["max_completion_length"],
        max_steps=g["max_steps"], per_device_train_batch_size=g["num_generations"],
        gradient_accumulation_steps=1, fp16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, use_vllm=False,
        logging_steps=1, save_strategy="no", report_to="none", seed=cfg["seed"],
        remove_unused_columns=False)
    trainer = GRPOTrainer(model=model, processing_class=processor, reward_funcs=[reward_iou],
                          args=args, train_dataset=build_dataset(a.data, a.n_images, cfg["seed"]))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    model.save_pretrained(out / "adapter")
    log = {"log_history": trainer.state.log_history,
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
           "train_minutes": round((time.time() - t0) / 60, 1), "base": a.base_out, "grpo": g}
    (out / "grpo_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    (out / "run.json").write_text(json.dumps({
        "trainable_params": trainable, "peak_vram_gb": log["peak_vram_gb"],
        "train_minutes": log["train_minutes"]}, indent=2), encoding="utf-8")
    print(f"xong GRPO: log ở {out / 'grpo_log.json'}")


if __name__ == "__main__":
    main()
