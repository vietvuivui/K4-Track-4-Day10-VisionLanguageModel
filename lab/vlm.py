"""Bước 3–7: nạp Qwen3-VL-4B 4-bit, dựng ví dụ huấn luyện, setup LoRA/đóng băng, train, suy luận."""
import json
import math
import random
import re
import time
from pathlib import Path

import torch
from PIL import Image

from lab.labels import build_prompt, objects_to_answer
from scripts.labcommon import read_jsonl

# Tên module lấy từ cây model Qwen3-VL-4B (in [n for n, _ in model.named_modules()] để xác nhận).
LLM_TARGETS = (r".*language_model\.layers\.\d+\."
               r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)")
VIT_TARGETS = r".*visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.linear_fc[12])"
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")
EXPECTED_MODULES = {"llm": 252, "llm+vit": 348}
IMAGE_PAD = "<|image_pad|>"
EXTRA_FILE = "extra_trainable.pt"
SEQ_PAD = {"input_ids": None, "attention_mask": 0, "mm_token_type_ids": 0, "labels": -100}


# ---------------------------------------------------------------- nạp model (bước 3)

def load_model(model_id="Qwen/Qwen3-VL-4B-Instruct"):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # giữ ViT (kể cả projector) và lm_head ở fp16 để projector huấn luyện đầy đủ được.
        # transformers 5.x so khớp theo tên đầy đủ nên ghi cả "model.visual".
        llm_int8_skip_modules=["visual", "model.visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    quantized = quantized_visual_modules(model)
    if quantized:
        print(f"CẢNH BÁO: {len(quantized)} Linear trong ViT vẫn bị lượng tử hoá 4-bit "
              f"(vd. {quantized[0]}); giải lượng tử về fp16 để projector huấn luyện đầy đủ được")
        dequantize_modules(model, quantized)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


def quantized_visual_modules(model):
    return [n for n, m in model.named_modules()
            if "visual" in n.split(".") and type(m).__name__ == "Linear4bit"]


def dequantize_modules(model, names):
    """Thay Linear4bit bằng nn.Linear fp16 (trọng số giải lượng tử từ NF4)."""
    import bitsandbytes.functional as bnbf

    for name in names:
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        old = getattr(parent, child)
        w = bnbf.dequantize_4bit(old.weight.data, old.weight.quant_state).to(torch.float16)
        new = torch.nn.Linear(old.in_features, old.out_features, bias=old.bias is not None,
                              device=w.device, dtype=torch.float16)
        with torch.no_grad():
            new.weight.copy_(w.reshape(new.weight.shape))
            if old.bias is not None:
                new.bias.copy_(old.bias.to(torch.float16))
        setattr(parent, child, new)
    left = quantized_visual_modules(model)
    if left:
        raise RuntimeError(f"ViT vẫn còn {len(left)} module 4-bit, vd. {left[0]}")


def describe_architecture(model_id="Qwen/Qwen3-VL-4B-Instruct"):
    """Đếm tham số từng khối trên meta device (không tải trọng số, không cần GPU) — câu 1."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    config = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)
    groups = {"vit_blocks+patch_embed": 0, "visual.merger": 0,
              "visual.deepstack_merger_list": 0, "llm (language_model)": 0, "lm_head": 0, "other": 0}
    for name, p in model.named_parameters():
        if ".visual.merger" in name:
            groups["visual.merger"] += p.numel()
        elif ".visual.deepstack_merger_list" in name:
            groups["visual.deepstack_merger_list"] += p.numel()
        elif ".visual." in name:
            groups["vit_blocks+patch_embed"] += p.numel()
        elif ".language_model." in name:
            groups["llm (language_model)"] += p.numel()
        elif name.startswith("lm_head"):
            groups["lm_head"] += p.numel()
        else:
            groups["other"] += p.numel()
    groups["total"] = sum(p.numel() for p in model.parameters())
    names = [n for n, _ in model.named_modules()]
    matched = {k: sum(1 for n in names if re.fullmatch(lora_regex(k), n))
               for k in EXPECTED_MODULES}
    return {"params": groups, "lora_modules": matched,
            "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", False)
                                        or getattr(config.text_config, "tie_word_embeddings", False))}


# ---------------------------------------------------------------- dữ liệu (bước 4)

def load_split(data_dir, split, fraction=1.0, seed=42, limit=None):
    entries = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")
    if fraction < 1.0:
        k = max(1, round(len(entries) * fraction))
        entries = sorted(random.Random(seed).sample(entries, k), key=lambda e: e["image_id"])
    return entries[:limit] if limit else entries


def load_image(data_dir, entry):
    with Image.open(Path(data_dir) / entry["image_path"]) as im:
        return im.convert("RGB")


def user_message(image, prompt):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


def encode_example(processor, image, prompt, answer):
    user = user_message(image, prompt)
    assistant = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt_enc = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    full_enc = processor.apply_chat_template([user, assistant], add_generation_prompt=False, **kw)
    n = prompt_enc["input_ids"].shape[1]
    if not torch.equal(full_enc["input_ids"][0, :n], prompt_enc["input_ids"][0]):
        raise ValueError("prompt không phải tiền tố của chuỗi đầy đủ: kiểm tra chat template")
    out = {k: v for k, v in full_enc.items() if k != "token_type_ids"}
    labels = full_enc["input_ids"].clone()
    labels[:, :n] = -100            # chỉ học phần trả lời (+ <|im_end|>)
    out["labels"] = labels
    return out


def collate(batch, pad_id):
    """Pad phải các tensor theo chuỗi; pixel_values/image_grid_thw nối theo chiều 0, không pad."""
    max_len = max(b["input_ids"].shape[1] for b in batch)
    out = {}
    for key, pad in SEQ_PAD.items():
        if key not in batch[0]:
            continue
        pad = pad_id if pad is None else pad
        rows = []
        for b in batch:
            t = b[key]
            rows.append(torch.cat([t, t.new_full((t.shape[0], max_len - t.shape[1]), pad)], dim=1))
        out[key] = torch.cat(rows, dim=0)
    for key in ("pixel_values", "image_grid_thw"):
        out[key] = torch.cat([b[key] for b in batch], dim=0)
    return out


def image_token_count(processor, input_ids):
    return int((input_ids == processor.tokenizer.convert_tokens_to_ids(IMAGE_PAD)).sum())


def make_example(processor, data_dir, entry, cfg):
    coord = cfg["coord_scale"]
    prompt = build_prompt(coord, entry["width"], entry["height"])
    answer = objects_to_answer(entry["objects"], coord, entry["width"], entry["height"])
    return encode_example(processor, load_image(data_dir, entry), prompt, answer)


# ---------------------------------------------------------------- LoRA / đóng băng (bước 5)

def lora_regex(targets):
    if targets == "llm":
        return LLM_TARGETS
    if targets == "llm+vit":
        return f"(?:{LLM_TARGETS})|(?:{VIT_TARGETS})"
    raise ValueError(f"lora.targets không hợp lệ: {targets}")


def is_projector(name):
    return any(k in name for k in PROJECTOR_KEYS)


def apply_setup(model, cfg):
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()      # cần cho gradient checkpointing
    lora = cfg["lora"]
    regex = lora_regex(lora["targets"])
    matched = [n for n, _ in model.named_modules() if re.fullmatch(regex, n)]
    expected = EXPECTED_MODULES[lora["targets"]]
    print(f"LoRA targets={lora['targets']}: {len(matched)} module (kỳ vọng {expected})")
    if not matched:
        raise RuntimeError("regex LoRA không khớp module nào: in model.named_modules() để kiểm tra")
    if len(matched) != expected:
        print("CẢNH BÁO: số module LoRA khác kỳ vọng, kiểm tra lại tên module")
    peft_model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], target_modules=regex))
    n_proj = 0
    if cfg["train_projector"]:
        for name, p in peft_model.named_parameters():
            if is_projector(name):
                p.requires_grad = True
                n_proj += p.numel()
    # master weight fp32 (LoRA + projector): GradScaler không unscale được gradient fp16,
    # và bước cập nhật nhỏ bị làm tròn mất nếu cộng vào trọng số fp16
    for p in peft_model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"tham số huấn luyện: {trainable:,} (projector {n_proj:,}) / tổng {total:,}")
    return peft_model, {"matched_modules": len(matched), "trainable_params": trainable,
                        "projector_params": n_proj}


def projector_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if p.requires_grad and is_projector(n)}


def load_projector(model, path):
    state = torch.load(path, map_location="cpu")
    params = dict(model.named_parameters())
    missing = [n for n in state if n not in params]
    if missing:
        raise KeyError(f"{len(missing)} tham số projector không khớp tên, vd. {missing[0]}")
    with torch.no_grad():
        for n, t in state.items():
            params[n].copy_(t.to(params[n].dtype))
    print(f"đã nạp lại {len(state)} tensor projector từ {path}")


# ---------------------------------------------------------------- huấn luyện (bước 6)

def _batches(entries, batch_size, seed):
    rng = random.Random(seed)
    while True:
        order = entries[:]
        rng.shuffle(order)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            yield order[i:i + batch_size]


def _lr_lambda(max_steps):
    warmup = max(1, int(0.05 * max_steps))

    def fn(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, max_steps - warmup)))
    return fn


def _save_checkpoint(path, model, opt, sched, scaler, step, losses, minutes, peak):
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    tmp = Path(str(path) + ".tmp")
    torch.save({"step": step, "trainable": trainable, "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                "losses": losses, "minutes": minutes, "peak_vram_gb": peak}, tmp)
    tmp.replace(path)


def train(cfg, out_dir, data_dir="data", resume=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    model, processor = load_model(cfg["model_id"])
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model, info = apply_setup(model, cfg)
    device = next(model.parameters()).device

    entries = load_split(data_dir, "train", float(cfg["train_fraction"]), cfg["seed"])
    print(f"train: {len(entries)} ảnh (fraction={cfg['train_fraction']}), coord={cfg['coord_scale']}")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(cfg["max_steps"]))
    scaler = torch.amp.GradScaler("cuda")

    ckpt = out_dir / "checkpoint.pt"
    start, losses, prev_minutes, prev_peak = 0, [], 0.0, 0.0
    if resume and ckpt.exists():
        st = torch.load(ckpt, map_location="cpu")
        named = dict(model.named_parameters())
        with torch.no_grad():
            for n, t in st["trainable"].items():
                named[n].copy_(t.to(named[n].device, named[n].dtype))
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start, losses = st["step"], st["losses"]
        prev_minutes, prev_peak = st["minutes"], st["peak_vram_gb"]
        print(f"tiếp tục từ bước {start}")

    it = _batches(entries, cfg["batch_size"], cfg["seed"] + start)
    pad_id = processor.tokenizer.pad_token_id
    torch.cuda.reset_peak_memory_stats()
    model.train()
    t0 = time.time()
    ga = cfg["grad_accum"]
    for step in range(start, cfg["max_steps"]):
        step_loss = 0.0
        for _ in range(ga):
            batch = collate([make_example(processor, data_dir, e, cfg) for e in next(it)], pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / ga
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss không hữu hạn ở bước {step}: nguy cơ tràn số fp16")
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        losses.append(round(step_loss, 5))
        done = step + 1
        if done % 10 == 0 or done == 1:
            print(f"bước {done}/{cfg['max_steps']} loss={step_loss:.4f} grad_norm={grad_norm:.2f} "
                  f"lr={sched.get_last_lr()[0]:.2e} {(time.time() - t0) / 60:.1f} phút")
        if done % cfg["save_every"] == 0 or done == cfg["max_steps"]:
            minutes = prev_minutes + (time.time() - t0) / 60
            peak = max(prev_peak, torch.cuda.max_memory_allocated() / 1024 ** 3)
            model.save_pretrained(out_dir / "adapter")      # chỉ lưu adapter LoRA
            if cfg["train_projector"]:                      # projector không nằm trong adapter
                torch.save(projector_state(model), out_dir / EXTRA_FILE)
            _save_checkpoint(ckpt, model, opt, sched, scaler, done, losses, minutes, peak)
            print(f"đã lưu checkpoint bước {done} vào {out_dir}")

    minutes = prev_minutes + (time.time() - t0) / 60
    peak = max(prev_peak, torch.cuda.max_memory_allocated() / 1024 ** 3)
    run = {"name": cfg["name"], "trainable_params": info["trainable_params"],
           "projector_params": info["projector_params"], "matched_modules": info["matched_modules"],
           "peak_vram_gb": round(peak, 2), "train_minutes": round(minutes, 1),
           "n_train_images": len(entries), "losses": losses, "config": cfg}
    (out_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"xong: peak_vram={run['peak_vram_gb']} GB, {run['train_minutes']} phút")
    return run


# ---------------------------------------------------------------- suy luận (bước 3, 7)

def load_for_inference(cfg, out_dir=None):
    model, processor = load_model(cfg["model_id"])
    if cfg.get("lora"):
        from peft import PeftModel

        adapter = Path(out_dir) / "adapter"
        if not adapter.exists():
            raise FileNotFoundError(f"không thấy adapter ở {adapter}: chạy train trước")
        model = PeftModel.from_pretrained(model, adapter)
        if cfg["train_projector"]:
            extra = Path(out_dir) / EXTRA_FILE
            if not extra.exists():
                raise FileNotFoundError(f"thiếu {extra}: E2/E3 sẽ trông giống E1 nếu không nạp")
            load_projector(model, extra)
    model.eval()
    return model, processor


@torch.inference_mode()
def generate(model, processor, images, prompts, max_new_tokens):
    msgs = [[user_message(im, pr)] for im, pr in zip(images, prompts)]
    inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def predict(cfg, out_dir, pred_path, data_dir="data", split="test", limit=None, model=None,
            processor=None):
    """Ghi {"image_id","raw"} mỗi dòng; chạy lại sẽ bỏ qua ảnh đã có (an toàn khi mất session)."""
    pred_path = Path(pred_path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    entries = load_split(data_dir, split, limit=limit)
    done = {r["image_id"] for r in read_jsonl(pred_path)} if pred_path.exists() else set()
    todo = [e for e in entries if e["image_id"] not in done]
    print(f"{split}: {len(entries)} ảnh, đã có {len(done)}, còn {len(todo)}")
    if not todo:
        return pred_path
    if model is None:
        model, processor = load_for_inference(cfg, out_dir)
    bs, t0 = cfg["eval_batch_size"], time.time()
    with open(pred_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            images = [load_image(data_dir, e) for e in chunk]
            prompts = [build_prompt(cfg["coord_scale"], e["width"], e["height"]) for e in chunk]
            raws = generate(model, processor, images, prompts, cfg["max_new_tokens"])
            for e, raw in zip(chunk, raws):
                f.write(json.dumps({"image_id": e["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            if (i // bs) % 10 == 0:
                print(f"  {i + len(chunk)}/{len(todo)} ảnh, {(time.time() - t0) / 60:.1f} phút")
    return pred_path
