import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import transformers
from transformers import BitsAndBytesConfig, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.data_processing import get_mask_from_json
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         dict_to_cuda)


def parse_args(args):
    parser = argparse.ArgumentParser(description="Detailed ReasonSeg benchmark for LISA")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--dataset_dir", default="./dataset")
    parser.add_argument("--val_dataset", default="ReasonSeg|val")
    parser.add_argument("--output_dir", default="./benchmark_outputs/reason_seg")
    parser.add_argument("--vision_pretrained", default=None)
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument(
        "--vision-tower",
        default=None,
        help="CLIP vision tower path/name; defaults to the value in model config",
    )
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--conv_type", default="llava_v1", choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--mask_threshold", default=0.0, type=float)
    parser.add_argument("--max_samples", default=-1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--save_visualizations", action="store_true", default=False)
    parser.add_argument("--max_visualizations", default=40, type=int)
    parser.add_argument("--save_masks", action="store_true", default=False)
    return parser.parse_args(args)


def torch_dtype_from_precision(precision):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def resolve_vision_tower(args):
    config = transformers.AutoConfig.from_pretrained(args.version)
    if args.vision_tower:
        config.vision_tower = args.vision_tower
        config.mm_vision_tower = args.vision_tower
    elif hasattr(config, "vision_tower"):
        args.vision_tower = config.vision_tower
    elif hasattr(config, "mm_vision_tower"):
        args.vision_tower = config.mm_vision_tower
        config.vision_tower = args.vision_tower
    else:
        raise ValueError(
            "Cannot resolve CLIP vision tower. Pass --vision-tower with a local "
            "openai/clip-vit-large-patch14 directory."
        )
    return config


class ReasonSegValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(self, base_image_dir, tokenizer, vision_tower, val_dataset, image_size=1024):
        splits = val_dataset.split("|")
        if len(splits) != 2:
            raise ValueError("ReasonSeg benchmark expects --val_dataset like 'ReasonSeg|val'")
        ds, split = splits
        self.images = sorted(
            glob.glob(os.path.join(base_image_dir, "reason_seg", ds, split, "*.jpg"))
        )
        if len(self.images) == 0:
            raise FileNotFoundError(
                "No jpg files found under {}".format(
                    os.path.join(base_image_dir, "reason_seg", ds, split)
                )
            )
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        return len(self.images)

    def preprocess(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        return F.pad(x, (0, padw, 0, padh))

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("Failed to read image: {}".format(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        json_path = image_path.replace(".jpg", ".json")
        mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
        sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        for text in sampled_sents:
            conv.messages = []
            text = text.strip()
            if is_sentence:
                question = DEFAULT_IMAGE_TOKEN + "\n {} Please output segmentation mask.".format(text)
            else:
                question = (
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(text)
                )
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())

        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = torch.from_numpy(np.stack([mask_json], axis=0))
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )


def collate_reason_seg(batch, tokenizer=None, use_mm_start_end=True):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    offset_list = [0]
    cnt = 0
    inferences = []

    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        _questions,
        _sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        masks_list.append(masks.float())
        label_list.append(label)
        resize_list.append(resize)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        conversation_list = [
            conversation.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            for conversation in conversation_list
        ]

    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": [None for _ in batch],
        "sampled_classes_list": [None for _ in batch],
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


def load_model_and_tokenizer(args):
    model_config = resolve_vision_tower(args)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch_dtype_from_precision(args.precision)
    model_kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        model_kwargs.update(
            {
                "torch_dtype": torch.float16,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        model_kwargs.update(
            {
                "torch_dtype": torch.float16,
                "quantization_config": BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )

    model_args = {
        "train_mask_decoder": True,
        "out_dim": args.out_dim,
        "vision_tower": args.vision_tower,
        "vision_pretrained": args.vision_pretrained,
        "use_mm_start_end": args.use_mm_start_end,
        "seg_token_idx": args.seg_token_idx,
    }

    model = LISAForCausalLM.from_pretrained(
        args.version,
        config=model_config,
        low_cpu_mem_usage=True,
        **model_args,
        **model_kwargs,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    if args.load_in_4bit or args.load_in_8bit:
        model.cuda()
    elif args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16":
        model = model.half().cuda()
    else:
        model = model.float().cuda()

    model.eval()
    return model, tokenizer


def cast_batch_precision(input_dict, precision):
    if precision == "fp16":
        input_dict["images"] = input_dict["images"].half()
        input_dict["images_clip"] = input_dict["images_clip"].half()
    elif precision == "bf16":
        input_dict["images"] = input_dict["images"].bfloat16()
        input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
    else:
        input_dict["images"] = input_dict["images"].float()
        input_dict["images_clip"] = input_dict["images_clip"].float()
    return input_dict


def binary_metrics(pred_mask, gt_mask):
    valid = gt_mask != 255
    target = (gt_mask == 1) & valid
    pred = pred_mask & valid

    intersection = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    pred_area = int(pred.sum())
    target_area = int(target.sum())
    valid_area = int(valid.sum())
    false_positive_area = int(np.logical_and(pred, np.logical_not(target)).sum())
    false_negative_area = int(np.logical_and(np.logical_not(pred), target).sum())

    iou = intersection / union if union > 0 else 1.0
    precision = intersection / pred_area if pred_area > 0 else (1.0 if target_area == 0 else 0.0)
    recall = intersection / target_area if target_area > 0 else 1.0
    dice_den = pred_area + target_area
    dice = 2 * intersection / dice_den if dice_den > 0 else 1.0

    return {
        "intersection": intersection,
        "union": union,
        "pred_area": pred_area,
        "target_area": target_area,
        "valid_area": valid_area,
        "false_positive_area": false_positive_area,
        "false_negative_area": false_negative_area,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "dice": dice,
    }


def overlay_mask(image_rgb, mask, color, alpha=0.5):
    result = image_rgb.copy()
    color_arr = np.array(color, dtype=np.float32)
    result[mask] = result[mask] * (1 - alpha) + color_arr * alpha
    return result.astype(np.uint8)


def make_visualization(image_path, pred_mask, gt_mask, save_path, title_text):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return False
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gt_target = gt_mask == 1
    ignore = gt_mask == 255
    pred = pred_mask.astype(bool)

    gt_vis = overlay_mask(image_rgb, gt_target, [0, 220, 0])
    gt_vis = overlay_mask(gt_vis, ignore, [160, 160, 160], alpha=0.45)
    pred_vis = overlay_mask(image_rgb, pred, [255, 40, 40])

    error_vis = image_rgb.copy()
    true_positive = pred & gt_target
    false_positive = pred & ~gt_target & ~ignore
    false_negative = ~pred & gt_target
    error_vis = overlay_mask(error_vis, true_positive, [0, 220, 0], alpha=0.55)
    error_vis = overlay_mask(error_vis, false_positive, [255, 40, 40], alpha=0.55)
    error_vis = overlay_mask(error_vis, false_negative, [30, 120, 255], alpha=0.55)
    error_vis = overlay_mask(error_vis, ignore, [160, 160, 160], alpha=0.35)

    panels = [image_rgb, gt_vis, pred_vis, error_vis]
    labels = ["image", "ground truth", "prediction", "tp/fp/fn"]
    labeled_panels = []
    for label, panel in zip(labels, panels):
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.rectangle(panel_bgr, (0, 0), (panel_bgr.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            panel_bgr,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        labeled_panels.append(panel_bgr)

    vis = cv2.hconcat(labeled_panels)
    cv2.rectangle(vis, (0, vis.shape[0] - 34), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        vis,
        title_text[:180],
        (10, vis.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(save_path), vis)
    return True


def safe_json_text(image_path):
    json_path = Path(str(image_path)).with_suffix(".json")
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with json_path.open("r", encoding="cp1252") as f:
            data = json.load(f)
    except FileNotFoundError:
        return "", None
    texts = data.get("text", [])
    return texts[0] if texts else "", data.get("is_sentence")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    preferred = [
        "sample_index",
        "mask_index",
        "image",
        "prompt",
        "is_sentence",
        "iou",
        "dice",
        "precision",
        "recall",
        "intersection",
        "union",
        "pred_area",
        "target_area",
        "valid_area",
        "false_positive_area",
        "false_negative_area",
        "pred_mask_path",
        "visualization_path",
    ]
    all_fields = set()
    for row in rows:
        all_fields.update(row.keys())
    fieldnames = [field for field in preferred if field in all_fields]
    fieldnames.extend(sorted(all_fields - set(fieldnames)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, args, elapsed_seconds):
    total_intersection = sum(row["intersection"] for row in rows)
    total_union = sum(row["union"] for row in rows)
    total_pred_area = sum(row["pred_area"] for row in rows)
    total_target_area = sum(row["target_area"] for row in rows)
    total_fp = sum(row["false_positive_area"] for row in rows)
    total_fn = sum(row["false_negative_area"] for row in rows)

    def avg(key):
        return float(np.mean([row[key] for row in rows])) if rows else 0.0

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.version,
        "vision_tower": args.vision_tower,
        "dataset_dir": args.dataset_dir,
        "val_dataset": args.val_dataset,
        "precision": args.precision,
        "mask_threshold": args.mask_threshold,
        "num_samples": len(rows),
        "gIoU": avg("iou"),
        "cIoU": total_intersection / total_union if total_union > 0 else 0.0,
        "mean_dice": avg("dice"),
        "mean_precision": avg("precision"),
        "mean_recall": avg("recall"),
        "total_intersection": total_intersection,
        "total_union": total_union,
        "total_pred_area": total_pred_area,
        "total_target_area": total_target_area,
        "total_false_positive_area": total_fp,
        "total_false_negative_area": total_fn,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_sample": elapsed_seconds / len(rows) if rows else 0.0,
    }

    sorted_rows = sorted(rows, key=lambda row: row["iou"])
    summary["worst_samples"] = [
        {
            "image": row["image"],
            "iou": row["iou"],
            "dice": row["dice"],
            "prompt": row["prompt"],
        }
        for row in sorted_rows[:5]
    ]
    summary["best_samples"] = [
        {
            "image": row["image"],
            "iou": row["iou"],
            "dice": row["dice"],
            "prompt": row["prompt"],
        }
        for row in sorted_rows[-5:][::-1]
    ]
    return summary


def write_markdown_summary(path, summary):
    lines = [
        "# LISA ReasonSeg Benchmark",
        "",
        f"- Model: `{summary['model']}`",
        f"- Dataset: `{summary['dataset_dir']}` / `{summary['val_dataset']}`",
        f"- Samples: `{summary['num_samples']}`",
        f"- Precision: `{summary['precision']}`",
        f"- Mask threshold: `{summary['mask_threshold']}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| gIoU | {summary['gIoU']:.4f} |",
        f"| cIoU | {summary['cIoU']:.4f} |",
        f"| Mean Dice | {summary['mean_dice']:.4f} |",
        f"| Mean Precision | {summary['mean_precision']:.4f} |",
        f"| Mean Recall | {summary['mean_recall']:.4f} |",
        f"| Seconds / sample | {summary['seconds_per_sample']:.2f} |",
        "",
        "## Worst Samples",
        "",
        "| Image | IoU | Dice | Prompt |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in summary["worst_samples"]:
        prompt = str(row["prompt"]).replace("|", "\\|")
        lines.append(f"| `{row['image']}` | {row['iou']:.4f} | {row['dice']:.4f} | {prompt} |")

    lines.extend(
        [
            "",
            "## Best Samples",
            "",
            "| Image | IoU | Dice | Prompt |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in summary["best_samples"]:
        prompt = str(row["prompt"]).replace("|", "\\|")
        lines.append(f"| `{row['image']}` | {row['iou']:.4f} | {row['dice']:.4f} | {prompt} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def markdown_relative_link(target_path, markdown_path):
    if not target_path:
        return ""
    return os.path.relpath(target_path, start=Path(markdown_path).parent)


def write_sorted_samples_markdown(path, rows):
    sorted_rows = sorted(rows, key=lambda row: (row["iou"], row["dice"], row["image"]))
    lines = [
        "# Samples Sorted By IoU",
        "",
        "| Rank | Sample | IoU | Dice | Precision | Recall | Target Area | Pred Area | Prompt | Visualization |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, row in enumerate(sorted_rows, start=1):
        prompt = str(row.get("prompt", "")).replace("|", "\\|")
        image = str(row["image"]).replace("|", "\\|")
        visualization_path = row.get("visualization_path", "")
        visualization_link = markdown_relative_link(visualization_path, path)
        visualization = f"[view]({visualization_link})" if visualization_link else ""
        lines.append(
            "| {rank} | `{image}` | {iou:.4f} | {dice:.4f} | {precision:.4f} | "
            "{recall:.4f} | {target_area} | {pred_area} | {prompt} | {visualization} |".format(
                rank=rank,
                image=image,
                iou=row["iou"],
                dice=row["dice"],
                precision=row["precision"],
                recall=row["recall"],
                target_area=row["target_area"],
                pred_area=row["pred_area"],
                prompt=prompt,
                visualization=visualization,
            )
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(args):
    args = parse_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    mask_dir = output_dir / "pred_masks"
    if args.save_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    model, tokenizer = load_model_and_tokenizer(args)

    dataset = ReasonSegValDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        args.val_dataset,
        args.image_size,
    )
    if args.max_samples > 0:
        dataset_len = min(len(dataset), args.max_samples)
        dataset = torch.utils.data.Subset(dataset, list(range(dataset_len)))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=lambda batch: collate_reason_seg(
            batch,
            tokenizer=tokenizer,
            use_mm_start_end=args.use_mm_start_end,
        ),
    )

    rows = []
    started_at = datetime.now()
    with torch.no_grad():
        for sample_idx, input_dict in enumerate(tqdm.tqdm(data_loader, desc="benchmark")):
            image_path = input_dict["image_paths"][0]
            prompt_text, is_sentence = safe_json_text(image_path)
            input_dict = dict_to_cuda(input_dict)
            input_dict = cast_batch_precision(input_dict, args.precision)

            output_dict = model(**input_dict)
            pred_masks = output_dict["pred_masks"][0].detach().float().cpu().numpy()
            gt_masks = output_dict["gt_masks"][0].detach().cpu().numpy().astype(np.uint8)

            for mask_idx, (pred_logits, gt_mask) in enumerate(zip(pred_masks, gt_masks)):
                pred_mask = pred_logits > args.mask_threshold
                metrics = binary_metrics(pred_mask, gt_mask)
                stem = Path(image_path).stem
                row = {
                    "sample_index": sample_idx,
                    "mask_index": mask_idx,
                    "image": image_path,
                    "prompt": prompt_text,
                    "is_sentence": is_sentence,
                    **metrics,
                }
                rows.append(row)

                if args.save_masks:
                    mask_path = mask_dir / f"{sample_idx:05d}_{stem}_mask{mask_idx}.png"
                    cv2.imwrite(str(mask_path), pred_mask.astype(np.uint8) * 255)
                    row["pred_mask_path"] = str(mask_path)

                should_save_vis = (
                    args.save_visualizations
                    and (args.max_visualizations < 0 or sample_idx < args.max_visualizations)
                )
                if should_save_vis:
                    vis_path = vis_dir / f"{sample_idx:05d}_{stem}_mask{mask_idx}.jpg"
                    title_text = (
                        f"IoU={metrics['iou']:.3f} Dice={metrics['dice']:.3f} "
                        f"P={metrics['precision']:.3f} R={metrics['recall']:.3f}"
                    )
                    if make_visualization(image_path, pred_mask, gt_mask, vis_path, title_text):
                        row["visualization_path"] = str(vis_path)

    elapsed_seconds = (datetime.now() - started_at).total_seconds()
    summary = summarize(rows, args, elapsed_seconds)

    sorted_rows = sorted(rows, key=lambda row: (row["iou"], row["dice"], row["image"]))
    write_csv(output_dir / "per_sample_metrics.csv", rows)
    write_jsonl(output_dir / "per_sample_metrics.jsonl", rows)
    write_csv(output_dir / "per_sample_metrics_by_iou.csv", sorted_rows)
    write_sorted_samples_markdown(output_dir / "samples_by_iou.md", sorted_rows)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_markdown_summary(output_dir / "summary.md", summary)

    print("\nBenchmark finished")
    print(f"Samples: {summary['num_samples']}")
    print(f"gIoU: {summary['gIoU']:.4f}")
    print(f"cIoU: {summary['cIoU']:.4f}")
    print(f"Mean Dice: {summary['mean_dice']:.4f}")
    print(f"Mean Precision: {summary['mean_precision']:.4f}")
    print(f"Mean Recall: {summary['mean_recall']:.4f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
