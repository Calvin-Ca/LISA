from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings
from .errors import InferenceError
from .image_io import encode_png_base64


@dataclass
class SegmentationResult:
    width: int
    height: int
    text: str
    masks: list[str]


class LisaBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loaded = False
        self._objects: dict[str, Any] = {}

    def load(self) -> None:
        if self.loaded:
            return

        import torch
        from transformers import (
            AutoTokenizer,
            BitsAndBytesConfig,
            CLIPImageProcessor,
        )

        from model.LISA import LISAForCausalLM
        from model.llava import conversation as conversation_lib
        from model.segment_anything.utils.transforms import ResizeLongestSide

        settings = self.settings
        tokenizer = AutoTokenizer.from_pretrained(
            settings.model_path,
            cache_dir=None,
            model_max_length=settings.model_max_length,
            padding_side="right",
            use_fast=False,
            local_files_only=True,
        )
        tokenizer.pad_token = tokenizer.unk_token
        seg_token_idx = tokenizer(
            "[SEG]", add_special_tokens=False
        ).input_ids[0]

        torch_dtype = torch.float32
        if settings.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif settings.precision == "fp16":
            torch_dtype = torch.float16

        kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        if settings.load_in_4bit:
            kwargs.update(
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
        elif settings.load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"],
                load_in_8bit=True,
            )

        model = LISAForCausalLM.from_pretrained(
            settings.model_path,
            low_cpu_mem_usage=True,
            vision_tower=settings.vision_tower,
            seg_token_idx=seg_token_idx,
            local_files_only=True,
            **kwargs,
        )
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.get_model().initialize_vision_modules(model.get_model().config)

        if settings.precision == "bf16":
            model = model.bfloat16().cuda(settings.gpu_index)
        elif (
            settings.precision == "fp16"
            and not settings.load_in_4bit
            and not settings.load_in_8bit
        ):
            model = model.half().cuda(settings.gpu_index)
        elif settings.precision == "fp32":
            model = model.float().cuda(settings.gpu_index)

        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(device=settings.gpu_index, dtype=torch_dtype)
        model.eval()

        self._objects = {
            "torch": torch,
            "tokenizer": tokenizer,
            "model": model,
            "clip_processor": CLIPImageProcessor.from_pretrained(
                settings.vision_tower,
                local_files_only=True,
            ),
            "transform": ResizeLongestSide(settings.image_size),
            "conversation_lib": conversation_lib,
        }
        self.loaded = True

    def segment(self, image_bgr, prompt_text: str) -> SegmentationResult:
        if not self.loaded:
            self.load()

        try:
            return self._segment(image_bgr, prompt_text)
        except Exception as exc:
            raise InferenceError(f"LISA inference failed: {exc}") from exc

    def _segment(self, image_bgr, prompt_text: str) -> SegmentationResult:
        import cv2
        import numpy as np
        import torch.nn.functional as F

        from model.llava.mm_utils import tokenizer_image_token
        from utils.utils import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )

        settings = self.settings
        torch = self._objects["torch"]
        tokenizer = self._objects["tokenizer"]
        model = self._objects["model"]
        clip_processor = self._objects["clip_processor"]
        transform = self._objects["transform"]
        conversation_lib = self._objects["conversation_lib"]

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        conversation = conversation_lib.conv_templates["llava_v1"].copy()
        conversation.messages = []
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
        replacement = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replacement)
        conversation.append_message(conversation.roles[0], prompt)
        conversation.append_message(conversation.roles[1], "")
        model_prompt = conversation.get_prompt()

        image_clip = clip_processor.preprocess(
            image_rgb, return_tensors="pt"
        )["pixel_values"][0].unsqueeze(0)
        image_clip = image_clip.cuda(settings.gpu_index)

        resized = transform.apply_image(image_rgb)
        resize_list = [resized.shape[:2]]
        image_tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous()
        pixel_mean = torch.tensor(
            [123.675, 116.28, 103.53], dtype=torch.float32
        ).view(-1, 1, 1)
        pixel_std = torch.tensor(
            [58.395, 57.12, 57.375], dtype=torch.float32
        ).view(-1, 1, 1)
        image_tensor = (image_tensor - pixel_mean) / pixel_std
        pad_h = settings.image_size - image_tensor.shape[-2]
        pad_w = settings.image_size - image_tensor.shape[-1]
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                "resized image exceeds configured LISA_IMAGE_SIZE"
            )
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h))
        image_tensor = image_tensor.unsqueeze(0).cuda(settings.gpu_index)

        if settings.precision == "bf16":
            image_clip = image_clip.bfloat16()
            image_tensor = image_tensor.bfloat16()
        elif settings.precision == "fp16":
            image_clip = image_clip.half()
            image_tensor = image_tensor.half()
        else:
            image_clip = image_clip.float()
            image_tensor = image_tensor.float()

        input_ids = tokenizer_image_token(
            model_prompt, tokenizer, return_tensors="pt"
        ).unsqueeze(0).cuda(settings.gpu_index)

        with torch.inference_mode():
            output_ids, pred_masks = model.evaluate(
                image_clip,
                image_tensor,
                input_ids,
                resize_list,
                [image_rgb.shape[:2]],
                max_new_tokens=settings.max_new_tokens,
                tokenizer=tokenizer,
            )

        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        text = " ".join(text.replace("\n", " ").split())
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[1].strip()

        encoded_masks: list[str] = []
        for pred_mask in pred_masks:
            if pred_mask.shape[0] == 0:
                continue
            mask = pred_mask.detach().float().cpu().numpy()[0]
            binary = (mask > settings.mask_threshold).astype(np.uint8) * 255
            encoded_masks.append(encode_png_base64(binary))

        return SegmentationResult(
            width=width,
            height=height,
            text=text,
            masks=encoded_masks,
        )
