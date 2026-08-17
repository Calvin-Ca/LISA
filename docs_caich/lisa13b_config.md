# LISA-13B `config.json` 配置参数说明

本文按功能类别说明远程模型目录 `LISA13B/config.json` 中的全部配置参数。该文件描述的是 Hugging Face 模型结构、LLaVA 多模态连接方式以及部分 LISA 分割模块配置；训练数据、LoRA、损失权重和 SAM 权重路径等运行参数不在该文件中。

当前配置对应的总体结构为：

```text
CLIP ViT-L/14 图像特征
        │
        ▼
线性 mm_projector：1024 → 5120
        │
        ├── 与文本 token embedding 拼接
        ▼
LLaMA-2 13B：40 层，hidden_size=5120
        │
        ▼
[SEG] token 的隐藏状态
        │
        ▼
LISA text_hidden_fcs：5120 → 256
        │
        ▼
SAM Prompt Encoder + Mask Decoder
```

## 1. 模型身份与加载兼容信息

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `_name_or_path` | `liuhaotian/llava-llama-2-13b-chat-lightning-preview` | 记录当前模型最初所基于的 Hugging Face 模型名称。它主要是来源信息，不代表运行时一定会从该地址下载模型；本项目实际从 `--version` 指定的本地目录加载。 |
| `architectures` | `["LISAForCausalLM"]` | 告诉 Transformers 保存和加载时对应的顶层模型类是 `LISAForCausalLM`。项目实现位于 `model/LISA.py`。 |
| `model_type` | `llava` | Hugging Face AutoConfig 使用的模型类型标识。LISA 在 LLaVA/LLaMA 的基础上扩展分割能力，因此仍使用 `llava`。 |
| `transformers_version` | `4.31.0` | 写出该配置时使用的 Transformers 版本，用于兼容性记录，不会自动锁定当前 Python 环境中的实际版本。 |

## 2. LLaMA-13B 主体结构

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `hidden_size` | `5120` | LLaMA 每个 token 的隐藏向量维度。CLIP 特征必须投影到该维度后才能插入语言模型；`[SEG]` 隐藏状态也从该维度投影到 SAM 的 256 维 prompt 空间。 |
| `intermediate_size` | `13824` | Transformer 层中 MLP/FFN 的中间维度。 |
| `num_hidden_layers` | `40` | LLaMA Transformer 层数。 |
| `num_attention_heads` | `40` | Query 注意力头数量。由此可得每个注意力头维度为 `5120 / 40 = 128`。 |
| `num_key_value_heads` | `40` | Key/Value 注意力头数量。它与 Query 头数相同，因此当前配置使用标准多头注意力，而不是减少 K/V 头数的 GQA。 |
| `hidden_act` | `silu` | MLP 使用的激活函数。 |
| `rms_norm_eps` | `1e-5` | RMSNorm 为避免除零及数值不稳定而使用的 epsilon。 |
| `initializer_range` | `0.02` | 新建参数时默认初始化分布的标准差。加载已有 checkpoint 时，大部分参数会被权重覆盖。 |
| `max_position_embeddings` | `4096` | 模型结构支持的最大位置长度。实际训练长度还会被 `train_ds.py --model_max_length` 限制，且图像 patch token 也占用上下文长度。 |
| `rope_scaling` | `null` | 不启用额外的 RoPE 上下文扩展策略。 |
| `pretraining_tp` | `1` | Transformers 中用于兼容预训练 tensor-parallel 权重切分的参数。`1` 表示不做该切分；它不是 DeepSpeed 的 GPU 数量配置。 |
| `torch_dtype` | `bfloat16` | checkpoint 建议使用的参数类型。运行时仍可被 `--precision` 或 `from_pretrained(torch_dtype=...)` 覆盖。 |
| `use_cache` | `false` | 默认关闭自回归 KV cache。训练时关闭 cache 可配合 gradient checkpointing；推理生成时是否启用还取决于具体调用和代码覆盖。 |
| `tie_word_embeddings` | `false` | 输入 token embedding 与输出 `lm_head` 不共享同一组参数。 |

## 3. 词表与特殊 token

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `vocab_size` | `32003` | 输入 embedding 和输出 `lm_head` 对应的词表大小。新增 token 的具体名称和 ID 应以同目录 tokenizer 文件为准。 |
| `bos_token_id` | `1` | 句子开始 token 的 ID。 |
| `eos_token_id` | `2` | 句子结束 token 的 ID。 |
| `pad_token_id` | `0` | padding token 的 ID。`train_ds.py` 加载 tokenizer 后还会将 `tokenizer.pad_token` 设置为 `unk_token`，调试时应同时查看 tokenizer 的实际映射。 |

`[SEG]` 是 LISA 的关键特殊 token，但它的 ID 不直接保存在这个 `config.json` 中。训练和推理启动时通过 tokenizer 查询：

```python
seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
```

随后将 `seg_token_idx` 作为构造参数传给 `LISAForCausalLM`。

## 4. CLIP 视觉塔与视觉特征选择

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `vision_tower` | `openai/clip-vit-large-patch14` | LISA/LLaVA 当前使用的视觉编码器名称。加载已有 LISA checkpoint 时，`model/LISA.py` 会用它设置 `mm_vision_tower`。 |
| `mm_vision_tower` | `openai/clip-vit-large-patch14` | LLaVA 多模态模块识别视觉塔时使用的字段。当前与 `vision_tower` 相同。训练入口可通过 `--vision-tower` 同时覆盖二者。 |
| `mm_hidden_size` | `1024` | CLIP 输出的单个视觉 patch 特征维度，也是 `mm_projector` 的输入维度。 |
| `mm_vision_select_layer` | `-2` | 从 CLIP Vision Transformer 的倒数第二层选取视觉 hidden states，而不是使用最后一层。 |
| `mm_vision_select_feature` | `patch` | 只保留 patch token，丢弃 CLIP 的 CLS token。当前 CLIP 224×224、patch size 14 时会得到 16×16，即 256 个 patch token。 |
| `image_aspect_ratio` | `square` | LLaVA 视觉输入按正方形处理。 |
| `image_grid_pinpoints` | `null` | 不配置多尺度/任意分辨率图像网格切分点；当前代码走单张正方形 CLIP 输入。 |

需要区分两条图像分支：本节字段只描述进入 LLaMA 的 CLIP 分支，不描述 SAM ViT-H 图像编码器。SAM 的结构在 `model/segment_anything/build_sam.py` 中构建，权重路径由 `--vision_pretrained` 提供。

## 5. LLaVA 多模态投影与图像 token

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `use_mm_proj` | `true` | 启用多模态投影层 `mm_projector`，把 CLIP 的 1024 维 patch 特征映射到 LLaMA 的 5120 维隐藏空间。 |
| `mm_use_im_start_end` | `true` | 在对话文本中使用 `<im_start><image><im_end>` 包裹图像占位符。 |
| `mm_use_im_patch_token` | `false` | 不为每个图像 patch 在文本中显式写入 `<im_patch>` token。它不表示禁用 patch 特征；运行时仍会用 256 个 CLIP patch embedding 替换一个 `<image>` 占位符。 |
| `tune_mm_mlp_adapter` | `false` | 不以“只微调多模态 MLP adapter”的模式训练。当前 `train_ds.py` 还会显式冻结 `mm_projector`。 |
| `freeze_mm_mlp_adapter` | `true` | 配置层面要求冻结 `mm_projector`。实际是否产生梯度最终以代码中的 `requires_grad` 为准。 |
| `pretrain_mm_mlp_adapter` | `null` | 初始化视觉模块时不另外加载独立的预训练 projector 文件，直接使用模型 checkpoint 中已有权重。 |

当前实现用一个 `<image>` token 替换 256 个视觉 patch embedding，因此序列长度净增加 255。`model/LISA.py` 中 `[SEG]` 位置对齐逻辑也硬编码补了 255 个位置；如果更换图像尺寸、patch size 或视觉塔，需要同步检查该逻辑。

## 6. 视觉重采样器配置

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `mm_resampler_type` | `null` | 不使用额外的视觉 token resampler。CLIP patch 特征经 `mm_projector` 后直接插入语言模型。 |
| `tune_mm_vision_resampler` | `false` | 不训练视觉重采样器；当前本身也未配置 resampler。 |

## 7. LISA 分割扩展配置

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `out_dim` | `256` | `text_hidden_fcs` 的输出维度。它把 `[SEG]` 的 5120 维 LLaMA hidden state 映射到 SAM Prompt Encoder 所需的 256 维 embedding。 |
| `train_mask_decoder` | `true` | 初始化 LISA 模块时允许 SAM Mask Decoder 进入训练状态并设置为可训练。SAM Image Encoder 和 Prompt Encoder 仍被冻结。 |

`out_dim=256` 必须与当前 SAM 的 `prompt_embed_dim=256` 保持一致。改变它时不能只修改配置，还必须同步修改或替换 SAM Prompt Encoder 和 Mask Decoder 的通道维度。

## 8. 参数之间的关键关系

### 8.1 CLIP 到 LLaMA

```text
mm_hidden_size=1024
        │
        ▼ mm_projector
hidden_size=5120
```

对应实现：

```python
self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
```

### 8.2 LLaMA 到 SAM

```text
hidden_size=5120
        │
        ▼ text_hidden_fcs
out_dim=256
        │
        ▼
SAM sparse text prompt
```

### 8.3 注意力维度

```text
head_dim = hidden_size / num_attention_heads
         = 5120 / 40
         = 128
```

`num_key_value_heads` 同样为 40，所以没有使用共享 K/V 头的 GQA 压缩。

## 9. 不在 `config.json` 中的关键运行参数

阅读实验配置时，不能只看 `LISA13B/config.json`。以下内容由训练脚本、命令行或 tokenizer 提供：

| 配置 | 当前常用来源 | 说明 |
|---|---|---|
| SAM checkpoint | `--vision_pretrained` | 例如 SAM ViT-H 权重路径。配置文件只保存 LISA 输出维度和是否训练 decoder，不保存 SAM 权重路径。 |
| `[SEG]` token ID | tokenizer + `train_ds.py` | 运行时查询并传入 `LISAForCausalLM`。 |
| 语言 CE 权重 | `--ce_loss_weight` | 当前默认 `1.0`。 |
| Mask BCE 权重 | `--bce_loss_weight` | 当前默认 `2.0`。 |
| Mask Dice 权重 | `--dice_loss_weight` | 当前默认 `0.5`。 |
| LoRA 配置 | `--lora_r` 等参数 | 当前实验常用 `r=8`、`alpha=16`、`dropout=0.05`，目标层为 `q_proj,v_proj`。 |
| 模型实际精度 | `--precision` | 可选 `fp32`、`bf16` 或 `fp16`，会覆盖或具体落实 `torch_dtype`。 |
| 文本最大长度 | `--model_max_length` | 当前训练默认 512，小于结构上限 `max_position_embeddings=4096`。 |
| SAM 输入尺寸 | `--image_size` | 当前默认 1024。 |
| 数据集和采样比例 | `--dataset`、`--sample_rates` | 决定训练任务组合，不属于模型结构。 |
| DeepSpeed/批量配置 | 启动命令和 `train_ds.py` | batch size、梯度累积、GPU 数量等不保存在模型配置中。 |

## 10. 配置读取与覆盖顺序

训练入口 `train_ds.py` 的主要顺序如下：

1. 使用 `AutoConfig.from_pretrained(--version)` 读取该文件。
2. 如果提供 `--vision-tower`，覆盖 `vision_tower` 和 `mm_vision_tower`。
3. 从 tokenizer 中添加或查找 `[SEG]`，得到 `seg_token_idx`。
4. 将 SAM 权重路径、损失权重、`out_dim` 等命令行参数传给 `LISAForCausalLM.from_pretrained()`。
5. 初始化 CLIP、LISA 投影层和 SAM 模块。
6. 根据训练代码设置 `requires_grad`，因此最终冻结状态以运行时代码为准，而不应只根据配置字段判断。

因此，排查模型结构或训练差异时，建议同时保存并核对：

```text
LISA13B/config.json
tokenizer 配置
实验 command.sh
train_ds.py 的参数默认值与冻结逻辑
实际加载的 CLIP/SAM 路径
```
