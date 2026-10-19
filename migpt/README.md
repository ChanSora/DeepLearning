# WebText 中文预训练

`train.py` 默认读取同目录的 `web_text_zh_train.json` 和
`web_text_zh_valid.json`（每行一条 JSON），使用 `bert-base-chinese` 的
WordPiece tokenizer。只复用分词器，不加载 BERT 模型权重。

```bash
pip install -r requirements.txt
```

```bash
# 清洗、编码全部数据并开始训练
python train.py --total-steps 70000

# 只准备数据，暂不训练
python train.py --prepare-only

# 先用每个文件前 1000 行检查处理结果，使用独立缓存目录
python train.py --prepare-only --max-records 1000 --cache-dir /tmp/webtext_sample
```

可通过 `--train-path`、`--val-path` 指定其他 WebText JSONL 文件。
`--min-chars` 设置清洗后回答的最小长度，默认 20；设为 1 可保留短回答。
`--max-records` 限制读取的原始行数，未指定时处理全部数据。
`--tokenizer` 可指定本地 `bert-base-chinese` tokenizer 目录，默认优先使用
Hugging Face 本地缓存，缺失时下载分词器文件。

每条文本按“标题、问题描述、回答”顺序拼接，清洗文件中文档之间使用空行分隔。
清洗会删除 HTML 标签、脚本、样式、URL 和异常控制字符，统一 Unicode
及空白，跳过无效 JSON、非文本或空回答、过短回答，并按清洗后的完整回答去重。
训练集优先保留，验证集中与训练集相同的回答会被排除。
不使用测试集 `web_text_zh_testa.json`。

词表裁剪只统计清洗后**训练集**实际分词产生的 token，并保留全部特殊 token：
`[PAD]`、`[UNK]`、`[CLS]`、`[SEP]`、`[MASK]`。未出现的普通 token
（包括未使用的 `[unusedN]`）会被移除。保留项按 BERT 原始 ID 排序，
重排为 `0 .. vocab_size-1`；模型 embedding 和输出层使用这个裁剪后的大小。
验证和推理时，原始分词结果中被裁掉的 token 映射到裁剪词表的 `[UNK]`。
分词规则参考 [Hugging Face BERT 文档](https://huggingface.co/docs/transformers/model_doc/bert)。

BERT 忽略空白，每篇文档显式编码为 `[CLS] 正文 token [SEP]`，不进行
512 token 截断；训练批次仍由 `block_size` 控制。WordPiece 原始分词规则
完整保存在 tokenizer 元数据中，避免直接删除原分词词典后导致重新切词；
模型只接收裁剪后的 token ID。

首次处理逐行读取原始文件，通过 SQLite 临时索引去重；输出保存在
`webtext_cache/`，需要额外磁盘空间：

- `train.txt`、`val.txt`：清洗后的文本，可直接查看。
- `train.bin`、`val.bin`：每个 token 占 4 字节的裁剪后 ID，训练时使用内存映射按批读取。
- `vocab.json`：裁剪后的 token 列表 `tokens`、新 ID 到 BERT 原始 ID 的映射
  `original_ids`、特殊 token 和完整分词规则。新 ID 是数组下标，`[UNK]` 不一定是 0。
- `metadata.json`：过滤统计、裁剪前后词表大小及缓存参数；输入路径、大小、
  修改时间、清洗参数或 tokenizer 名称变化时重新处理。旧字符缓存会自动失效。

训练配置集中在 `train.py` 的 `TrainConfig`，也可以通过命令行覆盖。
默认训练 70000 步，前 2000 步线性预热，之后余弦衰减；每 500 步验证一次，
每次对训练集和验证集各采样 500 批，每 2000 步保存 checkpoint。
`--n-head` 现在会实际传入模型，默认是 7；嵌入维度必须能被注意力头数整除。
CUDA 支持 bf16 时默认使用 bf16 混合精度，其余情况使用 fp32。

新训练默认使用 `model_v2.py` 的第二代模型：Q/K/V 使用一个线性层投影，
自注意力调用 PyTorch `scaled_dot_product_attention`，设置 `is_causal=True`，
不再逐头计算和存储注意力矩阵或预建三角 mask。缩放使用标准的 `1/sqrt(head_size)`。
训练时保留注意力 dropout，验证和生成时将其设为 0。
[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
会根据设备、数据类型等条件选择 FlashAttention 等后端；CPU 或不满足融合内核条件时
自动选择可用实现，无需安装额外的 `flash-attn` 包。默认 CUDA bf16 训练为使用融合内核创造条件，
但实际后端由 PyTorch 决定，不能仅凭调用 SDPA 就认定某次运行使用了 FlashAttention。

checkpoint 和推理权重会记录 `model_version`，新文件为 2；缺少版本标记的旧文件按
第一代模型加载。`--resume` 和 `--init-from` 保留输入文件的模型版本，
不会把第一代的逐头 Q/K/V 权重和优化器状态直接套到第二代的合并投影层上。
要训练第二代模型，使用新训练入口即可；生成脚本会按保存的版本自动恢复模型。

训练文件默认写入以下位置：

| 文件 | 用途 |
| --- | --- |
| `checkpoints/last.pt` | 最新完整 checkpoint，用于断点续训 |
| `checkpoints/ckpt_step_00002000.pt` | 按完成步数保存的 checkpoint，默认保留最近 10 个 |
| `checkpoints/best_model.pt` | 验证集 loss 最低时的权重及推理元数据，不含优化器 |
| `migpt_bert_pruned.pth` | 正常结束或安全停止时的最新推理权重及元数据 |
| `migpt_bert_pruned_vocab.json` | 导出的配套 tokenizer，便于单独查看或使用 |

完整 checkpoint 包含模型、AdamW、学习率调度器、已完成步数、历史最佳验证 loss、
历史最佳权重快照、训练配置、数据签名、tokenizer 和 Python/NumPy/PyTorch/CUDA
随机状态。复制单个完整 checkpoint 到新目录，也可恢复历史最佳权重。
模型权重以 `state_dict` 保存，同时记录模型结构，避免依赖完整 Python 模型对象。

```bash
# 从最近一次保存的完整断点继续，自动恢复原训练配置、数据路径和学习率进度
python train.py --resume checkpoints/last.pt

# 训练计划为 70000 步，先在累计 10000 步时保存并结束
python train.py --total-steps 70000 --max-steps 10000

# 从累计 10000 步继续到累计 20000 步
python train.py --resume checkpoints/last.pt --max-steps 20000

# 调整保存频率、历史 checkpoint 数量及最终导出路径
python train.py --save-interval 1000 --keep-last 5 --output runs/model.pth
```

`--max-steps` 是本次运行的目标**累计步数**，不是新增步数；不指定时训练到
`--total-steps`。`--total-steps` 定义完整的学习率计划，续训时自动沿用。
超过计划步数后，学习率保持在 `min_lr`。验证使用固定、独立的随机采样，
不会消耗训练随机序列；同一环境下的恢复测试覆盖了权重、优化器和随机状态的一致性。

按 Ctrl+C 或收到 SIGTERM 后，会完成当前训练步，验证并保存后退出（退出码 130）。
正常结束或安全停止时，即使未到保存间隔，也会保存完整 checkpoint。
保存采用同目录临时文件加原子替换，写入失败时已有文件仍可使用。

`--checkpoint-dir` 指定保存目录，续训时默认使用传入 checkpoint 的目录。
续训会检查原数据签名和 tokenizer 映射；模型结构、训练参数和学习率计划自动恢复，
仅允许覆盖 `--save-interval` 和 `--keep-last` 等保存设置。
新 token ID 与旧字符模型不兼容，不能直接续训。

只加载权重重新训练时使用 `--init-from`，优化器、学习率和步数从头开始。
输入可以是新推理权重、完整 checkpoint 或先前保存的完整 BERT 模型对象，
但必须使用相同的裁剪词表，并传入与原模型一致的结构参数。
例如旧模型实际使用 4 个注意力头时，需指定 `--n-head 4`。

用 `generate.py` 指定 prompt 和新增长度进行续写：

```bash
python generate.py --prompt "人工智能的发展" --length 200

# 直接使用最佳权重，tokenizer 已内置，无需单独提供词表
python generate.py --model checkpoints/best_model.pt --prompt "人工智能的发展" --length 200

# 指定模型、词表、设备和随机种子
python generate.py --model migpt_bert_pruned.pth --vocab migpt_bert_pruned_vocab.json \
    --prompt "如何学习深度学习？" --length 100 --device cpu --seed 42

# 控制采样温度和候选范围
python generate.py --model checkpoints/best_model.pt --prompt "人工智能的发展" \
    --length 200 --temperature 0.8 --top-k 50 --top-p 0.9 --seed 42

# 贪心输出：每步选择概率最大的 token
python generate.py --prompt "人工智能的发展" --length 100 --temperature 0
```

采样先调整 temperature，再执行 top-k，最后执行 top-p；过滤后重新归一化并采样。

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--temperature` | `0.6` | 非负有限数，越小分布越集中，越大越分散；0 表示贪心输出 |
| `--top-k` / `--top_k` | `0` | 仅保留概率最高的 k 个 token；0 表示关闭，超过词表大小时保留全部 |
| `--top-p` / `--top_p` | `1.0` | 保留累计概率达到 p 的最小候选集合；范围 `(0, 1]`，1 表示关闭 |

一、二代模型均支持 `model.generate(inputs, max_tokens=100, temperature=0.8, top_k=50, top_p=0.9)`。
模型 API 的 temperature 默认是 1.0；`generate.py` 保留之前的默认温度 0.6。
这些参数只影响推理，无需重新训练或转换 checkpoint。

输出包含原始 prompt 和新增文本。`--length`（也可写作 `--max-new-tokens`）
不包含 prompt 的长度；一个 WordPiece token 可能是汉字、英文词或子词，
因此新增 token 数不等于字符数。解码会合并 `##` 子词并跳过特殊 token。
生成的中文会去除汉字、中文标点之间多余的空格，例如 `你 好 ， 世 界` 输出为
`你好，世界`；英文单词之间的空格、换行及用户原始 prompt 保留。
prompt 超过上下文长度时，模型自动使用末尾的上下文。
新格式的完整 checkpoint、最佳权重和最终推理权重均内置 tokenizer；
不传 `--vocab` 时直接使用内置词表。旧模型对象则读取模型同目录的 `<模型名>_vocab.json`。
模型保存词表指纹，生成时会检查映射是否匹配。仍可通过 `--model` 指定旧字符模型，
使用原来的字符词表进行生成。

验证数据处理：

```bash
python -m unittest -v test_data.py test_generate.py test_train.py test_model_v2.py test_sampling.py
```
