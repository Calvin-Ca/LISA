# lisa13b-clean030-production-perf-shared-gpu-v1

## 状态

共享GPU方案已确认，脚本已准备，等待远程 Linux GPU 服务器执行。

## 背景

目标A100 40GB上长期运行不可停止的 `bge-m3` vLLM pooling服务，当前观测
显存约2,314 MiB。独占GPU性能实验因检测到该进程而正确退出，因此本实验
改为评估真实共享部署条件：

- 保持 `bge-m3` 在线，不停止、不暂停、不修改其配置。
- 记录LISA启动前的共享GPU基线显存。
- 报告LISA加载和推理相对基线的显存增量。
- 验证两个模型共存时LISA请求成功率、延迟、峰值显存和稳定性。
- 确认测试前已存在的 `VLLM::EngineCore` 在测试结束前仍然存在。

本实验结果只代表当前共享GPU部署条件，不能替代未来的LISA独占GPU纯性能
基线。由于无法获得 `bge-m3` 的业务健康接口，本实验只能检查其GPU进程
存活，不能证明其业务延迟完全没有受到影响。

## 模型与环境

- LISA版本：`lisa13b-clean030-v1`
- LISA制品：`artifacts/lisa-safety-seg/lisa13b-clean030-v1/merged_hf`
- LISA精度：bf16，未启用8bit/4bit
- 共存服务：`vllm serve /models/bge-m3 --runner pooling`
- 共存GPU进程名：`VLLM::EngineCore`
- GPU：NVIDIA A100-PCIE-40GB，40,960 MiB
- 驱动：`580.159.03`
- PyTorch：`2.1.0+cu121`
- Transformers：`4.31.0`
- vLLM进程不可停止，因此所有显存和延迟结论均标记为shared-GPU

## 固定请求

- 图片：
  `dataset/reason_seg/ReasonSeg/val/val__002__-helmet_missing-19-_jpg.rf.cef508999f7777acb7cb6470fd767fef__helmet_missing.jpg`
- Prompt：`标出未按规定佩戴安全帽的作业人员。`
- API：`POST /v1/segment`
- 地址：`127.0.0.1:8001`
- LISA worker：1
- LISA并发：1
- mask threshold：`0.0`

## 实验设计

共执行3轮，每轮都重新启动和停止LISA服务，但不操作 `bge-m3`。每轮包含：

1. 记录LISA启动前的已有GPU进程和显存。
2. 要求至少存在一个名称包含 `VLLM::EngineCore` 的共享进程，否则退出。
3. 启动200毫秒间隔的GPU显存、利用率和温度采样。
4. 启动单worker LISA Uvicorn并等待模型ready。
5. 执行1次首次请求、5次预热、30次正式计时和100次稳定性请求。
6. 校验所有LISA响应的request ID、尺寸、mask数量和PNG Base64。
7. 检查实验前已有的GPU计算进程PID仍存在。
8. 记录总峰值、相对基线峰值增量、峰值剩余显存和预热后显存漂移。
9. 停止本轮LISA服务，等待5秒后进入下一轮。

每轮136次请求，三轮共408次。三轮聚合报告使用最差P95、最高峰值显存、
最小剩余显存和最大显存漂移作为主要判断依据。

## 准入阈值

每轮必须同时满足：

- 136次LISA请求全部成功。
- 正式计时客户端P95不超过 `1500 ms`。
- GPU总峰值显存不超过 `36,864 MiB`。
- 峰值时至少保留约4GB显存。
- 稳定性阶段结束后相对预热完成时的显存增长不超过 `500 MiB`。
- 实验前已有的 `VLLM::EngineCore` PID没有消失。
- 不出现CUDA OOM、LISA模型重载或Uvicorn异常退出。

三轮全部通过才判定共享GPU基线通过。任一轮失败均保留结果，不自动启动
量化；先区分是共存负载波动、显存不足还是LISA服务问题。

## 执行命令

远程 Linux GPU 服务器执行：

```bash
bash exp/runs/lisa13b-clean030-production-perf-shared-gpu-v1/command.sh
```

脚本自包含模型、输入、请求数、阈值和共享进程要求，不依赖用户预先
`export` 环境变量。脚本拒绝覆盖已有非空输出目录。

## 预期产物

```text
exp/runs/lisa13b-clean030-production-perf-shared-gpu-v1/outputs/
├── round-1/
│   ├── runtime_config.json
│   ├── server.log
│   ├── gpu_metrics.csv
│   ├── requests.csv
│   ├── summary.json
│   └── summary.md
├── round-2/
├── round-3/
├── aggregate_summary.json
└── aggregate_summary.md
```

运行配置会记录已有计算进程的PID、进程名和显存。外部CLIP路径会转换为
可迁移的Hugging Face模型与snapshot标识，不写入服务器私有绝对路径。

## 结果

等待远程执行后填写。

## 结论

等待远程执行后填写。

## 局限

- `nvidia-smi` 只能提供整张GPU指标，不能精确拆分两个服务的瞬时利用率。
- LISA显存增量是相对共享基线的近似值；如果 `bge-m3` 在轮次内改变显存，
  增量会包含其变化。
- 未接入 `bge-m3` 健康和延迟接口，因此只检查其GPU进程存活。
- 固定单图片请求适合延迟和显存稳定性基线，不代表全部业务输入分布。
