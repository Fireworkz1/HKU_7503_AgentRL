# 训练与评测启动方式

## 环境要求

- Python 3.10+
- PyTorch >= 2.1（推荐 2.4.0+cu121）
- GPU: RTX 4090 24GB（单卡即可跑 DPO/评测，GRPO/PPO 推荐双卡）
- 依赖: `pip install -r requirements.txt`（minimind 项目根目录）

## 训练启动

### DPO 训练（推荐 v2 配置）
```bash
cd minimind/trainer
python train_dpo.py \
    --epochs 2 \
    --batch_size 4 \
    --learning_rate 5e-7 \
    --log_interval 50 \
    --save_interval 2000 \
    --from_weight full_sft \
    --data_path ../dataset/dpo.jsonl \
    --max_seq_len 1024 \
    --device cuda:0
```
- 无需 Reward Model，无需 sglang
- 基于 SFT 权重 + dpo.jsonl 偏好对数据
- 约 14 分钟完成（4090 单卡）
- 注意：官方默认 lr=4e-8 过低，训练几乎无效；推荐 5e-7，2 epoch

### GRPO 训练（规则奖励）
```bash
cd minimind/trainer
python train_grpo.py \
    --epochs 1 \
    --batch_size 4 \
    --learning_rate 5e-6 \
    --from_weight full_sft \
    --data_path ../dataset/rlaif.jsonl \
    --rollout_engine torch \
    --thinking_ratio 0.1 \
    --device cuda:0
```
- 默认使用规则奖励（长度分+think格式分+闭合分+重复惩罚）
- 加 `--reward_model_path <path>` 可启用 InternLM2-1.8B-Reward
- 加 `--rollout_engine sglang` 可用 sglang 加速 rollout

### PPO 训练（规则奖励）
```bash
cd minimind/trainer
python train_ppo.py \
    --epochs 1 \
    --batch_size 4 \
    --learning_rate 5e-6 \
    --from_weight full_sft \
    --data_path ../dataset/rlaif.jsonl \
    --rollout_engine torch \
    --device cuda:0
```

## 评测启动

### ToolCall 批量评测（80题）
```bash
cd minimind
python scripts/eval_toolcall_batch.py \
    --weight full_sft \
    --load_from ./model \
    --save_dir ./out
```
- `--weight` 可选: full_sft / dpo / grpo / ppo_actor / agent
- 结果保存到 `eval_results/toolcall_batch_{weight}.json`

### 主观QA + 数学评测
```bash
cd minimind
python scripts/eval_compare.py \
    --weights full_sft,dpo,grpo \
    --load_from ./model \
    --save_dir ./out
```

### 生成 Excel 报告
```bash
cd minimind
python scripts/gen_excel.py
```
- 需要 `ppo_train.log` 和 `grpo_sglang.log` 在当前目录
- 输出 `eval_results/training_summary.xlsx`

## 权重说明

| 文件名 | 来源 | 说明 |
|--------|------|------|
| full_sft_768.pth | 官方 | SFT baseline，所有 RL 训练的起点 |
| dpo_768_ours.pth | 我们训练 | DPO v1（lr=4e-8, 1ep，训练无效，作为对照） |
| dpo_v2_768.pth | 我们训练 | DPO v2（lr=5e-7, 2ep，loss 0.47→0.16） |
| grpo_768.pth | 官方 | GRPO，使用 InternLM2-1.8B-Reward |
| grpo_768_rule.pth | 我们训练 | GRPO，使用规则奖励 |
| ppo_actor_768.pth | 官方 | PPO，使用 InternLM2-1.8B-Reward |
| ppo_actor_768_rule.pth | 我们训练 | PPO，使用规则奖励 |
| agent_768.pth | 官方 | Agent CISPO，多轮工具调用 RL |
