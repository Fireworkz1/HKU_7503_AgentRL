# MiniMind RL 实验

基于 [MiniMind](https://github.com/jingyaogong/minimind) 项目（64M 参数 LLM），对比 DPO / GRPO / PPO 三种强化学习方法的训练效果。

## 实验概览

| 方法 | 基础权重 | Reward Model | 训练数据 | 训练配置 | GPU |
|------|---------|-------------|---------|---------|-----|
| SFT (baseline) | pretrain | - | sft_t2t_mini.jsonl | 官方提供 | - |
| DPO v1 | full_sft | 无需 RM | dpo.jsonl (17166条偏好对) | lr=4e-8, 1ep（训练无效，作为对照） | 4090 |
| DPO v2 | full_sft | 无需 RM | dpo.jsonl (17166条偏好对) | lr=5e-7, 2ep, ~14min | 4090 |
| GRPO (rule) | full_sft | 规则奖励 | rlaif.jsonl | lr=5e-6, 1ep, ~1h | 4090×2 |
| PPO (rule) | full_sft | 规则奖励 | rlaif.jsonl | actor_lr=3e-7, critic_lr=5e-7, 1ep, ~1h | 4090×2 |
| GRPO (official) | full_sft | InternLM2-1.8B-Reward | rlaif.jsonl | 官方提供 | - |
| PPO (official) | full_sft | InternLM2-1.8B-Reward | rlaif.jsonl | 官方提供 | - |
| Agent (CISPO) | full_sft | Ground Truth验证 | agent_rl.jsonl | 官方提供 | - |

## 评测结果（80题）

数据来源：`训练结果，数据优先看这个文件.xlsx`

| 类别 | 题数 | SFT | GRPO (rule) | DPO v2 |
|------|------|-----|-------------|--------|
| 单工具-数学计算 | 15 | 100% | 100% | 100% |
| 单工具-其他工具 | 15 | 100% | 100% | 100% |
| 多步链式调用 | 20 | 85% | 90% | 90% |
| Think 思考标签可靠性 | 30 | 40% | 86.7% | 90% |
| **合计** | **80** | **73.8%** | **92.5%** | **93.8%** |

> PPO 官方权重输出崩溃，无法评测。

**关键发现：**
- DPO v2 综合表现最好（93.8%），Think 标签可靠性从 SFT 的 40% 提升到 90%
- GRPO 紧随其后（92.5%），Think 标签可靠性 86.7%
- SFT 在 ToolCall 上表现不错，但 Think 标签可靠性极低（40%），严重拖累总分
- RL 训练的核心收益体现在 Think 思考标签的可靠性上

## 训练曲线

| 方法 | 起始 | 结束 | 趋势 |
|------|------|------|------|
| DPO v2 | loss 0.5403 | loss 0.1272 | 稳定下降，收敛正常 |
| GRPO (rule) | reward 0.2048 | reward 1.3687 | 快速上升，训练有效 |
| PPO (rule) | reward -0.25 | reward 0.2083 | 几乎不动，训练失败 |

详见 `charts/` 目录。

## 数学计算 & Think 标签评测

| 权重 | Think标签输出 | 思考质量 | 答案正确性 |
|------|-------------|---------|-----------|
| SFT | 基本为空 | 大多数时候无推理过程 | 全错 |
| PPO | 输出重复垃圾文本（"平衡平衡…"） | 完全无意义（reward hacking） | 全错 |
| DPO | 能输出，偶尔为空 | 有推理过程但逻辑错误 | 全错 |
| GRPO | 能输出，偶尔为空 | 有推理过程但逻辑错误 | 全错 |

> 64M 参数模型的纯数学推理能力极弱，所有权重表现类似。RL 无法凭空创造推理能力。
> PPO 的"平衡平衡"输出是典型的 reward hacking / mode collapse，详见 `PPO崩溃原因和分析.md`。

## 模型权重

权重文件（.pth）因体积过大未包含在仓库中。如需获取权重文件，请通过邮件联系：fireworkz@connect.hku.hk

| 文件名 | 来源 | 说明 |
|--------|------|------|
| full_sft_768.pth | 官方 | SFT baseline，所有 RL 训练的起点 |
| dpo_768_ours.pth | 我们训练 | DPO v1（lr=4e-8, 1ep，训练无效，作为对照） |
| dpo_v2_768.pth | 我们训练 | DPO v2（lr=5e-7, 2ep） |
| grpo_768.pth | 官方 | GRPO，使用 InternLM2-1.8B-Reward |
| grpo_768_rule.pth | 我们训练 | GRPO，使用规则奖励 |
| ppo_actor_768.pth | 官方 | PPO，使用 InternLM2-1.8B-Reward |
| ppo_actor_768_rule.pth | 我们训练 | PPO，使用规则奖励 |
| agent_768.pth | 官方 | Agent CISPO，多轮工具调用 RL |

## 目录结构

```
minimind_rl_experiment/
├── README.md                    # 本文件
├── HOW_TO_RUN.md                # 训练与评测启动方式
├── ANALYSIS.md                  # DPO/GRPO/PPO 方法对比分析
├── 训练参数和细节.md              # 各方法训练参数详细记录
├── PPO崩溃原因和分析.md           # PPO mode collapse 根因分析
├── 实际案例.md                   # Think 标签评测实际输出案例
├── 训练结果，数据优先看这个文件.xlsx  # 训练结果汇总（含 Chain-of-Thoughts 和自建 ToolCall 评测）
├── weights/                     # 模型权重
│   ├── full_sft_768.pth         # SFT baseline（官方）
│   ├── dpo_768_ours.pth         # DPO v1（lr=4e-8, 1ep，训练无效）
│   ├── dpo_v2_768.pth           # DPO v2（lr=5e-7, 2ep）
│   ├── grpo_768.pth             # GRPO（官方，RM打分）
│   ├── grpo_768_rule.pth        # GRPO（我们训练，规则奖励）
│   ├── ppo_actor_768.pth        # PPO（官方，RM打分）
│   ├── ppo_actor_768_rule.pth   # PPO（我们训练，规则奖励）
│   └── agent_768.pth            # Agent CISPO（官方）
├── logs/                        # 训练日志
│   ├── dpo_train_v2.log         # DPO v2 日志
│   ├── grpo_sglang.log          # GRPO 训练日志
│   └── ppo_train.log            # PPO 训练日志
├── eval_results/                # 评测结果
│   ├── 官方评测版（不含PPO DPO）.md  # 80题 ToolCall 官方评测
│   └── reward_curve.png         # Reward 曲线图
├── charts/                      # 训练曲线图
│   ├── dpo_training_curve.png   # DPO v2 loss 曲线
│   ├── grpo_training_curve.png  # GRPO 训练曲线
│   ├── grpo_vs_ppo_reward.png   # GRPO vs PPO reward 对比
│   └── reward_curve.png         # Reward 曲线
├── training/                    # 训练脚本（来自 minimind 项目）
│   ├── train_dpo.py
│   ├── train_grpo.py
│   ├── train_ppo.py
│   ├── train_agent.py
│   └── trainer_utils.py
└── scripts/                     # 评测与可视化脚本
    ├── eval_toolcall_batch.py   # 80题 ToolCall 批量评测
    ├── eval_compare.py          # 主观QA + 数学评测
    ├── gen_excel.py             # Excel 报告生成
    ├── update_excel_dpo.py      # DPO 数据更新到 Excel
    └── plot_dpo_curve.py        # DPO 训练曲线绘制
```
