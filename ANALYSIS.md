# DPO vs GRPO vs PPO 方法对比分析

## 三种方法的核心区别

| 维度 | DPO | GRPO | PPO |
|------|-----|------|-----|
| 训练方式 | 离线（off-policy） | 在线（on-policy） | 在线（on-policy） |
| 数据格式 | chosen/rejected 偏好对 | prompt-only，模型自生成 | prompt-only，模型自生成 |
| Reward Model | 不需要 | 需要（RM 或规则） | 需要（RM 或规则） |
| 模型架构 | 策略模型 + 冻结 ref | 策略模型 + 冻结 ref | Actor + Critic + 冻结 ref |
| 优势估计 | 隐式（log ratio） | 组内相对优势（N个rollout） | GAE（Critic网络估计） |
| 训练速度 | 最快（无采样开销） | 中等（需要rollout） | 最慢（双网络+rollout） |
| Reward Hacking 风险 | 低（无显式reward） | 中（可能hack RM） | 高（Critic+RM双重hack） |

## 在 MiniMind（64M）上的实际表现

### 训练曲线
- **DPO** : loss 0.5403 → 0.1272，下降 76%，收敛正常
- **GRPO (rule)**: reward 0.2048 → 1.3687，训练曲线漂亮，约 1 小时
- **PPO (rule)**: reward -0.25 → 0.2083，训练基本失败，约 1 小时

### ToolCall & Think 评测（80题，数据来源：训练结果xlsx）

| 类别 | 题数 | SFT | GRPO (rule) | DPO v2 |
|------|------|-----|-------------|--------|
| 单工具-数学计算 | 15 | 100% | 100% | 100% |
| 单工具-其他工具 | 15 | 100% | 100% | 100% |
| 多步链式调用 | 20 | 85% | 90% | 90% |
| Think 思考标签可靠性 | 30 | 40% | 86.7% | 90% |
| **合计** | **80** | **73.8%** | **92.5%** | **93.8%** |

> 官方评测版（不含 DPO/PPO，仅 SFT/GRPO/Agent）见 `eval_results/官方评测版（不含PPO DPO）.md`，仅供参考。

### 关键发现

1. **DPO 最稳定但需注意学习率**: lr=4e-8 训练无效，lr=5e-7 才有明显收敛。综合评测 93.8%，表现最好。
2. **GRPO 训练信号最强**: reward 曲线上升明显，但出现轻微 reward hacking。综合评测 92.5%。
3. **PPO 在小模型上失败**: Critic 网络在 64M 参数量下无法给出有效的优势估计
4. **RL 训练的核心收益在 Think 标签**: SFT 的 Think 可靠性仅 40%，DPO 提升到 90%，GRPO 提升到 86.7%
5. **对齐税**: 所有 RL 方法都存在"对齐税"——偏好对齐会轻微牺牲语义理解能力

## 为什么 PPO 失败

1. **Critic 网络能力不足**: 64M 参数的 Critic 无法准确估计状态价值
2. **GAE 优势估计噪声大**: 不准确的 Critic 导致 advantage 信号充满噪声
3. **双网络优化不稳定**: Actor 和 Critic 需要协同训练，小模型上容易震荡


## 结论

对于 64M 参数量的小模型：
- **推荐 DPO**: 训练快、稳定、不需要 RM，适合资源有限的场景
- **GRPO 适合特定任务**: 如 tool call，有明确的 ground truth 验证时效果好
- **PPO 不推荐**: 小模型上 Critic 网络是瓶颈，建议 >1B 参数再考虑
