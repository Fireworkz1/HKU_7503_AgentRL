# PPO Mode Collapse 分析

## 现象

PPO 在 MiniMind（64M）上训练基本失败：
- Reward：-0.25 → 0.2083（几乎不动），对比 GRPO 0.2048 → 1.3687
- 官方 PPO 权重在 ToolCall 评测中输出崩溃，完全不可用
- Critic Loss 高方差（0.16~0.48），无法收敛

## 根因分析

### 1. Critic 网络能力不足

PPO 使用独立的 CriticModel（继承 MiniMindForCausalLM，替换 lm_head 为 value_head）。
64M 参数的 Critic 无法准确估计状态价值 V(s)，导致 GAE 优势信号充满噪声。

```
# PPO 的优势估计依赖 Critic 的 V(s)
delta = token_rewards[:, t] + gamma * V(s_{t+1}) - V(s_t)
advantage = sum(gamma * lambda)^t * delta_t
```

当 V(s) 估计不准时，advantage 信号本质上是随机的，Actor 无法从中学到有效策略。

### 2. 双网络优化死锁

- Actor 依赖 Critic 提供准确的优势估计来更新策略
- Critic 依赖 Actor 的 rollout 来学习价值函数
- 两者相互依赖，在小模型上形成恶性循环：Critic 不准 → Actor 乱更新 → rollout 质量差 → Critic 更不准

官方 README 原文：
> "Critic 需要逐步收敛才能准确估计价值函数，而 Actor 的策略更新又依赖 Critic 的优势估计。两者相互依赖，形成复杂的优化过程。"

### 3. 内存开销限制了训练效率

PPO 需要维护 Actor + Critic + Ref Model 三个网络（PPO 还有 Reward Model 时是四个），内存占用约为单网络的 1.5~2 倍。在 4090 24GB 上，这限制了 batch size 和序列长度，进一步削弱了训练信号质量。

## 对比：GRPO 为什么能成功

| 维度 | PPO | GRPO |
|------|-----|------|
| 优势估计 | GAE，依赖 Critic 的 V(s) | 组内相对优势，无需 Critic |
| 网络数量 | Actor + Critic + Ref | Policy + Ref |
| 收敛速度 | 慢（Critic 拖后腿） | 快（单网络直接优化） |
| 稳定性 | 差（双网络互相依赖） | 好（组内标准化天然稳定） |

GRPO 的核心创新：对每个 prompt 生成 N 个回答，在组内做 reward 标准化得到相对优势：
```
advantages = (rewards - mean_r) / (std_r + 1e-4)
```
完全绕过了 Critic 网络，消除了 PPO 的根本瓶颈。

## 可能的优化方向

### A. 共享 Backbone（降低 Critic 负担）
让 Actor 和 Critic 共享 Transformer backbone，只在最后一层分叉。减少 Critic 的独立参数量，利用 Actor 的表征能力。但本质上已经在向 GRPO 靠拢。

### B. 调整超参数
- 降低 `kl_coef`（当前 0.02）→ 给 Actor 更多探索空间
- 增大 `early_stop_kl`（当前 0.25）→ 避免过早停止更新
- 降低 Critic 学习率（当前 5e-7）→ 让 Critic 更稳定
- 但这些只能缓解，不能解决根本问题。

### C. 换用 CISPO（官方推荐）
CISPO 是 GRPO 的变体，解决了 ratio clipping 导致梯度截断的问题：
- 标准 PPO/GRPO：`clipped_ratio × advantage`，ratio 被 clip 后梯度为 0
- CISPO：`clipped_ratio × log_prob`，即使 ratio 被 clip，梯度仍然存在

## 结论

对于 64M 参数量的小模型，PPO 的 Critic 网络是根本性瓶颈，不是调参能解决的。
官方推荐：**用 GRPO 或 CISPO 替代 PPO**。PPO 更适合 >1B 参数的模型，Critic 有足够容量时才能发挥作用。
