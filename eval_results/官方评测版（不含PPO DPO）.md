# ToolCall 批量评测结果（80题，官方评测版，仅供参考）

> 本文件仅含 SFT / GRPO / Agent 三组权重，不含 DPO 和 PPO。
> 权威数据以 `训练结果，数据优先看这个文件.xlsx` 为准。

## 评分维度
- **called_any**: 是否调用了任何工具
- **called_correct**: 是否调用了正确的首个工具
- **chain_complete**: 多步任务是否完整链式执行（所有期望工具都被调用）

## 汇总

| 类别 | 题数 | SFT 链式完整 | GRPO 链式完整 | Agent 链式完整 |
|------|------|-------------|--------------|---------------|
| 单工具-数学计算 | 15 | 15 (100.0%) | 15 (100.0%) | 15 (100.0%) |
| 单工具-其他工具 | 15 | 15 (100.0%) | 15 (100.0%) | 14 (93.3%) |
| 多步链式调用 | 20 | 17 (85.0%) | 13 (65.0%) | 17 (85.0%) |
| 工具选择能力 | 15 | 15 (100.0%) | 14 (93.3%) | 15 (100.0%) |
| 中英文&表述变体 | 15 | 14 (93.3%) | 15 (100.0%) | 14 (93.3%) |
| **合计** | **80** | **76 (95.0%)** | **72 (90.0%)** | **75 (93.8%)** |

## 详细数据

### GRPO
- total=80, called_any=79, called_correct=78, chain_complete=72
- math_single: total=15, called_any=15, called_correct=15, chain_complete=15
- single_other: total=15, called_any=15, called_correct=15, chain_complete=15
- multi_step: total=20, called_any=20, called_correct=19, chain_complete=13
- tool_select: total=15, called_any=14, called_correct=14, chain_complete=14
- lang_variant: total=15, called_any=15, called_correct=15, chain_complete=15

### SFT (full_sft)
- total=80, called_any=80, called_correct=79, chain_complete=76
- math_single: total=15, called_any=15, called_correct=15, chain_complete=15
- single_other: total=15, called_any=15, called_correct=15, chain_complete=15
- multi_step: total=20, called_any=20, called_correct=20, chain_complete=17
- tool_select: total=15, called_any=15, called_correct=15, chain_complete=15
- lang_variant: total=15, called_any=15, called_correct=14, chain_complete=14
