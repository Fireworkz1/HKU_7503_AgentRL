"""
批量评估对比脚本：对比 SFT / PPO / GRPO 等多组权重的表现
用法：python scripts/eval_compare.py --device cuda
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import random
import warnings
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed
warnings.filterwarnings('ignore')

# 要对比的权重配置
WEIGHT_CONFIGS = [
    {"name": "SFT", "weight": "full_sft"},
    {"name": "PPO(official-RM)", "weight": "ppo_actor"},
    {"name": "GRPO(official-RM)", "weight": "grpo"},
    {"name": "PPO(rule)", "weight": "ppo_actor_rule"},
    {"name": "GRPO(rule)", "weight": "grpo_rule"},
]

# 主观问答 prompts
QA_PROMPTS = [
    '你有什么特长？',
    '为什么天空是蓝色的',
    '请用Python写一个计算斐波那契数列的函数',
    '解释一下"光合作用"的基本过程',
    '比较一下猫和狗作为宠物的优缺点',
    '解释什么是机器学习',
    '鲁迅的《狂人日记》是如何批判封建礼教的？',
    '推荐一些中国的美食',
]

# 数学计算 prompts（直接计算，不用 tool call）
MATH_PROMPTS = [
    {"q": "计算 94 - 35 等于多少？", "answer": 59},
    {"q": "计算 3 的平方是多少？", "answer": 9},
    {"q": "计算 29 + 64 等于多少？", "answer": 93},
    {"q": "计算 10 的平方是多少？", "answer": 100},
    {"q": "计算 59 乘以 48 等于多少？", "answer": 2832},
    {"q": "计算 72 乘以 91 等于多少？", "answer": 6552},
    {"q": "计算 180 除以 12 等于多少？", "answer": 15},
    {"q": "计算 5 的立方是多少？", "answer": 125},
    {"q": "计算 11 的平方是多少？", "answer": 121},
    {"q": "计算 72 + 10 等于多少？", "answer": 82},
    {"q": "计算 17 的平方是多少？", "answer": 289},
    {"q": "计算 14 的立方是多少？", "answer": 2744},
    {"q": "计算 256 乘以 37 等于多少？", "answer": 9472},
    {"q": "计算 1000 除以 8 等于多少？", "answer": 125},
    {"q": "计算 99 + 101 等于多少？", "answer": 200},
]


def load_model(weight_name, args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=False,
    ))
    ckp = f'./{args.save_dir}/{weight_name}_{args.hidden_size}.pth'
    if not os.path.exists(ckp):
        return None, tokenizer
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    return model.half().eval().to(args.device), tokenizer


def generate_response(model, tokenizer, prompt, args, seed=42):
    setup_seed(seed)
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=0.95, temperature=0.85, repetition_penalty=1.0
        )
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response


def extract_number(text):
    """从回复中提取数字答案"""
    import re
    # 去掉 think 标签内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 找所有数字（包括负数和小数）
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        # 取最后一个数字（通常是最终答案）
        try:
            val = float(numbers[-1])
            return int(val) if val == int(val) else val
        except:
            pass
    return None


def run_qa_eval(configs, args):
    """主观问答评估"""
    print("\n" + "=" * 80)
    print("主观问答对比")
    print("=" * 80)

    results = {}
    for cfg in configs:
        name = cfg["name"]
        weight = cfg["weight"]
        model, tokenizer = load_model(weight, args)
        if model is None:
            print(f"\n[SKIP] {name}: 权重文件不存在")
            continue
        results[name] = []
        print(f"\n--- {name} ({weight}) ---")
        for prompt in QA_PROMPTS:
            resp = generate_response(model, tokenizer, prompt, args)
            results[name].append(resp)
            print(f"\n[Q]: {prompt}")
            print(f"[{name}]: {resp[:300]}{'...' if len(resp) > 300 else ''}")
        del model
        torch.cuda.empty_cache()

    return results


def run_math_eval(configs, args):
    """数学计算评估"""
    print("\n" + "=" * 80)
    print("数学计算评估")
    print("=" * 80)

    scores = {}
    details = {}
    for cfg in configs:
        name = cfg["name"]
        weight = cfg["weight"]
        model, tokenizer = load_model(weight, args)
        if model is None:
            print(f"\n[SKIP] {name}: 权重文件不存在")
            continue
        correct = 0
        total = len(MATH_PROMPTS)
        details[name] = []
        for i, item in enumerate(MATH_PROMPTS):
            resp = generate_response(model, tokenizer, item["q"], args)
            pred = extract_number(resp)
            gt = item["answer"]
            ok = (pred == gt)
            if ok:
                correct += 1
            mark = "✅" if ok else "❌"
            details[name].append({"q": item["q"], "gt": gt, "pred": pred, "ok": ok})
            print(f"[{name}] {i + 1}/{total} | {mark} | {item['q'][:20]}... | gt={gt} | pred={pred}")
        scores[name] = (correct, total)
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("数学计算汇总")
    print("=" * 80)
    for name, (c, t) in scores.items():
        print(f"  {name}: {c}/{t} = {c / t * 100:.1f}%")

    return scores, details


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_from', default='model', type=str)
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_new_tokens', default=512, type=int)
    parser.add_argument('--open_thinking', default=0, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--eval_type', default='all', choices=['qa', 'math', 'all'], type=str)
    parser.add_argument('--weights', default=None, type=str, help="逗号分隔的权重名，如 full_sft,grpo,ppo_actor")
    args = parser.parse_args()

    if args.weights:
        configs = []
        for w in args.weights.split(','):
            w = w.strip()
            configs.append({"name": w, "weight": w})
    else:
        configs = WEIGHT_CONFIGS

    if args.eval_type in ('qa', 'all'):
        run_qa_eval(configs, args)
    if args.eval_type in ('math', 'all'):
        run_math_eval(configs, args)


if __name__ == "__main__":
    main()
