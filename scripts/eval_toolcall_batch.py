import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import time
import random
import argparse
import warnings
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

# ==================== 工具定义（复用 eval_toolcall.py） ====================
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如123+456、2**10、sqrt(144)"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前日期和时间，支持指定时区", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "时区名称，如Asia/Shanghai、America/New_York", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成指定范围内的随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "最小值", "default": 0}, "max": {"type": "integer", "description": "最大值", "default": 100}}, "required": []}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本的字符数和单词数", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要统计的文本"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "进行单位换算，支持长度、重量、温度等", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "要转换的数值"}, "from_unit": {"type": "string", "description": "源单位，如km、miles、kg、pounds、celsius、fahrenheit"}, "to_unit": {"type": "string", "description": "目标单位"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取指定城市的当前天气信息，包括温度、湿度和天气状况", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "城市名称，如北京、上海、New York"}, "unit": {"type": "string", "description": "温度单位，celsius或fahrenheit", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询两种货币之间的实时汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "源货币代码，如USD、CNY、EUR"}, "to_currency": {"type": "string", "description": "目标货币代码，如USD、CNY、EUR"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "将文本翻译成目标语言", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要翻译的文本"}, "target_language": {"type": "string", "description": "目标语言，如english、chinese、japanese、french"}}, "required": ["text", "target_language"]}}},
]

MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("²", "**2").replace("³", "**3").replace("（", "(").replace("）", ")")))},
    "get_current_time": lambda args: {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "random_number": lambda args: {"result": random.randint(int(args.get("min", 0)), int(args.get("max", 100)))},
    "text_length": lambda args: {"characters": len(args.get("text", "")), "words": len(args.get("text", "").split())},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * 0.621371, 2), "from": f"{args.get('value', 0)} {args.get('from_unit', '')}", "to": args.get("to_unit", "")},
    "get_current_weather": lambda args: {"city": args.get("location"), "temperature": "22°C", "humidity": "65%", "condition": "晴"},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency", ""), "to": args.get("to_currency", ""), "rate": 7.15},
    "translate_text": lambda args: {"translated": "hello world"},
}

TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}
def get_tools(names):
    return [TOOL_MAP[n] for n in names]

# ==================== 80 个测试用例 ====================
TEST_CASES = [
    # ===== 1. 单工具-数学计算 (15题) =====
    {"prompt": "帮我算一下 128 加 256 等于多少", "tools": ["calculate_math", "get_current_time"], "expect_tool": "calculate_math", "category": "math_single"},
    {"prompt": "99 乘以 88 是多少？", "tools": ["calculate_math", "random_number"], "expect_tool": "calculate_math", "category": "math_single"},
    {"prompt": "2的10次方是多少", "tools": ["calculate_math", "text_length"], "expect_tool": "calculate_math", "category": "math_single"},
    {"prompt": "1024 除以 32 等于几", "tools": ["calculate_math", "unit_converter"], "expect_tool": "calculate_math", "category": "math_single"},
    {"prompt": "帮我计算 (15+25)*3 的结果", "tools": ["calculate_math", "get_current_time"], "expect_tool": "calculate_math", "category": "math_single"},
    {"prompt": "7的3次方减去100等于多少", "tools": ["calculate_math", "translate_text"], "expect_tool": "calculate_math", "category": "math_single"},
    {"prompt": "请计算 500-137 的差", "tools": ["calculate_math", "get_exchange_rate"], "expect_tool": "calculate_math", "category": "math_single"},
    {"prompt": "45乘以20再加上100是多少", "tools": ["calculate_math", "random_number"], "expect_tool": "calculate_math", "category": "math_single"},
    {"prompt": "3600除以60再除以60等于几", "tools": ["calculate_math", "get_current_weather"], "expect_tool": "calculate_math", "category": "math_single"},
    {"prompt": "帮我算 12*12+13*13", "tools": ["calculate_math", "text_length"], "expect_tool": "calculate_math", "category": "math_single"},
    {"prompt": "计算一下 2**8 是多少", "tools": ["calculate_math", "unit_converter"], "expect_tool": "calculate_math", "category": "math_single"},
    {"prompt": "100的平方是多少", "tools": ["calculate_math", "get_current_time"], "expect_tool": "calculate_math", "category": "math_single"},
    {"prompt": "帮我算 (100+200)*5-500", "tools": ["calculate_math", "translate_text"], "expect_tool": "calculate_math", "category": "math_single"},
    {"prompt": "999加1等于多少", "tools": ["calculate_math", "random_number"], "expect_tool": "calculate_math", "category": "math_single"},
    {"prompt": "50*50 等于多少", "tools": ["calculate_math", "get_exchange_rate"], "expect_tool": "calculate_math", "category": "math_single"},

    # ===== 2. 单工具-其他工具 (15题) =====
    {"prompt": "帮我把50公里换算成英里", "tools": ["unit_converter", "calculate_math"], "expect_tool": "unit_converter", "category": "single_other"},
    {"prompt": "200磅等于多少公斤", "tools": ["unit_converter", "get_current_time"], "expect_tool": "unit_converter", "category": "single_other"},
    {"prompt": "上海现在天气如何", "tools": ["get_current_weather", "get_current_time"], "expect_tool": "get_current_weather", "category": "single_other"},
    {"prompt": "深圳今天天气怎么样", "tools": ["get_current_weather", "translate_text"], "expect_tool": "get_current_weather", "category": "single_other"},
    {"prompt": "纽约现在几点了", "tools": ["get_current_time", "get_current_weather"], "expect_tool": "get_current_time", "category": "single_other"},
    {"prompt": "东京现在是什么时间", "tools": ["get_current_time", "random_number"], "expect_tool": "get_current_time", "category": "single_other"},
    {"prompt": "欧元兑美元汇率是多少", "tools": ["get_exchange_rate", "calculate_math"], "expect_tool": "get_exchange_rate", "category": "single_other"},
    {"prompt": "日元兑人民币的汇率", "tools": ["get_exchange_rate", "get_current_time"], "expect_tool": "get_exchange_rate", "category": "single_other"},
    {"prompt": "把'谢谢'翻译成日语", "tools": ["translate_text", "text_length"], "expect_tool": "translate_text", "category": "single_other"},
    {"prompt": "把'早上好'翻译成法语", "tools": ["translate_text", "get_current_time"], "expect_tool": "translate_text", "category": "single_other"},
    {"prompt": "帮我统计一下'人工智能改变世界'这句话有多少个字", "tools": ["text_length", "translate_text"], "expect_tool": "text_length", "category": "single_other"},
    {"prompt": "计算'hello world'的字符数", "tools": ["text_length", "calculate_math"], "expect_tool": "text_length", "category": "single_other"},
    {"prompt": "帮我生成一个1到100的随机数", "tools": ["random_number", "calculate_math"], "expect_tool": "random_number", "category": "single_other"},
    {"prompt": "给我一个50到500之间的随机数", "tools": ["random_number", "text_length"], "expect_tool": "random_number", "category": "single_other"},
    {"prompt": "把25摄氏度换算成华氏度", "tools": ["unit_converter", "get_current_weather"], "expect_tool": "unit_converter", "category": "single_other"},

    # ===== 3. 多步链式调用 (20题) =====
    {"prompt": "帮我生成一个1到500的随机数，然后计算它的立方", "tools": ["random_number", "calculate_math"], "expect_tool": "random_number", "expect_chain": ["random_number", "calculate_math"], "category": "multi_step"},
    {"prompt": "生成一个10到100的随机数，再算它乘以7的结果", "tools": ["random_number", "calculate_math"], "expect_tool": "random_number", "expect_chain": ["random_number", "calculate_math"], "category": "multi_step"},
    {"prompt": "先算一下 25*4 等于多少，然后把结果从公里换算成英里", "tools": ["calculate_math", "unit_converter"], "expect_tool": "calculate_math", "expect_chain": ["calculate_math", "unit_converter"], "category": "multi_step"},
    {"prompt": "计算 60*24 的结果，再把它从摄氏度转成华氏度", "tools": ["calculate_math", "unit_converter"], "expect_tool": "calculate_math", "expect_chain": ["calculate_math", "unit_converter"], "category": "multi_step"},
    {"prompt": "查一下北京天气，顺便告诉我现在几点", "tools": ["get_current_weather", "get_current_time"], "expect_tool": "get_current_weather", "expect_chain": ["get_current_weather", "get_current_time"], "category": "multi_step"},
    {"prompt": "上海天气怎么样？现在是什么时间？", "tools": ["get_current_weather", "get_current_time"], "expect_tool": "get_current_weather", "expect_chain": ["get_current_weather", "get_current_time"], "category": "multi_step"},
    {"prompt": "查一下美元兑欧元汇率，然后帮我算1000美元能换多少欧元", "tools": ["get_exchange_rate", "calculate_math"], "expect_tool": "get_exchange_rate", "expect_chain": ["get_exchange_rate", "calculate_math"], "category": "multi_step"},
    {"prompt": "查英镑兑人民币汇率，再算5000英镑等于多少人民币", "tools": ["get_exchange_rate", "calculate_math"], "expect_tool": "get_exchange_rate", "expect_chain": ["get_exchange_rate", "calculate_math"], "category": "multi_step"},
    {"prompt": "把'机器学习'翻译成英文，然后统计翻译结果的字符数", "tools": ["translate_text", "text_length"], "expect_tool": "translate_text", "expect_chain": ["translate_text", "text_length"], "category": "multi_step"},
    {"prompt": "翻译'深度学习'为英文，再数一下有几个字符", "tools": ["translate_text", "text_length"], "expect_tool": "translate_text", "expect_chain": ["translate_text", "text_length"], "category": "multi_step"},
    {"prompt": "生成一个1到1000的随机数，然后算它除以7的结果", "tools": ["random_number", "calculate_math", "text_length"], "expect_tool": "random_number", "expect_chain": ["random_number", "calculate_math"], "category": "multi_step"},
    {"prompt": "先查广州天气，再查现在时间", "tools": ["get_current_weather", "get_current_time", "translate_text"], "expect_tool": "get_current_weather", "expect_chain": ["get_current_weather", "get_current_time"], "category": "multi_step"},
    {"prompt": "帮我算 150*3，然后把结果公里数换算成英里", "tools": ["calculate_math", "unit_converter", "get_current_time"], "expect_tool": "calculate_math", "expect_chain": ["calculate_math", "unit_converter"], "category": "multi_step"},
    {"prompt": "生成一个1到200的随机数，计算它的平方", "tools": ["random_number", "calculate_math", "unit_converter"], "expect_tool": "random_number", "expect_chain": ["random_number", "calculate_math"], "category": "multi_step"},
    {"prompt": "查一下人民币兑日元汇率，算10000人民币能换多少日元", "tools": ["get_exchange_rate", "calculate_math", "get_current_time"], "expect_tool": "get_exchange_rate", "expect_chain": ["get_exchange_rate", "calculate_math"], "category": "multi_step"},
    {"prompt": "把'自然语言处理'翻译成英文，统计字符数", "tools": ["translate_text", "text_length", "calculate_math"], "expect_tool": "translate_text", "expect_chain": ["translate_text", "text_length"], "category": "multi_step"},
    {"prompt": "查东京天气和当前时间", "tools": ["get_current_weather", "get_current_time", "translate_text"], "expect_tool": "get_current_weather", "expect_chain": ["get_current_weather", "get_current_time"], "category": "multi_step"},
    {"prompt": "生成一个随机数再算它加100的结果", "tools": ["random_number", "calculate_math"], "expect_tool": "random_number", "expect_chain": ["random_number", "calculate_math"], "category": "multi_step"},
    {"prompt": "帮我把'计算机科学'翻译成英文，然后告诉我翻译后有多少个单词", "tools": ["translate_text", "text_length", "random_number"], "expect_tool": "translate_text", "expect_chain": ["translate_text", "text_length"], "category": "multi_step"},
    {"prompt": "查一下伦敦天气，同时把温度22摄氏度换算成华氏度", "tools": ["get_current_weather", "unit_converter", "get_current_time"], "expect_tool": "get_current_weather", "expect_chain": ["get_current_weather", "unit_converter"], "category": "multi_step"},

    # ===== 4. 工具选择能力 (15题) =====
    {"prompt": "我想知道 365 乘以 24 是多少", "tools": ["calculate_math", "get_current_weather", "translate_text", "random_number"], "expect_tool": "calculate_math", "category": "tool_select"},
    {"prompt": "今天杭州天气好不好", "tools": ["get_current_weather", "calculate_math", "get_exchange_rate", "text_length"], "expect_tool": "get_current_weather", "category": "tool_select"},
    {"prompt": "现在是几点钟", "tools": ["get_current_time", "calculate_math", "unit_converter", "translate_text"], "expect_tool": "get_current_time", "category": "tool_select"},
    {"prompt": "帮我把10英里换算成公里", "tools": ["unit_converter", "calculate_math", "get_current_weather", "random_number"], "expect_tool": "unit_converter", "category": "tool_select"},
    {"prompt": "美元兑日元多少钱", "tools": ["get_exchange_rate", "calculate_math", "get_current_time", "text_length"], "expect_tool": "get_exchange_rate", "category": "tool_select"},
    {"prompt": "把'人工智能'翻译成英文", "tools": ["translate_text", "calculate_math", "get_current_weather", "random_number"], "expect_tool": "translate_text", "category": "tool_select"},
    {"prompt": "帮我数一下'强化学习是人工智能的重要分支'有多少个字", "tools": ["text_length", "calculate_math", "translate_text", "get_current_time"], "expect_tool": "text_length", "category": "tool_select"},
    {"prompt": "给我一个随机数", "tools": ["random_number", "calculate_math", "get_current_weather", "unit_converter"], "expect_tool": "random_number", "category": "tool_select"},
    {"prompt": "帮我算一下 77*13", "tools": ["calculate_math", "translate_text", "get_exchange_rate", "text_length"], "expect_tool": "calculate_math", "category": "tool_select"},
    {"prompt": "成都今天下雨吗", "tools": ["get_current_weather", "get_current_time", "calculate_math", "translate_text"], "expect_tool": "get_current_weather", "category": "tool_select"},
    {"prompt": "把100华氏度转成摄氏度", "tools": ["unit_converter", "calculate_math", "get_current_time", "get_exchange_rate"], "expect_tool": "unit_converter", "category": "tool_select"},
    {"prompt": "欧元兑人民币汇率查一下", "tools": ["get_exchange_rate", "get_current_weather", "random_number", "text_length"], "expect_tool": "get_exchange_rate", "category": "tool_select"},
    {"prompt": "翻译'你好'为法语", "tools": ["translate_text", "text_length", "calculate_math", "random_number"], "expect_tool": "translate_text", "category": "tool_select"},
    {"prompt": "帮我算 2**20", "tools": ["calculate_math", "unit_converter", "get_current_weather", "translate_text"], "expect_tool": "calculate_math", "category": "tool_select"},
    {"prompt": "伦敦现在什么时间", "tools": ["get_current_time", "get_current_weather", "calculate_math", "get_exchange_rate"], "expect_tool": "get_current_time", "category": "tool_select"},

    # ===== 5. 中英文混合 & 表述变体 (15题) =====
    {"prompt": "What is 256 times 128?", "tools": ["calculate_math", "get_current_time"], "expect_tool": "calculate_math", "category": "lang_variant"},
    {"prompt": "Calculate 999 minus 111", "tools": ["calculate_math", "random_number"], "expect_tool": "calculate_math", "category": "lang_variant"},
    {"prompt": "How is the weather in Beijing today?", "tools": ["get_current_weather", "get_current_time"], "expect_tool": "get_current_weather", "category": "lang_variant"},
    {"prompt": "What time is it now?", "tools": ["get_current_time", "calculate_math"], "expect_tool": "get_current_time", "category": "lang_variant"},
    {"prompt": "Convert 50 miles to kilometers", "tools": ["unit_converter", "calculate_math"], "expect_tool": "unit_converter", "category": "lang_variant"},
    {"prompt": "What is the USD to EUR exchange rate?", "tools": ["get_exchange_rate", "get_current_time"], "expect_tool": "get_exchange_rate", "category": "lang_variant"},
    {"prompt": "Translate 'good morning' to Chinese", "tools": ["translate_text", "text_length"], "expect_tool": "translate_text", "category": "lang_variant"},
    {"prompt": "Generate a random number between 1 and 100", "tools": ["random_number", "calculate_math"], "expect_tool": "random_number", "category": "lang_variant"},
    {"prompt": "算个数：64的平方根是多少", "tools": ["calculate_math", "unit_converter"], "expect_tool": "calculate_math", "category": "lang_variant"},
    {"prompt": "麻烦帮忙看看现在啥时候了", "tools": ["get_current_time", "get_current_weather"], "expect_tool": "get_current_time", "category": "lang_variant"},
    {"prompt": "我想把30公斤换算成磅", "tools": ["unit_converter", "calculate_math"], "expect_tool": "unit_converter", "category": "lang_variant"},
    {"prompt": "Check the weather in New York and convert 0 celsius to fahrenheit", "tools": ["get_current_weather", "unit_converter", "get_current_time"], "expect_tool": "get_current_weather", "category": "lang_variant"},
    {"prompt": "帮我查一下英镑兑美元的汇率", "tools": ["get_exchange_rate", "calculate_math"], "expect_tool": "get_exchange_rate", "category": "lang_variant"},
    {"prompt": "How many characters are in 'artificial intelligence'?", "tools": ["text_length", "translate_text"], "expect_tool": "text_length", "category": "lang_variant"},
    {"prompt": "Generate a random number from 100 to 999, then square it", "tools": ["random_number", "calculate_math", "text_length"], "expect_tool": "random_number", "category": "lang_variant"},
]

CATEGORY_NAMES = {
    "math_single": "单工具-数学计算",
    "single_other": "单工具-其他工具",
    "multi_step": "多步链式调用",
    "tool_select": "工具选择能力",
    "lang_variant": "中英文&表述变体",
}

# ==================== 模型与推理 ====================
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def parse_tool_calls(text):
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))
        except Exception:
            pass
    return calls


def execute_tool(call):
    name = call.get("name", "")
    try:
        args = call.get("arguments", {})
        args = json.loads(args) if isinstance(args, str) else args
    except Exception:
        args = {}
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return {"error": f"未知工具: {name}"}
    try:
        return fn(args)
    except Exception as e:
        return {"error": f"工具执行失败: {str(e)[:80]}"}


def generate_silent(model, tokenizer, messages, tools, args):
    """静默生成，不打印 streamer 输出"""
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature
        )
    return tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)


def run_case_auto(prompt, tools, args, model, tokenizer, max_turns=5):
    """自动运行一个 case，返回 (所有 tool_call 列表, 最终回复)"""
    messages = [{"role": "user", "content": prompt}]
    all_tool_calls = []
    final_content = ""
    for _ in range(max_turns):
        content = generate_silent(model, tokenizer, messages, tools, args)
        tool_calls = parse_tool_calls(content)
        if not tool_calls:
            final_content = content
            break
        all_tool_calls.extend(tool_calls)
        messages.append({"role": "assistant", "content": content})
        for tc in tool_calls:
            result = execute_tool(tc)
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
    else:
        final_content = content
    return all_tool_calls, final_content


def evaluate_case(tool_calls, final_content, expect_tool, expect_chain=None):
    """评估单个 case：
    - called_any: 是否调用了任何工具
    - called_correct: 是否调用了正确的首个工具
    - chain_complete: 多步任务是否完整链式执行（所有期望工具都被调用）
    """
    called_any = len(tool_calls) > 0
    called_correct = any(tc.get("name") == expect_tool for tc in tool_calls)
    called_names = [tc.get("name", "") for tc in tool_calls]
    if expect_chain:
        chain_complete = all(t in called_names for t in expect_chain)
    else:
        chain_complete = called_correct  # 单步任务：调对就算完整
    return called_any, called_correct, chain_complete


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="MiniMind ToolCall 批量评测 (80题)")
    parser.add_argument('--load_from', default='../model', type=str)
    parser.add_argument('--save_dir', default='../out', type=str)
    parser.add_argument('--weight', default='full_sft', type=str)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int)
    parser.add_argument('--max_new_tokens', default=512, type=int)
    parser.add_argument('--temperature', default=0.9, type=float)
    parser.add_argument('--top_p', default=0.9, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  ToolCall 批量评测 | 权重: {args.weight} | 共 {len(TEST_CASES)} 题")
    print(f"{'='*60}")

    model, tokenizer = init_model(args)
    setup_seed(42)

    results = []
    cat_stats = {}

    for i, case in enumerate(TEST_CASES):
        prompt = case["prompt"]
        tools = get_tools(case["tools"])
        expect_tool = case["expect_tool"]
        expect_chain = case.get("expect_chain")
        category = case["category"]

        try:
            tool_calls, final_content = run_case_auto(prompt, tools, args, model, tokenizer)
            called_any, called_correct, chain_complete = evaluate_case(tool_calls, final_content, expect_tool, expect_chain)
        except Exception as e:
            tool_calls, final_content = [], str(e)
            called_any, called_correct, chain_complete = False, False, False

        if expect_chain:
            status = "✅" if chain_complete else ("⚠️" if called_correct else "❌")
        else:
            status = "✅" if called_correct else ("⚠️" if called_any else "❌")
        called_names = [tc.get("name", "?") for tc in tool_calls]
        chain_tag = f" chain={'✓' if chain_complete else '✗'}" if expect_chain else ""
        print(f"[{args.weight}] {i+1:2d}/{len(TEST_CASES)} | {status} | {category:15s} | expect={expect_tool:20s} | called={called_names}{chain_tag}")

        results.append({"idx": i+1, "category": category, "called_any": called_any, "called_correct": called_correct, "chain_complete": chain_complete})

        if category not in cat_stats:
            cat_stats[category] = {"total": 0, "called_any": 0, "called_correct": 0, "chain_complete": 0}
        cat_stats[category]["total"] += 1
        cat_stats[category]["called_any"] += int(called_any)
        cat_stats[category]["called_correct"] += int(called_correct)
        cat_stats[category]["chain_complete"] += int(chain_complete)

    # 汇总
    total = len(results)
    total_any = sum(r["called_any"] for r in results)
    total_correct = sum(r["called_correct"] for r in results)
    total_chain = sum(r["chain_complete"] for r in results)

    print(f"\n{'='*70}")
    print(f"  汇总 | 权重: {args.weight}")
    print(f"{'='*70}")
    print(f"{'类别':<20s} | {'总数':>4s} | {'调用工具':>8s} | {'工具正确':>8s} | {'链式完整':>8s} | {'完整率':>6s}")
    print(f"{'-'*70}")
    for cat, name in CATEGORY_NAMES.items():
        s = cat_stats.get(cat, {"total": 0, "called_any": 0, "called_correct": 0, "chain_complete": 0})
        rate = f"{s['chain_complete']/s['total']*100:.1f}%" if s['total'] > 0 else "N/A"
        print(f"{name:<18s} | {s['total']:>4d} | {s['called_any']:>8d} | {s['called_correct']:>8d} | {s['chain_complete']:>8d} | {rate:>6s}")
    print(f"{'-'*70}")
    rate_chain = f"{total_chain/total*100:.1f}%"
    print(f"{'合计':<18s} | {total:>4d} | {total_any:>8d} | {total_correct:>8d} | {total_chain:>8d} | {rate_chain:>6s}")
    print(f"{'='*70}")

    # 保存 JSON 结果
    out_path = f"../eval_results/toolcall_batch_{args.weight}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"weight": args.weight, "total": total, "called_any": total_any, "called_correct": total_correct,
                    "chain_complete": total_chain, "category_stats": cat_stats, "details": results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
