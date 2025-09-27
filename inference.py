import mindspore
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.peft import PeftModel
import gradio as gr

# ================== 1. 设置设备 ==================
mindspore.set_device('Ascend')  # 或 'GPU' / 'CPU'

# ================== 2. 加载基座模型 ==================
base_model_path = "./Qwen2.5-7B"
adapter_model_path = "./output/qwen2.5-7B/checkpoint-3800/adapter_model"

print(">>> 加载基座模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    ms_dtype=mindspore.bfloat16
)

print(">>> 加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# ================== 3. 加载 LoRA 微调权重 ==================
print(">>> 加载 LoRA adapter 权重...")
model = PeftModel.from_pretrained(base_model, adapter_model_path)

# ================== 4. 定义生成函数 ==================
def generate_response(user_input: str) -> str:
    if not user_input.strip():  # 空字符串或全是空格
        return "⚠️ 请输入问题后再提交。"
    
    prompt_messages = [
        {"role": "system", "content": "你是西电新生百事通，请根据问题给出答案"},
        {"role": "user", "content": user_input}
    ]
    
    # 构造模型输入
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="ms")
    
    # 推理生成
    outputs = model.generate(
        **inputs,
        max_length=512,
        do_sample=True,
        top_k=1
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# ================== 5. 模型预热 ==================
print(">>> 模型预热中...")
try:
    warmup_result = generate_response("你是谁？")
    print(">>> 预热完成，示例输出：", warmup_result[:100], "...")
except Exception as e:
    print(">>> 预热失败:", e)

# ================== 6. 创建 Gradio 界面 ==================
with gr.Blocks() as demo:
    gr.Markdown("西电新生百事通")
    
    with gr.Row():
        user_input = gr.Textbox(
            label="请输入你的问题", 
            placeholder="在这里输入问题...", 
            lines=2
        )
        submit_btn = gr.Button("生成回答")
    
    output_box = gr.Textbox(label="模型回答", lines=10)

    # 直接填充输入框，不会自动提交
    gr.Examples(
        examples=[
            ["新生入学后需要进行哪些方面的复查？"],
            ["实验班分流综合成绩计算公式"],
            ["进入教改卓越班或试点班的选拔标准是什么？"],
            ["如何激活统一认证账号？"],
            ["英语听力考试所用的调频耳机频率？"],
            ["英语高级班和普通班在考四级时间上有什么区别？"],
            ["入学体检时需要携带哪些材料？"],
            ["新生应如何缴纳和结算教材费用？"],
            ["校园乐跑对男生和女生的单次有效跑步要求分别是什么？"],
            ["申请转专业需要满足哪些条件？"],
            ["军训服装费的缴费金额和缴费时间是什么？"],
            ["学校统一办理的用于接收学校发放款项的银行卡是哪家银行的？"],
            ["如何进入西安电子科技大学图书馆？"],
            ["校内怎么取顺丰快递？"],
            ["社会实践必须完成多少次才能毕业？"],
            ["如何根据英语开学考试成绩确定英语课程的分班？"],
            ["体育课成绩包含什么"],
            ["如何使用校园网？"],
            ["在宿舍内可以使用电吹风机吗？"],
            ["我校的校训是什么？"],
            ["如何计算一门课程的学分绩？"],
            ["实验班年度考核不通过会怎样"],
            ["学生申请转专业需要满足哪些条件？"],
            ["卓越班学生是否有机会获得外校推免名额？"],
            ["体育课成绩的评定包含哪些考核要素？"],
            ["如何填写家庭经济困难学生认定申请表？"],
            ["家庭经济困难学生认定的申请流程具体包括哪些步骤？"],
        ],
        inputs=user_input
    )

    # 点击按钮才会提交
    submit_btn.click(generate_response, inputs=user_input, outputs=output_box)

demo.launch()