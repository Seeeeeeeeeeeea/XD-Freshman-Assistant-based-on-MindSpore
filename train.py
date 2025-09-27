import mindspore
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.dataset import load_dataset
from mindnlp.engine.callbacks import TrainerCallback, TrainerState, TrainerControl
from mindnlp.peft import LoraConfig, TaskType, get_peft_model
import os
from mindnlp.engine import TrainingArguments, Trainer

mindspore.set_device('Ascend')

model = AutoModelForCausalLM.from_pretrained("./Qwen2.5-7B", ms_dtype=mindspore.bfloat16)  

tokenizer = AutoTokenizer.from_pretrained("./Qwen2.5-7B", ms_dtype=mindspore.bfloat16)

# alpaca_data = load_dataset(path="json", data_files="alpaca_zh/alpaca_data_zh_51k.json")
alpaca_data = load_dataset(path="json", data_files="./formatted.json")
print(f"数据集列名：{alpaca_data.get_col_names()}")

def process_func(_instruction, _input, output):
    MAX_LENGTH = 1024
    prompt_messages = [
        {"role": "user", "content": f"{_instruction}\n{_input}"},
    ]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    response_text = f"{output}{tokenizer.eos_token}"
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
    response_ids = tokenizer(response_text, add_special_tokens=False)['input_ids']
    input_ids = prompt_ids + response_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + response_ids
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    padding_length = MAX_LENGTH - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length   
    return input_ids, attention_mask, labels

formatted_dataset = alpaca_data.map(
    operations=[process_func],
    input_columns=['instruction', 'input', 'output'], 
    output_columns=["input_ids", "attention_mask", "labels"],
    num_parallel_workers=16
)
print(formatted_dataset.get_col_names())

for input_ids, attention_mask, labels in formatted_dataset.create_tuple_iterator():
    print("input_ids:\n",input_ids)
    print("attention_mask:\n",attention_mask)
    print("labels:\n",labels)
    print("decoded input id:", tokenizer.decode(input_ids))
    break

# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 设置训练模式
    r=8, # Lora 秩
    lora_alpha=16, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)
print("LoraConfig:\n",config)

model = get_peft_model(model, config)

# 定义训练超参数
args = TrainingArguments(
    # 输出保存路径
    output_dir="./output/qwen2.5-7B",
    # 训练批大小
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    # 每100个step打印一次日志
    logging_steps=10,
    # # 训练epoch数
    num_train_epochs=10,
    # 模型权重保存长，1000个step保存一次模型权重
    save_steps=1900,
    # 学习率
    learning_rate=1e-4
)

PREFIX_CHECKPOINT_DIR = "checkpoint"
SAFE_WEIGHTS_NAME = "safe_model_qat.bin"
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )       

        # 保存adapter权重
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path, safe_serialization=True)

        # remove base model safetensors to free more space
        base_model_path = os.path.join(checkpoint_folder, SAFE_WEIGHTS_NAME)
        os.remove(base_model_path) if os.path.exists(base_model_path) else None

        return control

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=formatted_dataset,
    callbacks=[SavePeftModelCallback],
)

trainer.train()
