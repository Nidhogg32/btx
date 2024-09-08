import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import Qwen2TokenizerFast
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化分布式进程组
dist.init_process_group(backend="nccl")  # 使用NCCL作为后端，适用于NVIDIA GPU

# 获取当前进程的本地 GPU ID
local_rank = int(os.getenv("LOCAL_RANK", 0))

# 设置当前进程使用的 GPU
torch.cuda.set_device(local_rank)

# 加载分词器和模型
local_tokenizer_path = "/home/roo/dream/zqw/qwen/qwen2-7B/qwen/Qwen2-7B"
tokenizer = Qwen2TokenizerFast.from_pretrained(local_tokenizer_path)
model = AutoModelForCausalLM.from_pretrained("/home/roo/dream/zqw/qwen/qwen2-7B/qwen/Qwen2-7B", torch_dtype=torch.float16)
model = model.to(f'cuda:{local_rank}')

model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# 2. 加载和处理自定义的JSON数据集
def preprocess_function(examples):
    # 只使用 'instruction' 和 'answer' 字段进行输入的拼接
    inputs = ["Q: " + q + " A: " + a for q, a in zip(examples['instruction'], examples['answer'])]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    # 设置 labels
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    # 删除无关列
    if "id" in examples:
        del examples["id"]
    if "metrics" in examples:
        del examples["metrics"]
    if "type" in examples:
        del examples["type"]

    return model_inputs


# 加载和处理自定义的JSON数据集
dataset = load_dataset('json', data_files="wukong.json", split='train')
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 3. 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=1,  # 每设备批次大小
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,  # 启用梯度累积
    num_train_epochs=3,
    logging_dir='./logs',
    remove_unused_columns=False,
    fp16=True,  # 启用混合精度训练
    deepspeed="/home/roo/dream/zqw/btx-qwen2-7b/ds_config.json"  # 指定DeepSpeed配置文件
)

# 打印当前设备信息
print(f"Using device: {local_rank}, total GPUs: {torch.cuda.device_count()}")

# 4. 使用Trainer进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset
)

# 5. 训练模型
trainer.train()

# 6. 保存微调后的模型
model.module.save_pretrained("./finetuned_qwen_model")  # 使用 .module 保存DDP模型的权重
tokenizer.save_pretrained("./finetuned_qwen_model")
