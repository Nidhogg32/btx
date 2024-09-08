import torch
import torch.nn as nn
from modeling_qwen2_btx import Qwen2BtxForCausalLLM
from modeling_qwen2 import Qwen2Model
from configuration_qwen2_btx import Qwen2Config
from torchvision.transforms import ToTensor
from transformers import Qwen2TokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM
local_tokenizer_path = "/home/roo/dream/zqw/btx-qwen2-7b/local_tokenizer"
tokenizer = Qwen2TokenizerFast.from_pretrained(local_tokenizer_path)
a=tokenizer("Hello world")["input_ids"]
a_tensor = torch.tensor(a)
a_tensor=a_tensor.unsqueeze(0)
# 初始化融合模型
model_qwen = AutoModelForCausalLM.from_pretrained("/home/roo/dream/zqw/qwen/qwen2-7B/qwen/Qwen2-7B", torch_dtype=torch.bfloat16)
# print(model_qwen)
# 加载多个原始模型的权重
fused_model = Qwen2BtxForCausalLLM(Qwen2Config())
# print(fused_model)
qwen_state_dic_01=model_qwen.state_dict().copy()
qwen_state_dic_02=qwen_state_dic_01.copy()
qwen_state_dic_03=qwen_state_dic_01.copy()
moe_state_dic=fused_model.state_dict().copy()
# 融合
for i in qwen_state_dic_01.keys():
    if "mlp" in i:
        # 获取 layer 索引
        layer_index = i.split('.')[2]

        # 创建 expert 和 shared_expert 的新键
        expert_0_key = f"model.layers.{layer_index}.mlp.experts.0.{'.'.join(i.split('.')[4:])}"
        expert_1_key = f"model.layers.{layer_index}.mlp.experts.1.{'.'.join(i.split('.')[4:])}"
        shared_expert_key = f"model.layers.{layer_index}.mlp.shared_expert.{'.'.join(i.split('.')[4:])}"

        # 加载权重到 expert 和 shared_expert
        moe_state_dic[expert_0_key] = qwen_state_dic_01[i]
        moe_state_dic[expert_1_key] = qwen_state_dic_02[i]
        moe_state_dic[shared_expert_key] = qwen_state_dic_03[i]
    else:
        moe_state_dic[i]=(qwen_state_dic_01[i]+qwen_state_dic_02[i]+qwen_state_dic_03[i])*(1/3)
fused_model.load_state_dict(moe_state_dic)
y = fused_model(a_tensor)
print(y)

