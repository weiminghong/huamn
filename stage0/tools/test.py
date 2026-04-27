import os
import torch
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

model_path = "/share2/program/models/Qwen3.5-4B"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

print("step1: load processor")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print("step2: load model")
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path,
    trust_remote_code=True,
)
print("step3: move to cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("done")