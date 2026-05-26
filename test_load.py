import torch
from transformers import AutoConfig, LlamaForCausalLM, BitsAndBytesConfig

# Try to load vicuna checkpoint simply
model_name = "vicuna" # assuming vicuna is a local folder or replace with actual
try:
    model = LlamaForCausalLM.from_pretrained(
        "vicuna",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
    )
    print("Loaded with device_map auto successfully")
except Exception as e:
    import traceback
    traceback.print_exc()

