import torch
from transformers import LlamaForCausalLM

try:
    print("Testing patched device_map with explicit dict and keeping config as is")
    # Setting device_map explicitly stops transformers from initializing a device_map and dispatching
    model = LlamaForCausalLM.from_pretrained(
        "vicuna",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map={'': "cuda:0"} # map all layers to first CUDA device explicitly
    )
    print("Success dict map string cuda:0")
except Exception as e:
    import traceback
    traceback.print_exc()

