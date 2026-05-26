import torch
from transformers import LlamaForCausalLM, BitsAndBytesConfig
import accelerate

print(accelerate.__version__)
import transformers
print(transformers.__version__)

try:
    print("Testing patched device_map with explicit dispatch")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    # The trick seems to be using device_map = {"": 0} or {"": "cuda:0"} 
    # instead of specifying submodules, to prevent .to() being called by 
    # accelerate when it finds something isn't on the target device
    model = LlamaForCausalLM.from_pretrained(
        "vicuna",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map={"": "cuda:0"}
    )
    print("Success dict map string cuda:0")
except Exception as e:
    import traceback
    traceback.print_exc()

