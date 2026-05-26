import torch
from transformers import LlamaForCausalLM, BitsAndBytesConfig

try:
    print("Testing device_map='auto'")
    model = LlamaForCausalLM.from_pretrained(
        "vicuna",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
    )
    print("Success auto")
except Exception as e:
    import traceback
    traceback.print_exc()

try:
    print("Testing patched device_map")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    model = LlamaForCausalLM.from_pretrained(
        "vicuna",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map={'model': 0, 'lm_head': 0}
    )
    print("Success dict map")
except Exception as e:
    import traceback
    traceback.print_exc()

