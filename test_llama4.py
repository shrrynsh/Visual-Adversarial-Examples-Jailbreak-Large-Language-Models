import torch
import transformers.modeling_utils
old_to = transformers.modeling_utils.PreTrainedModel.to
def patched_to(self, *args, **kwargs):
    if getattr(self, "is_loaded_in_8bit", False) or getattr(self, "is_loaded_in_4bit", False):
        return self
    return old_to(self, *args, **kwargs)
transformers.modeling_utils.PreTrainedModel.to = patched_to

from transformers import LlamaForCausalLM

LlamaForCausalLM.from_pretrained("vicuna", torch_dtype=torch.float16, load_in_8bit=True, device_map={'': 0})
print("SUCCESS!")
