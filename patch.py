import transformers.modeling_utils
import accelerate.big_modeling

old_to = transformers.modeling_utils.PreTrainedModel.to

def patched_to(self, *args, **kwargs):
    if getattr(self, "is_loaded_in_8bit", False) or getattr(self, "is_loaded_in_4bit", False):
        return self
    
    quant_method = getattr(self, "quantization_method", None)
    if quant_method is not None and "bitsandbytes" in str(quant_method).lower():
        return self

    return old_to(self, *args, **kwargs)

transformers.modeling_utils.PreTrainedModel.to = patched_to

# Also patch dispatch_model to not call to() if 8-bit.
old_dispatch = accelerate.big_modeling.dispatch_model

def patched_dispatch(model, *args, **kwargs):
    is_quantized = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    quant_method = getattr(model, "quantization_method", None)
    if quant_method is not None and "bitsandbytes" in str(quant_method).lower():
        is_quantized = True

    if is_quantized:
        # Already loaded on GPUs if using bitsandbytes, prevent .to call
        # but we still want hooks if device_map is set, so we just wrap model.to temporarily
        orig_to = model.to
        model.to = lambda *x, **y: model
        try:
            return old_dispatch(model, *args, **kwargs)
        finally:
            model.to = orig_to
    return old_dispatch(model, *args, **kwargs)

accelerate.big_modeling.dispatch_model = patched_dispatch
try:
    transformers.modeling_utils.dispatch_model = patched_dispatch
except AttributeError:
    pass
