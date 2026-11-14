"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel


def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,  # Or your target model path/name
    max_seq_length=max_seq_length,  # Set maximum sequence length
    )

    return model, tokenizer

# Step 2 - count_total_parameters
import numpy as np
def count_total_parameters(model):
    count_ = 0
    for param in model.parameters():
        count_ += np.prod(param.size())
    return int(count_)

# Step 3 - is_model_4bit_quantized
# from transformers import BitsAndBytesConfig

from bitsandbytes.nn import Linear4bit

def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    for name, module in model.named_modules():
        if isinstance(module, Linear4bit):
            return True
    return False

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    # TODO: return the list of attention projection module names LoRA should adapt
    return ['q_proj','k_proj','v_proj','o_proj']

# Step 6 - attach_lora_adapters
# from FastLanguageModel import get_peft_model
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # TODO: wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules
    return FastLanguageModel.get_peft_model(model=model, r=r, lora_alpha=lora_alpha, target_modules = get_lora_target_modules() if target_modules is None else target_modules)

# Step 7 - count_trainable_parameters
import numpy as np
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    count_ = 0
    for param in model.parameters():
        if param.requires_grad:
            count_ += np.prod(param.size())
    
    return int(count_)

# Step 8 - trainable_fraction
import numpy as np
def trainable_fraction(trainable_count, total_count):
    return (trainable_count / total_count)

# Step 9 - build_instruction_examples (not yet solved)
# TODO: implement

# Step 10 - format_instruction_example (not yet solved)
# TODO: implement

# Step 11 - format_all_examples (not yet solved)
# TODO: implement

# Step 12 - build_text_dataset (not yet solved)
# TODO: implement

# Step 13 - tokenize_text (not yet solved)
# TODO: implement

# Step 14 - count_tokens (not yet solved)
# TODO: implement

# Step 15 - build_training_arguments (not yet solved)
# TODO: implement

# Step 16 - build_sft_trainer (not yet solved)
# TODO: implement

# Step 17 - run_sft_training (not yet solved)
# TODO: implement

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

