import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, AutoConfig, Qwen3ForCausalLM
from prompt import text as aprompt


def inspect():
    model_path = "/warehouse/Qwen3-1.7B"
    config: Qwen3Config = AutoConfig.from_pretrained(model_path)
    print(config)
    print(config._attn_implementation)
    print("=" * 100)

    model = Qwen3ForCausalLM.from_pretrained(model_path)
    print(model.config)
    print(model.config._attn_implementation)


def load_model():
    model_path = "/warehouse/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda()
    return tokenizer, model


def run_prefill(model: Qwen3ForCausalLM, inputs):
    outputs = model(
        **inputs,
    )
    return outputs


def main():
    count = 10
    tokenizer, model = load_model()
    inputs = tokenizer(aprompt, return_tensors="pt").to(model.device)

    run_prefill(model, inputs)

    begin = time.time()
    for i in range(count):
        outputs = run_prefill(model, inputs)
    end = time.time()

    print(f"generate count {count}, mean time {(end - begin) / count:.3f}")


if __name__ == "__main__":
    main()
