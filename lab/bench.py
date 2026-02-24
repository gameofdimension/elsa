import time
from transformers import AutoConfig, AutoTokenizer
import json
import torch
from lab.qwen import Qwen3ForCausalLM
from prompt import text as aprompt


def load_model(device, dtype):
    model_path = "/warehouse/Qwen3-1.7B"
    config = AutoConfig.from_pretrained(model_path)

    index_path = model_path + "/model.safetensors.index.json"
    with open(index_path, "r") as f:
        index_obj = json.load(f)
    all_safetensors = list(index_obj["weight_map"].values())

    from safetensors.torch import load_file
    state_dict = {}
    for fname in all_safetensors:
        part_state_dict = load_file(model_path + "/" + fname)
        state_dict.update(part_state_dict)

    model = Qwen3ForCausalLM(config=config).to(dtype=dtype, device=device)
    model.load_state_dict(state_dict)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    return tokenizer, model


def main():
    dtype = torch.bfloat16
    device = torch.device("cuda")
    tokenizer, model = load_model(device, dtype)
    inputs = tokenizer(aprompt, return_tensors="pt").to(model.device)

    model(inputs['input_ids'])

    count = 10
    begin = time.time()
    for _ in range(count):
        model(inputs['input_ids'])
    end = time.time()
    print(f"generate count {count}, mean time {(end - begin) / count:.3f}")


if __name__ == "__main__":
    main()
