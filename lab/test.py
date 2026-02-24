import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models import Qwen3ForCausalLM

from lab.qwen import Qwen3ForCausalLM as MyQwen3ForCausalLM
from prompt import text as aprompt


def load_model(dtype):
    model_path = "/warehouse/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).cuda()
    return tokenizer, model


def load_my_model(device, dtype):
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

    my_model = MyQwen3ForCausalLM(config=config).to(dtype=dtype, device=device)
    my_model.load_state_dict(state_dict)
    return my_model


def run_inference(model: Qwen3ForCausalLM, inputs):
    with torch.no_grad():
        outputs = model(
            **inputs,
        )
    return outputs.logits


def run_my_inference(model: MyQwen3ForCausalLM, input_ids):
    logits = model(
        input_ids
    )
    return logits


def main():
    dtype = torch.float32
    tokenizer, model = load_model(dtype)
    my_model = load_my_model(model.device, dtype)

    for x, y in zip(model.named_parameters(), my_model.named_parameters()):
        assert x[0] == y[0]
        name = x[0]
        delta = (x[1] - y[1]).abs().max()
        assert delta < 1e-5, f"max delta: {delta} for {name}"

    inputs = tokenizer(aprompt, return_tensors="pt").to(model.device)
    gold = run_inference(model, inputs)
    out = run_my_inference(my_model, inputs['input_ids'])

    delta = (gold - out).abs().max()
    print(f"max delta: {delta}")


if __name__ == "__main__":
    main()
