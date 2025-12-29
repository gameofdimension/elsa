import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models import Qwen3ForCausalLM

from elsa.profiler import profile_function
from prompt import text as aprompt


def load_model():
    model_path = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda()
    return tokenizer, model


@profile_function()
def run_inference(model: Qwen3ForCausalLM, tokenizer, prompt):

    max_new_tokens = 5
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    num_beams = 20
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_beams,
    )
    input_len = inputs["input_ids"].shape[-1]
    response = tokenizer.batch_decode(outputs[..., input_len:])
    return input_len, outputs, response


def main():
    count = int(sys.argv[1])
    tokenizer, model = load_model()

    run_inference(model, tokenizer, aprompt)

    begin = time.time()
    for i in range(count):
        input_len, outputs, response = run_inference(model, tokenizer, aprompt)
        output_len = outputs.shape[-1] - input_len
        print(i, len(outputs), input_len, output_len)
    end = time.time()

    print(f"generate count {count}, mean time {(end - begin) / count:.3f}")


if __name__ == "__main__":
    main()
