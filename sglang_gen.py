import sys
import time

import sglang as sgl
from sglang.srt.entrypoints.engine import Engine

from prompt import text as aprompt


def main():
    count = int(sys.argv[1])
    attention_backend = "flashinfer"  # "flashinfer" # "fa3"
    llm: Engine = sgl.Engine(model_path="Qwen/Qwen3-1.7B", attention_backend=attention_backend)
    params = {"n": 20, "max_new_tokens": 5}
    outputs = llm.generate(aprompt, sampling_params=params)

    begin = time.time()
    for i in range(count):
        outputs = llm.generate(aprompt[i:], sampling_params=params)
        e2e_latency = outputs[0]['meta_info']['e2e_latency']
        prompt_tokens = outputs[0]['meta_info']["prompt_tokens"]
        completion_tokens = outputs[0]['meta_info']["completion_tokens"]
        print(i, len(outputs), prompt_tokens, completion_tokens, f"{e2e_latency:.3f}")
    end = time.time()

    print(f"generate count {count}, mean time {(end - begin) / count:.3f}")


if __name__ == '__main__':
    main()

