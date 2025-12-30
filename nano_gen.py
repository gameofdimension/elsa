import sys
import time
from nanovllm import LLM, SamplingParams

from prompt import text as aprompt


def main(ckpt):
    count = int(sys.argv[1])
    llm = LLM(ckpt, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=5, ignore_eos=True)
    n = 20
    prompt_lst = [aprompt for _ in range(n)]
    params_lst = [sampling_params for _ in range(n)]
    outputs = llm.generate(prompt_lst, params_lst)
    # outputs[0]["text"]

    begin = time.time()
    for i in range(count):
        outputs = llm.generate(prompt_lst, params_lst)
        print(i, len(outputs), len(outputs[0]["token_ids"]))
    end = time.time()
    print(f"generate count {count}, mean time {(end - begin) / count:.3f}")


if __name__ == '__main__':
    ckpt = "Qwen/Qwen3-1.7B"
    main(ckpt)
